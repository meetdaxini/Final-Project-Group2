import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=(dropout if n_layers > 1 else 0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=(dropout if n_layers > 1 else 0))
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Define the Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if torch.rand(1) < teacher_forcing_ratio else top1
        return outputs

#%%

import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

def extract_one_third_from_all(dataset):
    one_third_dataset = {}
    for split in ['train', 'validation', 'test']:
        num_samples = len(dataset[split]) // 50 # Calculate one-third of the original size
        part_to_return = dataset[split].shuffle(seed=42).select(range(num_samples))
        dataset[split] = part_to_return  # Store the one-third portion in a new dictionary
    return dataset

dataset = extract_one_third_from_all(dataset)

# To show the size of each split in the one-third dataset
print(len(dataset['train']))
print(len(dataset['validation']))
print(len(dataset['test']))


#%%

# Tokenizer and Vocabulary
tokenizer = get_tokenizer('basic_english')
def build_vocab(data_iter):
    token_stream = (token for item in data_iter for token in tokenizer(item['article']))
    vocab = build_vocab_from_iterator([token_stream], specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

vocab = build_vocab(dataset['train'])

class SummarizationDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = torch.tensor([self.vocab[token] for token in self.tokenizer(self.data[idx]['article'])], dtype=torch.long)
        tgt = torch.tensor([self.vocab[token] for token in self.tokenizer(self.data[idx]['highlights'])], dtype=torch.long)
        return src, tgt

# Prepare data
train_dataset = SummarizationDataset(dataset['train'], vocab, tokenizer)
val_dataset = SummarizationDataset(dataset['validation'], vocab, tokenizer)
test_dataset = SummarizationDataset(dataset['test'], vocab, tokenizer)

# DataLoader setup
PAD_IDX = vocab['<pad>']
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


# Adjust the batch size and accumulation steps as necessary
BATCH_SIZE = 12 # Reduced batch size
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)



# Model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(Encoder(len(vocab), 256, 800, 3, 0.3), Decoder(len(vocab), 256, 800, 3, 0.3), device).to(device)
# criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

# Usage
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)



#%%
def train_with_accumulation(model, dataloader, optimizer, criterion, clip, accumulation_steps):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()  # Move optimizer.zero_grad() outside the batch loop

    for batch_idx, (src, trg) in enumerate(dataloader):
        src = src.to(device)
        trg = trg.to(device)

        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss = loss / accumulation_steps  # Normalize loss to account for accumulation
        loss.backward()

        # Perform optimization step after 'accumulation_steps' batches
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps  # Scale up loss to the correct value

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0)  # Turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Train and evaluate the model with adjusted settings
N_EPOCHS = 10
CLIP = 1
ACCUMULATION_STEPS = 4  # Increase or decrease based on the GPU capacity

for epoch in range(N_EPOCHS):
    train_loss = train_with_accumulation(model, train_dataloader, optimizer, criterion, CLIP, ACCUMULATION_STEPS)
    valid_loss = evaluate(model, val_dataloader, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')

#%%

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

# Assuming optimizer is already defined
#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

def train_with_accumulation(model, dataloader, optimizer, criterion, clip, accumulation_steps, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', leave=False)
    for batch_idx, (src, trg) in progress_bar:
        src = src.to(device)
        trg = trg.to(device)

        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss = loss / accumulation_steps  # Adjust loss for the accumulation
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, total=len(dataloader), desc='Validating', leave=False)
        for src, trg in progress_bar:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, 0)  # No teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'val_loss': loss.item()})

    return epoch_loss / len(dataloader)

# Train and evaluate with learning rate adjustments based on validation loss
N_EPOCHS = 10
CLIP = 1
ACCUMULATION_STEPS = 1

for epoch in range(N_EPOCHS):
    start_time = time.time()
    teacher_forcing_ratio = max(0.5, 0.95 - 0.05 * epoch)

    train_loss = train_with_accumulation(model, train_dataloader, optimizer, criterion, CLIP, ACCUMULATION_STEPS, teacher_forcing_ratio)
    valid_loss = evaluate(model, val_dataloader, criterion)

    scheduler.step(valid_loss)  # Adjust the learning rate based on the validation loss

    end_time = time.time()
    epoch_duration = end_time - start_time
    current_lr = scheduler.optimizer.param_groups[0]['lr']


    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, LR: {current_lr:.6f}, Time: {epoch_duration:.2f}s')



#%%

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

# Initialize the scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=0.95)  # Decays the learning rate of each parameter group by gamma every step_size epochs

def train_with_accumulation(model, dataloader, optimizer, criterion, clip, accumulation_steps, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', leave=False)
    for batch_idx, (src, trg) in progress_bar:
        src = src.to(device)
        trg = trg.to(device)

        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

    return epoch_loss / len(dataloader)



def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, total=len(dataloader), desc='Validating', leave=False)
        for src, trg in progress_bar:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'val_loss': loss.item()})  # Update progress bar with validation loss

    return epoch_loss / len(dataloader)

# Training and evaluation with progress updates
N_EPOCHS = 10
CLIP = 1
ACCUMULATION_STEPS = 1

for epoch in range(N_EPOCHS):
    start_time = time.time()
    teacher_forcing_ratio = max(0.5, 0.95 - 0.05 * epoch)  # Decrease teacher forcing ratio over epochs

    train_loss = train_with_accumulation(model, train_dataloader, optimizer, criterion, CLIP, ACCUMULATION_STEPS, teacher_forcing_ratio)
    valid_loss = evaluate(model, val_dataloader, criterion)
    scheduler.step()  # Decrease the learning rate

    end_time = time.time()
    epoch_duration = end_time - start_time

    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_duration:.2f}s')


#%%

from tqdm import tqdm
import time

def train_with_accumulation(model, dataloader, optimizer, criterion, clip, accumulation_steps):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()  # Initialize gradients

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', leave=False)
    for batch_idx, (src, trg) in progress_bar:
        src = src.to(device)
        trg = trg.to(device)

        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss = loss / accumulation_steps  # Adjust loss for the accumulation
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})  # Update progress bar with live loss

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, total=len(dataloader), desc='Validating', leave=False)
        for src, trg in progress_bar:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'val_loss': loss.item()})  # Update progress bar with validation loss

    return epoch_loss / len(dataloader)


# Training and evaluation with progress updates
N_EPOCHS = 10
CLIP = 1
ACCUMULATION_STEPS = 4

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train_with_accumulation(model, train_dataloader, optimizer, criterion, CLIP, ACCUMULATION_STEPS)
    valid_loss = evaluate(model, val_dataloader, criterion)

    end_time = time.time()
    epoch_duration = end_time - start_time

    print(
        f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, Time: {epoch_duration:.2f}s')

#%%
# Save the model's state dictionary
torch.save(model.state_dict(), 'seq2seq_model.pth')

# Since `vocab` is a torchtext.vocab object, it's easiest to save the vocabulary using Python's pickle
import pickle
with open('vocab.pickle', 'wb') as f:
    pickle.dump(vocab, f)

#%%
# Load the model (ensure the model class is defined in the scope)
model = Seq2Seq(Encoder(len(vocab), 256, 512, 2, 0.5), Decoder(len(vocab), 256, 512, 2, 0.5), device)
model.load_state_dict(torch.load('seq2seq_model.pth'))
model.to(device)
model.eval()

# Load the vocabulary
with open('vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)



#%%
def summarize(model, text, tokenizer, vocab, device):
    tokens = ['<bos>'] + tokenizer(text) + ['<eos>']
    numericalized_tokens = [vocab.stoi[token] for token in tokens]
    tensor = torch.LongTensor(numericalized_tokens).unsqueeze(1).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor, tensor, 0)  # Since we do not have target, pass src as trg

    generated_tokens = []
    for i in range(1, outputs.size(0)):
        top1 = outputs[i].argmax(1).item()
        if top1 == vocab.stoi['<eos>']:
            break
        generated_tokens.append(vocab.itos[top1])

    return ' '.join(generated_tokens)

# Example usage: Generate summary for a sample text from the test dataset
sample_text = dataset['test'][0]['article']
summary = summarize(model, sample_text, tokenizer, vocab, device)
print("Actual Summary:", dataset['test'][0]['highlights'])
print("Generated Summary:", summary)
