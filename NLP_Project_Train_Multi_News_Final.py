import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict
from evaluate import load
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
multi_news_dataset = load_dataset("multi_news")
multi_news = multi_news_dataset.map(lambda example: {'article': example['document'], 'summary': example['summary']}, remove_columns=['document'])
multi_df_train = pd.DataFrame(multi_news['train'])
multi_df_validation = pd.DataFrame(multi_news['validation'])
multi_df_test = pd.DataFrame(multi_news['test'])
print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())
multi_df_train = multi_df_train.drop_duplicates(subset=['summary'])
multi_df_validation = multi_df_validation.drop_duplicates(subset=['summary'])
multi_df_test = multi_df_test.drop_duplicates(subset=['summary'])
multi_df_train = multi_df_train.drop_duplicates(subset=['article'])
multi_df_validation = multi_df_validation.drop_duplicates(subset=['article'])
multi_df_test = multi_df_test.drop_duplicates(subset=['article'])
def calculate_avg_tokens(dataset):
    # Calculate the word count for articles and summaries
    dataset['article_word_count'] = dataset['article'].apply(lambda x: len(x.split()))
    dataset['summary_word_count'] = dataset['summary'].apply(lambda x: len(x.split()))
    # Calculate average tokens
    avg_tokens_article = dataset['article_word_count'].mean()
    avg_tokens_summary = dataset['summary_word_count'].mean()
    return print("Average tokens in articles:", avg_tokens_article), print("Average tokens in summaries:", avg_tokens_summary)
def filter_by_word_count(dataset, column, max_word_count):
    dataset['word_count'] = dataset[column].apply(lambda x: len(x.split()))
    filtered_df = dataset[dataset['word_count'] <= max_word_count].copy()
    filtered_df.drop(columns=['word_count'], inplace=True)
    return filtered_df
multi_df_train = filter_by_word_count(multi_df_train, 'article', 5000)
print("Shape of the multi_df_train filtered dataset:", multi_df_train.shape)
multi_df_validation = filter_by_word_count(multi_df_validation, 'article', 5000)
print("Shape of the multi_df_validation filtered dataset:", multi_df_validation.shape)
multi_df_test = filter_by_word_count(multi_df_test, 'article', 5000)
print("Shape of the multi_df_test filtered dataset:", multi_df_test.shape)
def remove_empty_rows(dataset):
    # Filter out rows with empty articles or summaries
    dataset = dataset[(dataset['article'] != '') & (dataset['summary'] != '')]
    return dataset
multi_df_train = remove_empty_rows(multi_df_train)
multi_df_validation = remove_empty_rows(multi_df_validation)
multi_df_test = remove_empty_rows(multi_df_test)
print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())
print("multi_df_train")
calculate_avg_tokens(multi_df_train)
print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())
multi_df_train = multi_df_train[['article', 'summary']]
print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())
multi_df_test.to_csv('multi_label_test.csv', index=False)

# Bart model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum').to(device)

# Preprocess data
def preprocess_data(df, tokenizer, max_length_x=1024, max_length_y=200):
    df['article'] = df['article'].apply(lambda x: ' '.join(x.split()[:max_length_x]))
    df['summary'] = df['summary'].apply(lambda x: ' '.join(x.split()[:max_length_y]))
    x = df['article'].tolist()
    y = df['summary'].tolist()
    encodings = tokenizer(x, truncation=True, padding=True, max_length=max_length_x)
    labels_encodings = tokenizer(y, truncation=True, padding=True, max_length=max_length_y)
    return encodings, labels_encodings

train_encodings, train_labels_encodings = preprocess_data(multi_df_train, tokenizer)
val_encodings, val_labels_encodings = preprocess_data(pd.DataFrame(multi_df_validation), tokenizer)
test_encodings, test_labels_encodings = preprocess_data(multi_df_test, tokenizer)

# Convert to PyTorch tensors
def create_tensors(encodings, labels_encodings, device=device):
    input_ids = torch.tensor(encodings['input_ids']).to(device)
    attention_mask = torch.tensor(encodings['attention_mask']).to(device)
    labels = torch.tensor(labels_encodings['input_ids']).to(device)
    labels_attention_mask = torch.tensor(labels_encodings['attention_mask']).to(device)
    return input_ids, attention_mask, labels, labels_attention_mask

train_input_ids, train_attention_mask, train_labels, train_labels_attention_mask = create_tensors(train_encodings, train_labels_encodings, device)
val_input_ids, val_attention_mask, val_labels, val_labels_attention_mask = create_tensors(val_encodings, val_labels_encodings, device)
test_input_ids, test_attention_mask, test_labels, test_labels_attention_mask = create_tensors(test_encodings, test_labels_encodings, device)

# Create Tensor datasets and loaders
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels, train_labels_attention_mask)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels, val_labels_attention_mask)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels, test_labels_attention_mask)

batch_size = 7
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Hyperparameters
num_epochs = 5
learning_rate = 2e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Early Stopping Implementation
best_val_loss = float('inf')
patience = 3
epochs_without_improvement = 0

# Training loop with early stopping
def train(model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels, labels_attention_mask = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=labels_attention_mask)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_loader)

def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels, labels_attention_mask = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=labels_attention_mask)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_loader)

def test(model, loader):
    model.eval()
    generated_summaries = []
    gold_summaries = []
    for batch in loader:
        input_ids, attention_mask, labels, _ = batch
        with torch.no_grad():
            output = model.generate(input_ids.to(device), attention_mask=attention_mask.to(device), max_length=200,num_beams=4)
        generated_summary = tokenizer.batch_decode(output, skip_special_tokens=True)
        generated_summaries.extend(generated_summary)
        gold_summaries.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
    return generated_summaries, gold_summaries

def calculate_rouge(generated_summaries, gold_summaries):
    metric = load("rouge")
    rouge_scores = metric.compute(predictions=generated_summaries, references=gold_summaries)
    return rouge_scores

# Print table header
print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'ROUGE-1':^10} | {'ROUGE-2':^10} | {'ROUGE-L':^10}")

# Initialize DataFrame to store training results
training_results = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, scheduler)
    val_loss = validate(model, val_loader)
    generated_summaries, gold_summaries = test(model, val_loader)
    rouge_scores = calculate_rouge(generated_summaries, gold_summaries)
    print(f"{epoch + 1:^7} | {train_loss:^12.6f} | {val_loss:^10.6f} | {rouge_scores['rouge1']:^10.4f} | {rouge_scores['rouge2']:^10.4f} | {rouge_scores['rougeL']:^10.4f}")
    training_results.append({
        'Epoch': epoch + 1,
        'Train Loss': train_loss,
        'Val Loss': val_loss,
        'ROUGE-1': rouge_scores['rouge1'],
        'ROUGE-2': rouge_scores['rouge2'],
        'ROUGE-L': rouge_scores['rougeL']
    })
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model_Multi_News_final.pt')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

# Save training results to CSV
training_results_df = pd.DataFrame(training_results)
training_results_df.to_csv('training_results.csv', index=False)
