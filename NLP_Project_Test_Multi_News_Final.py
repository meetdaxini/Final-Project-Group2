import torch
import pandas as pd
from datasets import load_dataset
from evaluate import load
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
model.load_state_dict(torch.load('best_model_Multi_News_final.pt'))
model.to(device)

# Load test data and preprocess
df_test = pd.read_csv('multi_label_test.csv')

def preprocess_data_test(df, tokenizer, max_length_x=1024, max_length_y=200):
    df['article'] = df['article'].apply(lambda x: ' '.join(x.split()[:max_length_x]))
    df['summary'] = df['summary'].apply(lambda x: ' '.join(x.split()[:max_length_y]))
    x = df['article'].tolist()
    y = df['summary'].tolist()
    encodings = tokenizer(x, truncation=True, padding=True, max_length=max_length_x, return_tensors='pt')
    labels_encodings = tokenizer(y, truncation=True, padding=True, max_length=max_length_y, return_tensors='pt')
    return encodings, labels_encodings

test_encodings, test_labels_encodings = preprocess_data_test(df_test[ :10], tokenizer)
test_input_ids = test_encodings['input_ids'].to(device)
test_attention_mask = test_encodings['attention_mask'].to(device)
test_labels = test_labels_encodings['input_ids'].to(device)

# Create DataLoader for test data
batch_size = 7
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Test function
def test_holdout(model, loader):
    model.eval()
    generated_summaries = []
    gold_summaries = []
    for batch in loader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            output = model.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=100,
                num_beams=4,
                decoder_start_token_id=model.config.eos_token_id,
                eos_token_id=model.config.eos_token_id
            )
        generated_summaries_holdout = tokenizer.batch_decode(output, skip_special_tokens=True)
        generated_summaries.extend(generated_summaries_holdout)
        gold_summaries.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
    return generated_summaries, gold_summaries

# Evaluate on test set
generated_summaries_holdout, gold_summaries_holdout = test_holdout(model, test_loader)

# Calculate ROUGE scores
def calculate_rouge(generated_summaries, gold_summaries):
    metric = load("rouge")
    rouge_scores = metric.compute(predictions=generated_summaries, references=gold_summaries)
    return rouge_scores
rouge_scores = calculate_rouge(generated_summaries_holdout, gold_summaries_holdout)

# Print ROUGE scores
print("ROUGE-1:", rouge_scores['rouge1'])
print("ROUGE-2:", rouge_scores['rouge2'])
print("ROUGE-L:", rouge_scores['rougeL'])



