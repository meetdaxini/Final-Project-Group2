# %%
import transformers
from datasets import load_dataset, load_metric
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import json
import os

output_dir = 'training_logs'
os.makedirs(output_dir, exist_ok=True)

class SaveResultsCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        epoch = state.epoch
        output_file = os.path.join(output_dir, f'results_epoch_{epoch}.json')
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
# %%
cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
metric = load_metric('rouge')
model_name = 'facebook/bart-large-xsum'

# %%
max_input = 1024
max_target = 200
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# %%
def tokenize_function(data_to_process):
  """
  https://medium.com/@ferlatti.aldo/fine-tuning-a-chat-summarizer-c18625bc817d
  """
  #get all the articles
  inputs = [article for article in data_to_process['article']]
  #tokenize the articles
  model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)
  #tokenize the summaries
  with tokenizer.as_target_tokenizer():
    targets = tokenizer(data_to_process['summary'], max_length=max_target, padding='max_length', truncation=True)

  #set labels
  model_inputs['labels'] = targets['input_ids']
  #return the tokenized data
  #input_ids, attention_mask and labels
  return model_inputs

# %%
# %%
cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
cnn_daily_mail = cnn_dailymail_dataset.map(lambda example: {'article': example['article'], 'summary': example['highlights']}, remove_columns=['highlights', 'id'])


cnn_df_train = pd.DataFrame(cnn_daily_mail['train'])
cnn_df_validation = pd.DataFrame(cnn_daily_mail['validation'])
cnn_df_test = pd.DataFrame(cnn_daily_mail['test'])

cnn_df_train = cnn_df_train.drop_duplicates(subset=['summary'])
cnn_df_validation = cnn_df_validation.drop_duplicates(subset=['summary'])
cnn_df_test = cnn_df_test.drop_duplicates(subset=['summary'])

cnn_df_train = cnn_df_train.drop_duplicates(subset=['article'])
cnn_df_validation = cnn_df_validation.drop_duplicates(subset=['article'])
cnn_df_test = cnn_df_test.drop_duplicates(subset=['article'])


def calculate_avg_tokens(dataset):

    # Calculate the word count for articles and summaries
    dataset['article_word_count'] = dataset['article'].apply(lambda x: len(x.split()))
    dataset['summary_word_count'] = dataset['summary'].apply(lambda x: len(x.split()))

    # Calculate average tokens
    avg_tokens_article = dataset['article_word_count'].mean()
    avg_tokens_summary = dataset['summary_word_count'].mean()

    return print("Average tokens in articles:", avg_tokens_article), print("Average tokens in summaries:", avg_tokens_summary)


# Calculate the word count for articles
def filter_by_word_count(dataset, column, max_word_count):

    # Calculate the word count for the specified column
    dataset['word_count'] = dataset[column].apply(lambda x: len(x.split()))

    # Create a new DataFrame removing rows with word count exceeding the maximum
    filtered_df = dataset[dataset['word_count'] <= max_word_count].copy()

    # Drop the temporary column used for word count
    filtered_df.drop(columns=['word_count'], inplace=True)

    return filtered_df



#%%
# cnn news
cnn_df_train = filter_by_word_count(cnn_df_train, 'article', 5000)
print("Shape of the cnn_df_train filtered dataset:", cnn_df_train.shape)

cnn_df_validation = filter_by_word_count(cnn_df_validation, 'article', 5000)
print("Shape of the cnn_df_validation filtered dataset:", cnn_df_validation.shape)

cnn_df_test = filter_by_word_count(cnn_df_test, 'article', 5000)
print("Shape of the cnn_df_test filtered dataset:", cnn_df_test.shape)

#%%
def remove_empty_rows(dataset):
    # Filter out rows with empty articles or summaries
    dataset = dataset[(dataset['article'] != '') & (dataset['summary'] != '')]
    return dataset


cnn_df_train = remove_empty_rows(cnn_df_train).reset_index()
cnn_df_validation = remove_empty_rows(cnn_df_validation).reset_index()
cnn_df_test = remove_empty_rows(cnn_df_test).reset_index()


# %%
print("\nSummary Statistics for cnn_df_train Training Dataset:")
print(cnn_df_train.describe())



#%%
print("cnn_df_train")
calculate_avg_tokens(cnn_df_train)

#%%

print("\nSummary Statistics for cnn_df_train Training Dataset:")
print(cnn_df_train.describe())
#%%
train_dataset = Dataset.from_pandas(cnn_df_train)
validation_dataset = Dataset.from_pandas(cnn_df_validation)
test_dataset = Dataset.from_pandas(cnn_df_test)

cnn_final_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})


#%%
cnn_final_dataset

#%%
tokenized_dataset = cnn_final_dataset.map(tokenize_function, batched=True)


# %%
#load model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

# %% [markdown]
# Depending on computing power, batch size can go as low as 1 if necessary

# %%
batch_size = 8

# %%
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

# %%
def compute_rouge(pred):
  predictions, labels = pred
  #decode the predictions
  decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  #decode labels
  decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  #compute results
  res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
  #get %
  res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

  pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  res['gen_len'] = np.mean(pred_lens)

  return {k: round(v, 4) for k, v in res.items()}

# %%
args = transformers.Seq2SeqTrainingArguments(
    'bart-large-xsum-cnn_daily_mail_plus_final',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=20,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    eval_accumulation_steps=1,
    fp16=True,
    save_strategy='epoch',
)

# %%
trainer = transformers.Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge,
    callbacks=[SaveResultsCallback()]
)

# %%
trainer.train()
# %%
# Test the model
test_results = trainer.evaluate(tokenized_dataset['test'])
print("Test results:", test_results)

# %%
model_output_dir = 'bart-large-xsum-cnn_daily_final'
trainer.save_model(model_output_dir)

