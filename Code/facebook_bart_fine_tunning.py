# %%
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import BartTokenizerFast
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer
from transformers import TrainingArguments

# %%
cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
multi_news_dataset = load_dataset("multi_news")
cnn_daily_mail = cnn_dailymail_dataset.map(lambda example: {'article': example['article'], 'summary': example['highlights']}, remove_columns=['highlights', 'id'])
multi_news = multi_news_dataset.map(lambda example: {'article': example['document'], 'summary': example['summary']}, remove_columns=['document'])

# %%
def concatenate_splits(dataset1, dataset2):
    return DatasetDict({
        split: concatenate_datasets([dataset1[split], dataset2[split]])
        for split in dataset1.keys()
    })

# %%
combined_dataset = concatenate_splits(cnn_daily_mail, multi_news)


# %%
combined_dataset["train"] = combined_dataset["train"].shuffle(seed=42)
combined_dataset["test"] = combined_dataset["test"].shuffle(seed=42)
combined_dataset["validation"] = combined_dataset["validation"].shuffle(seed=42)
# Print some information about the combined dataset
for split in combined_dataset.keys():
    print(f"Size of {split} split:", len(combined_dataset[split]))
    print(f"Example from {split} split:", combined_dataset[split][0])

# %%
max_input = 512
max_target = 128
batch_size = 3

# %%
model_name = "facebook/bart-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
tokenizer = BartTokenizerFast.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)

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
# def tokenize_function(examples):
#     return tokenizer(examples["article"], examples["summary"], truncation=True, padding="max_length", max_length=512)
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)


# %%

training_args = TrainingArguments(
    output_dir="output_facebook_bart",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_total_limit=2,
    weight_decay=0.01,
    eval_accumulation_steps=3,
    push_to_hub=False,
)

# %%

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

# %%
