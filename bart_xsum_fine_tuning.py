# %%
import transformers
from datasets import load_dataset, load_metric
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch

# %%
cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
multi_news_dataset = load_dataset("multi_news")
metric = load_metric('rouge')
model_name = 'facebook/bart-large-xsum'

# %%
max_input = 512
max_target = 128
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)

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

tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

# %% [markdown]
# ## If memory problems
# 
# - sample data to smaller sizes

# %%
#sample the data
train_sample = tokenized_dataset['train'].shuffle(seed=42).select(range(30000))
validation_sample = tokenized_dataset['validation'].shuffle(seed=42).select(range(6000))
test_sample = tokenized_dataset['test'].shuffle(seed=42).select(range(6000))

# %%
tokenized_dataset['train'] = train_sample
tokenized_dataset['validation'] = validation_sample
tokenized_dataset['test'] = test_sample

# %%
tokenized_dataset

# %% [markdown]
# ## Training process

# %%
#load model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

# %% [markdown]
# Depending on computing power, batch size can go as low as 1 if necessary

# %%
batch_size = 12

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
    'summary-combined',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size= batch_size,
    gradient_accumulation_steps=20,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    eval_accumulation_steps=1,
    fp16=True
    )

# %%
trainer = transformers.Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge
)

# %%
trainer.train()

# Testing the fine tuned model

# %%
article = """
LOS ANGELES (AP) — In her first interview since the NBA banned her estranged husband, Shelly Sterling says she will fight to keep her share of the Los Angeles Clippers and plans one day to divorce Donald Sterling. 
 
 (Click Prev or Next to continue viewing images.) 
 
 ADVERTISEMENT (Click Prev or Next to continue viewing images.) 
 
 Los Angeles Clippers co-owner Shelly Sterling, below, watches the Clippers play the Oklahoma City Thunder along with her attorney, Pierce O'Donnell, in the first half of Game 3 of the Western Conference... (Associated Press) 
 
 Shelly Sterling spoke to Barbara Walters, and ABC News posted a short story with excerpts from the conversation Sunday. 
 
 NBA Commissioner Adam Silver has banned Donald Sterling for making racist comments and urged owners to force Sterling to sell the team. Silver added that no decisions had been made about the rest of Sterling's family. 
 
 According to ABC's story, Shelly Sterling told Walters: "I will fight that decision." 
 
 Sterling also said that she "eventually" will divorce her husband, and that she hadn't yet done so due to financial considerations. ||||| Shelly Sterling said today that "eventually, I am going to" divorce her estranged husband, Donald Sterling, and if the NBA tries to force her to sell her half of the Los Angeles Clippers, she would "absolutely" fight to keep her stake in the team. 
 
 "I will fight that decision," she told ABC News' Barbara Walters today in an exclusive interview. "To be honest with you, I'm wondering if a wife of one of the owners, and there's 30 owners, did something like that, said those racial slurs, would they oust the husband? Or would they leave the husband in?" 
 
 Sterling added that the Clippers franchise is her "passion" and "legacy to my family." 
 
 "I've been with the team for 33 years, through the good times and the bad times," she added. 
 
 These comments come nearly two weeks after NBA Commissioner Adam Silver announced a lifetime ban and a $2.5 million fine for Donald Sterling on April 29, following racist comments from the 80-year-old, which were caught on tape and released to the media. 
 
 Read: Barbara Walters' Exclusive Interview With V. Stiviano 
 
 Being estranged from her husband, Shelly Sterling said she would "have to accept" whatever punishment the NBA handed down to him, but that her stake in the team should be separate. 
 
 "I was shocked by what he said. And -- well, I guess whatever their decision is -- we have to live with it," she said. "But I don't know why I should be punished for what his actions were." 
 
 An NBA spokesman said this evening that league rules would not allow her tol hold on to her share. 
 
 "Under the NBA Constitution, if a controlling owner's interest is terminated by a 3/4 vote, all other team owners' interests are automatically terminated as well," NBA spokesman Mike Bass said. "It doesn't matter whether the owners are related as is the case here. These are the rules to which all NBA owners agreed to as a condition of owning their team." 
 
 Sherry Sterling's lawyer, Pierce O'Donnell, disputed the league's reading of its constitution. 
 
 "We do not agree with the league's self-serving interpretation of its constitution, its application to Shelly Sterling or its validity under these unique circumstances," O'Donnell said in a statement released this evening in reposnse the NBA. "We live in a nation of laws. California law and the United States Constitution trump any such interpretation." 
 
 If the league decides to force Donald Sterling to sell his half of the team, Shelly Sterling doesn't know what he will do, but the possibility of him transferring full ownership to her is something she "would love him to" consider. 
 
 Related: NBA Bans Clippers Owner Donald Sterling For Life 
 
 "I haven't discussed it with him or talked to him about it," she said. 
 
 The lack of communication between Rochelle and Donald Sterling led Walters to question whether she plans to file for divorce. 
 
 "For the last 20 years, I've been seeing attorneys for a divorce," she said, laughing. "In fact, I have here-- I just filed-- I was going to file the petition. I signed the petition for a divorce. And it came to almost being filed. And then, my financial advisor and my attorney said to me, 'Not now.'" 
 
 Sterling added that she thinks the stalling of the divorce stems from "financial arrangements." 
 
 But she said "Eventually, I'm going to." 
 
 She also told Walters she thinks her estranged husband is suffering from "the onset of dementia." 
 
 Since Donald Sterling's ban, several celebrities have said they would be willing to buy the team from Sterling, including Oprah Winfrey and Magic Johnson. Sterling remains the owner, though his ban means he can have nothing to do with running the team and can't attend any games. 
 
 Silver announced Friday that former Citigroup chairman and former Time Warner chairman Richard Parsons has been named interim CEO of the team, but nothing concrete in terms of ownership or whether Sterling will be forced to sell the team. Parsons will now take over the basic daily operations for the team and oversee the team's president. 
 
 Read: What You Need to Know This Week About Donald Sterling 
 
 ABC News contacted Donald Sterling for comment on his wife's interview, but he declined.
"""

# %%
model_inputs = tokenizer(article,  max_length=max_input, padding='max_length', truncation=True)

# %%
model_inputs

# %%
raw_pred, _, _ = trainer.predict([model_inputs])

# %%
raw_pred

# %%
tokenizer.decode(raw_pred[0])

# %%
tokenizer.decode(raw_pred[0])

# %%
# Test the model
test_results = trainer.evaluate(tokenized_dataset['test'])
print("Test results:", test_results)

# %%
model_output_dir = 'summary-combined/final_model'
trainer.save_model(model_output_dir)


