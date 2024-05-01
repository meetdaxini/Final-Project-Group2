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

#%%
import pandas as pd

# Convert DatasetDict to DataFrame
multi_df_train = pd.DataFrame(multi_news['train'])
multi_df_validation = pd.DataFrame(multi_news['validation'])
multi_df_test = pd.DataFrame(multi_news['test'])

cnn_df_train = pd.DataFrame(cnn_daily_mail['train'])
cnn_df_validation = pd.DataFrame(cnn_daily_mail['validation'])
cnn_df_test = pd.DataFrame(cnn_daily_mail['test'])

df_train = pd.DataFrame(combined_dataset['train'])
df_validation = pd.DataFrame(combined_dataset['validation'])
df_test = pd.DataFrame(combined_dataset['test'])

# %%

# Explore the training dataset
print("Training Dataset Info:")
print(df_train.info())

print("\nSummary Statistics for Training Dataset:")
print(df_train.describe())

print("\nSummary Statistics for cnn_df_train Training Dataset:")
print(cnn_df_train.describe())

print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())


# %%
# Find duplicate rows in the training dataset
duplicates = df_train[df_train.duplicated(keep=False)]

# Print duplicates and their corresponding actual rows
print("Duplicate Rows and Corresponding Actual Rows:")
for idx, row in duplicates.iterrows(5):
    actual_rows = df_train[(df_train['article'] == row['article']) & (df_train['summary'] == row['summary'])]
    print("Duplicate:")
    print(row)
    print("Actual:")
    print(actual_rows)
    print("="*50)

# %%
multi_df_train = multi_df_train.drop_duplicates(subset=['summary'])
multi_df_validation = multi_df_validation.drop_duplicates(subset=['summary'])
multi_df_test = multi_df_test.drop_duplicates(subset=['summary'])

cnn_df_train = cnn_df_train.drop_duplicates(subset=['summary'])
cnn_df_validation = cnn_df_validation.drop_duplicates(subset=['summary'])
cnn_df_test = cnn_df_test.drop_duplicates(subset=['summary'])

df_train = df_train.drop_duplicates(subset=['summary'])
df_validation = df_validation.drop_duplicates(subset=['summary'])
df_test = df_test.drop_duplicates(subset=['summary'])


#%%
multi_df_train = multi_df_train.drop_duplicates(subset=['article'])
multi_df_validation = multi_df_validation.drop_duplicates(subset=['article'])
multi_df_test = multi_df_test.drop_duplicates(subset=['article'])

cnn_df_train = cnn_df_train.drop_duplicates(subset=['article'])
cnn_df_validation = cnn_df_validation.drop_duplicates(subset=['article'])
cnn_df_test = cnn_df_test.drop_duplicates(subset=['article'])

df_train = df_train.drop_duplicates(subset=['article'])
df_validation = df_validation.drop_duplicates(subset=['article'])
df_test = df_test.drop_duplicates(subset=['article'])

#%%

def calculate_avg_tokens(dataset):

    # Calculate the word count for articles and summaries
    dataset['article_word_count'] = dataset['article'].apply(lambda x: len(x.split()))
    dataset['summary_word_count'] = dataset['summary'].apply(lambda x: len(x.split()))

    # Calculate average tokens
    avg_tokens_article = dataset['article_word_count'].mean()
    avg_tokens_summary = dataset['summary_word_count'].mean()

    return print("Average tokens in articles:", avg_tokens_article), print("Average tokens in summaries:", avg_tokens_summary)



#%%

# Calculate the word count for articles
def filter_by_word_count(dataset, column, max_word_count):

    # Calculate the word count for the specified column
    dataset['word_count'] = dataset[column].apply(lambda x: len(x.split()))

    # Create a new DataFrame removing rows with word count exceeding the maximum
    filtered_df = dataset[dataset['word_count'] <= max_word_count].copy()

    # Drop the temporary column used for word count
    filtered_df.drop(columns=['word_count'], inplace=True)

    return filtered_df

# %%
# combined news
df_train = filter_by_word_count(df_train, 'article', 5000)
print("Shape of the df_train filtered dataset:", df_train.shape)

df_validation = filter_by_word_count(df_validation, 'article', 5000)
print("Shape of the df_validation filtered dataset:", df_validation.shape)

df_test = filter_by_word_count(df_test, 'article', 5000)
print("Shape of the df_test filtered dataset:", df_test.shape)

#%%
# cnn news
cnn_df_train = filter_by_word_count(cnn_df_train, 'article', 5000)
print("Shape of the cnn_df_train filtered dataset:", cnn_df_train.shape)

cnn_df_validation = filter_by_word_count(cnn_df_validation, 'article', 5000)
print("Shape of the cnn_df_validation filtered dataset:", cnn_df_validation.shape)

cnn_df_test = filter_by_word_count(cnn_df_test, 'article', 5000)
print("Shape of the cnn_df_test filtered dataset:", cnn_df_test.shape)

#%%
# multi news
multi_df_train = filter_by_word_count(multi_df_train, 'article', 5000)
print("Shape of the multi_df_train filtered dataset:", multi_df_train.shape)

multi_df_validation = filter_by_word_count(multi_df_validation, 'article', 5000)
print("Shape of the multi_df_validation filtered dataset:", multi_df_validation.shape)

multi_df_test = filter_by_word_count(multi_df_test, 'article', 5000)
print("Shape of the multi_df_test filtered dataset:", multi_df_test.shape)


#%%

def remove_empty_rows(dataset):
    # Filter out rows with empty articles or summaries
    dataset = dataset[(dataset['article'] != '') & (dataset['summary'] != '')]
    return dataset

multi_df_train = remove_empty_rows(multi_df_train)
multi_df_validation = remove_empty_rows(multi_df_validation)
multi_df_test = remove_empty_rows(multi_df_test)

cnn_df_train = remove_empty_rows(cnn_df_train)
cnn_df_validation = remove_empty_rows(cnn_df_validation)
cnn_df_test = remove_empty_rows(cnn_df_test)

df_train = remove_empty_rows(df_train)
df_validation = remove_empty_rows(df_validation)
df_test = remove_empty_rows(df_test)

# %%
print("\nSummary Statistics for Training Dataset:")
print(df_train.describe())

print("\nSummary Statistics for cnn_df_train Training Dataset:")
print(cnn_df_train.describe())

print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())

#%%
print("multi_df_train")
calculate_avg_tokens(multi_df_train)
print("cnn_df_train")
calculate_avg_tokens(cnn_df_train)
print("df_train")
calculate_avg_tokens(df_train)

#%%
print("\nSummary Statistics for Training Dataset:")
print(df_train.describe())

print("\nSummary Statistics for cnn_df_train Training Dataset:")
print(cnn_df_train.describe())

print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())

#%%
multi_df_train = multi_df_train[['article', 'summary']]
cnn_df_train = cnn_df_train[['article', 'summary']]
df_train = df_train[['article', 'summary']]

#%%
print("\nSummary Statistics for Training Dataset:")
print(df_train.describe())

print("\nSummary Statistics for cnn_df_train Training Dataset:")
print(cnn_df_train.describe())

print("\nSummary Statistics for multi_df_train Training Dataset:")
print(multi_df_train.describe())

#%%


