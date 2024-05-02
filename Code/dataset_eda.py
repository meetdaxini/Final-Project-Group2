# %%
from datasets import load_dataset

# %%
cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
multi_news_dataset = load_dataset("multi_news")

# %%
cnn_dailymail_df_train =  cnn_dailymail_dataset["train"].to_pandas()

# %%
multi_news_dataset_df_train = multi_news_dataset["train"].to_pandas()

# %%
print(cnn_dailymail_df_train.iloc[1]["article"])

# %%
print(cnn_dailymail_df_train.iloc[1]["highlights"])

# %%
print(multi_news_dataset_df_train.iloc[1]["document"])

# %%
print(multi_news_dataset_df_train.iloc[1]["summary"])

# %%
cnn_dailymail_dataset["train"][0]["article"]

# %%
multi_news_dataset

# %%
def calculate_length(text):
    return len(text.split())

# %%
cnn_dailymail_df_train

# %%
avg_article_length_cnn = cnn_dailymail_df_train["article"].map(lambda x: calculate_length(x)).mean()
print("CNN/DailyMail - Average words in Article :", avg_article_length_cnn)

# %%
avg_summary_length_cnn = cnn_dailymail_df_train["highlights"].map(lambda x: calculate_length(x)).mean()
print("CNN/DailyMail - Average words in Summary:", avg_summary_length_cnn)

# %%
avg_article_length_multi_news = multi_news_dataset_df_train["document"].map(lambda x: calculate_length(x)).mean()
print("Multi News - Average words in Article:", avg_article_length_multi_news)

# %%
avg_summary_length_multi_new = multi_news_dataset_df_train["summary"].map(lambda x: calculate_length(x)).mean()
print("Multi News - Average words in Summary:", avg_summary_length_multi_new)

# %%



