# NLP News Summarization Project

## Overview

This project uses state-of-the-art NLP techniques for news summarization. Additionally, a Streamlit app is provided for easy interaction with the models.

## Installation

### Prerequisites

To get started, you need to install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Streamlit App

To run the Streamlit app:

1. Navigate to the application directory:

```bash
cd Code/app
```

2. Download the fine-tuned models:

```bash
wget https://storage.googleapis.com/nlp-grp-2-bucket/best_model_Multi_News_final.pt
wget https://storage.googleapis.com/nlp-grp-2-bucket/bart-large-xsum-cnn_daily_final.zip
unzip bart-large-xsum-cnn_daily_final.zip
```

3. Update the News API key in `Code/app/utils.py` on line 7.
   You can get a free API key from [newsapi.org](https://newsapi.org/).

4. Start the Streamlit server:

```bash
python3 -m streamlit run news_summarization_app.py --server.port=8888
```

## Fine-Tuning BART

### Fine-tuning `facebook/bart-large-xsum` on CNN/Daily Mail

To fine-tune the BART model on the CNN/Daily Mail dataset:

```bash
python3 Code/bart_xsum_fine_tuning.py
```

## Fine-Tuning BART on Multi News Dataset

### Step 1: Fine-tuning on Multi News

To effectively utilize the code for the `facebook/bart-large-xsum` model trained on the Multi News dataset, follow these steps:

1. Execute the `NLP_Project_Train_Multi_News_Final.py` script. This will generate two crucial artifacts:
   - `multi_label_test.csv`: This file serves as the test set for evaluation purposes.
   - `best_model_Multi_News_final.pt`: This file represents the optimal model trained on the Multi News dataset.

### Step 2: Testing the Model

To test the BART model trained on the Multi News dataset:

1. Run the `NLP_Project_Test_Multi_News_Final.py` script.
2. Use the `multi_label_test.csv` dataset as the testing set.
3. Validate the performance of the model using `best_model_Multi_News_final.pt`.

### Step 3: Generating Text

To generate text using the BART model trained on the Multi News dataset:

1. Run the `Generating_Text_Multi_News_Final.py` script.
2. Use `best_model_Multi_News_final.pt` to execute the desired functionality.

## LLaMA 3 to Generate News Summary

To generate a news summary using LLaMA 3, use `llama.ipynb` in a Jupyter notebook.
Just update the `access_token` with your Hugging Face access token, which you can get from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
