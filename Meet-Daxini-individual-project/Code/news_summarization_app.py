import streamlit as st
from summarization import generate_summary
from utils import search_news, process_input
import newspaper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model_name = "bart-large-xsum-cnn_daily_final"
xsum_model_name = "facebook/bart-large-xsum"


@st.cache_resource
def load_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model

@st.cache_resource
def load_xsum_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.load_state_dict(torch.load('app/best_model_Multi_News_final.pt'))
    model.to(device)
    return tokenizer, model

st.set_page_config(page_title="News Summary", page_icon="📰", layout="centered")
st.title("📰 News Summary")
cnn_tokenizer, cnn_model = load_tokenizer_and_model(cnn_model_name)
xsum_tokenizer, xsum_model = load_xsum_tokenizer_and_model(xsum_model_name)



with st.sidebar:
    input_option = st.radio("Select:", ("Search Query or URL(s)", "Paste Text"))
    max_tokens = st.slider("Maximum Token Length", min_value=50, max_value=200, value=200)
    min_tokens = st.slider("Minimum Token Length", min_value=10, max_value=200, value=40)
    length_penalty = st.slider("Length Penalty", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

if min_tokens > max_tokens:
    min_tokens = max_tokens
    st.warning("Minimum token length cannot be greater than maximum token length. Setting minimum token length to the maximum value.")

if input_option == "Search Query or URL(s)":
    input_help = """
    You can enter one of the following:
    - A search query to search for news articles
    - A single URL of a news article
    - Multiple URLs of news articles separated by commas or spaces
    """
    user_input = st.text_input("Enter your input:", help=input_help)
    
    if user_input:
        input_type, input_value = process_input(user_input)
        try:
            if input_type == "query":
                with st.spinner(f"Searching for articles related to: {input_value}..."):
                    input_value = search_news(input_value)
            
            with st.spinner(f"Fetching news articles from {len(input_value)} URLs:" + "\n" + "\n".join([f"- {url}" for url in input_value])):
                articles = [newspaper.article(url) for url in input_value]
                st.write("---")
            if articles:
                article_contents = []
                for article in articles:
                    if article.top_image:
                        st.image(article.top_image, use_column_width=True)
                    st.subheader(article.title)
                    if article.authors:
                        author_label = "Author" if len(article.authors) == 1 else "Authors"
                        authors = ", ".join(article.authors)
                    else:
                        author_label = "Author"
                        authors = "Unknown"
                    publish_datetime = article.publish_date.strftime("%B %d, %Y at %H:%M") if article.publish_date else "Date Unknown"
                    st.write(f"{author_label}: {authors}")
                    st.write(f"Published: {publish_datetime}")
                    with st.expander("Read Entire Article..."):
                        st.write(article.text)
                    st.write(f"Article link: {article.url}")
                    
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(article.title + ' ' + article.text, cnn_tokenizer, cnn_model, max_tokens, min_tokens, length_penalty=length_penalty)
                    st.write(f"Summary: {summary}")
                    st.write("---")
                    article_contents.append(article.title + ' ' + article.text)

                if len(article_contents)>1:
                    all_article_contents = " ||||| ".join(article_contents)                
                    with st.spinner("Generating summary of all articles..."):
                        all_articles_summary = generate_summary(all_article_contents, xsum_tokenizer, xsum_model, max_tokens, min_tokens, length_penalty=length_penalty)
                    st.write("Summary of All Articles:")
                    st.write(all_articles_summary)
            else:
                st.error(f"No articles found with your search query: {user_input}. Please try a different search query.")
        except Exception as e:
            print(e)
            st.error("Failed to fetch news articles. Please try again.")
            
elif input_option == "Paste Text":
    user_text = st.text_area("Paste your text here:")
    
    if user_text:
        with st.spinner("Generating summary..."):
            summary = generate_summary(user_text, cnn_tokenizer, cnn_model, max_tokens, min_tokens, length_penalty=length_penalty)
        st.write(f"Summary: {summary}")

st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stTextInput input {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)