import streamlit as st
from summarization import generate_summary, load_tokenizer_and_model
from utils import search_news, process_input
import newspaper

device = 'cuda'
model_name = "facebook/bart-large-cnn"
tokenizer, model = load_tokenizer_and_model(model_name)

# st.set_page_config(page_title="News Summary", page_icon="📰", layout="centered")

st.title("📰 News Summary")
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
            st.write(f"Searching for news articles related to: {input_value}")
            with st.spinner("Searching for articles..."):
                input_value = search_news(input_value)

        st.write(f"Fetching news articles from {len(input_value)} URLs:")
        for url in input_value:
            st.write(f"- {url}")
        st.write("---")
        with st.spinner("Fetching articles..."):
            articles = [newspaper.article(url) for url in input_value]

        if articles:
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
                    summary = generate_summary(article.title + ' ' + article.text, tokenizer, model, 200, 50, length_penalty=3)
                st.write(f"Summary: {summary}")
                st.write("---")
        else:
            st.error(f"No articles found with your search query: {user_input}. Please try a different search query.")
    except Exception as e:
        print(e)
        st.error("Failed to fetch news articles. Please try again.")

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