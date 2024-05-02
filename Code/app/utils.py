import json
import os
import re
import requests

base_url = "https://newsapi.org/v2/everything"
api_key = "6cbe9af59e314c74b353a23e9bc7c622"

def search_news(query, use_cache=False):
    cache_path = 'news_response_cache.json'
    if use_cache and os.path.exists(cache_path):
        print("Using cached data")
        with open(cache_path, 'r') as file:
            articles_json = json.load(file)
    else:
        params = {
            "q": query,
            "apiKey": api_key,
            "pageSize": 100,
            "language": "en"
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception("Fetching articles failed")
        articles_json = response.json()

        if use_cache:
            with open(cache_path, 'w') as file:
                json.dump(articles_json, file)
            print("Cached new data")

    articles = sorted(articles_json["articles"], key=lambda a: a["publishedAt"], reverse=True)
    return [
        article["url"] for article in articles if "https://removed.com" not in article["url"].lower() and "yahoo" not in article["url"].lower()
    ][:5]
    

def process_input(user_input):
    url_pattern = re.compile(r'https?://\S+')
    urls = url_pattern.findall(user_input)
    
    if urls:
        return "urls", urls
    else:
        return "query", user_input.strip()