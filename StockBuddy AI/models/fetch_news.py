import yfinance as yf
import pandas as pd

def get_stock_news_yfinance(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news  # Fetch news articles
    articles = []

    for item in news:
        content = item.get("content", {})  # Get nested content dictionary
        # Fix the issue by checking if "clickThroughUrl" exists before accessing "url"
        articles.append({
            "title": content.get("title", "No Title"),
            "link": content.get("clickThroughUrl", {}).get("url", "No Link") if content.get("clickThroughUrl") else "No Link",
            "publisher": content.get("provider", {}).get("displayName", "Unknown"),
            "pubDate": content.get("pubDate", "No Date"),
        })


    return pd.DataFrame(articles)

# Example Usage
news_df = get_stock_news_yfinance("AAPL")
print(news_df.head())
