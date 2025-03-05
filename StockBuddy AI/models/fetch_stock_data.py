import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Example Usage
df = fetch_stock_data('AAPL', '2023-01-01', '2025-03-01')

print(df.head())  # Check the output




