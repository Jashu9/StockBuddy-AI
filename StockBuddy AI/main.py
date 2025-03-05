from flask import Flask, jsonify, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Prepare data for LSTM and prediction
def prepare_lstm_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['scaled_price'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Create sequences for LSTM
    X, y = [], []
    for i in range(60, len(df)):
        X.append(df['scaled_price'][i-60:i].values)
        y.append(df['scaled_price'][i])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Build and train LSTM model
def build_lstm_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    return model

@app.route('/predict', methods=['GET'])
def predict_stock_price():
    ticker = request.args.get('ticker', default='AAPL', type=str)

    # Fetch historical stock data
    df = fetch_stock_data(ticker, '2023-01-01', '2025-03-01')
    X, y, scaler = prepare_lstm_data(df)
    
    # Build and train LSTM model
    model = build_lstm_model(X, y)
    
    # Predict on the last sequence
    predicted_price = model.predict(X[-1].reshape(1, X.shape[1], 1))
    predicted_price = scaler.inverse_transform(predicted_price)

    # Prepare the response with the predicted stock price
    response = {
        'predicted_price': predicted_price[0][0]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
