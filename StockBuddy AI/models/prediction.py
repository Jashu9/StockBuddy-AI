import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Assuming df is your DataFrame with the stock data
# Check if df is sorted by Date
df = df.sort_index()

# Prepare data: Use Close prices for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_price'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create sequences for LSTM: Using the last 60 days to predict the next day
X, y = [], []
for i in range(60, len(df)):
    X.append(df['scaled_price'][i-60:i].values)  # Last 60 days' scaled price
    y.append(df['scaled_price'][i])  # The next day's scaled price (target)

X, y = np.array(X), np.array(y)

# Reshape X to be 3D for LSTM: (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile & Train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Predict next value: Use the last sequence from the dataset
predicted_price = model.predict(X[-1].reshape(1, X.shape[1], 1))

# Inverse transform to get actual predicted price
predicted_price = scaler.inverse_transform(predicted_price)

# Display the predicted price
print("Predicted Price:", predicted_price[0][0])

train_size = int(len(df) * 0.8)  # 80% for training
train_data, test_data = df[:train_size], df[train_size:]

# Prepare training sequences
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data['scaled_price'][i-60:i].values)
    y_train.append(train_data['scaled_price'][i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Prepare testing sequences (same approach)
X_test, y_test = [], []
for i in range(60, len(test_data)):
    X_test.append(test_data['scaled_price'][i-60:i].values)
    y_test.append(test_data['scaled_price'][i])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape X_test for LSTM
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile & Train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict on the test set
predicted_prices = model.predict(X_test)

# Inverse transform the predicted prices to original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reverse scaling on actual prices

from sklearn.metrics import mean_squared_error
import math

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
print("Root Mean Squared Error (RMSE):", rmse)

import matplotlib.pyplot as plt

# Plot actual vs predicted prices
plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# Define a Reader and load DataFrame into Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "stock_symbol", "rating"]], reader)

# Train/Test Split
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

# Predict for a user
user_id = 123  # Replace with actual user ID
stock_symbol = "AAPL"
prediction = model.predict(user_id, stock_symbol)
print(f"Predicted rating for {stock_symbol}: {prediction.est}")