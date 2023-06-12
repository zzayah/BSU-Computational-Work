import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Step 1: Data Collection
symbol = "AAPL"  # Stock symbol
start_date = "2015-01-01"  # Start date of data
end_date = "2022-01-01"  # End date of data

# Download stock data from Yahoo Finance using yfinance library
data = yf.download(symbol, start=start_date, end=end_date)

if data.empty:
    print("No data available for the specified stock symbol and date range.")
    exit()

# Step 2: Data Preprocessing
cleaned_df = data[['Close']].reset_index()
cleaned_df.columns = ['Date', 'Close']

# Normalize the stock prices
scaler = MinMaxScaler(feature_range=(0, 1))
cleaned_df['Close'] = scaler.fit_transform(cleaned_df[['Close']])

# Step 3: Feature Engineering
lookback = 10

# Create input features and target labels
X, y = [], []
for i in range(lookback, len(cleaned_df)):
    X.append(cleaned_df['Close'].values[i - lookback:i])
    y.append(cleaned_df['Close'].values[i])

if len(X) == 0 or len(y) == 0:
    print("Insufficient data for training the model.")
    exit()

# Convert the lists to numpy arrays
X, y = np.array(X), np.array(y)

# Step 4: Train-Validation-Test Split
train_size = int(0.6 * len(X))  # 60% of the data for training
val_size = int(0.2 * len(X))  # 20% of the data for validation
test_size = len(X) - train_size - val_size  # Remaining data for testing

if train_size == 0 or val_size == 0 or test_size == 0:
    print("Insufficient data for training, validation, or testing.")
    exit()

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Step 5: Model Creation and Training
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=256, return_sequences=True, input_shape=(lookback, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

# Step 6: Model Evaluation
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Step 7: Backtesting and Performance Analysis (Strategy Implementation)
y_pred = model.predict(X_test)

# Example: Calculate daily returns based on predicted prices
y_pred_prices = scaler.inverse_transform(y_pred)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

daily_returns = np.diff(y_pred_prices, axis=0) / y_pred_prices[:-1]
cumulative_returns = np.cumprod(1 + daily_returns) - 1

# Step 8: Visualization
date_range = cleaned_df['Date'][train_size + val_size + lookback + 1:]

plt.figure(figsize=(10, 6))
plt.plot(date_range, cumulative_returns)
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot MAPE over time
plt.figure(figsize=(8, 6))
plt.plot(date_range[:-1], mape)
plt.title('Mean Absolute Percentage Error (MAPE) over Time')
plt.xlabel('Date')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()