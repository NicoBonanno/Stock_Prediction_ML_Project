import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Download Nvidia stock data and save to CSV
data = yf.download('NVDA', start='2010-01-01', end='2024-12-31')
data.to_csv('nvidia_stock_data.csv')
df = pd.read_csv('nvidia_stock_data.csv')
print(df.head())

# Drop rows with missing values
data.dropna(inplace=True)

# Add technical indicators
data['MA30'] = data['Close'].rolling(window=30).mean()  # 30-day moving average
data['Prev_Close'] = data['Close'].shift(1)

# Drop rows with NaN values
data.dropna(inplace=True)

# Normalize data
scaler = MinMaxScaler()
data[['MA30', 'Prev_Close', 'Close']] = scaler.fit_transform(data[['MA30', 'Prev_Close', 'Close']])

# Define features and target
x = data[['MA30', 'Prev_Close']]
y = data[['Close']]

# Convert data into 3D format for LSTM (samples, timesteps, features)
x_train = []
y_train = []
time_steps = 60  # Using past 60 days to predict the next day's price

for i in range(time_steps, len(x)):
    x_train.append(x.iloc[i-time_steps:i].values)
    y_train.append(y.iloc[i].values)

x_train, y_train = np.array(x_train), np.array(y_train)

# Split into training and testing datasets
split_ratio = 0.8
split_index = int(len(x_train) * split_ratio)
x_train, x_test = x_train[:split_index], x_train[split_index:]
y_train, y_test = y_train[:split_index], y_train[split_index:]

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(x_train, y_train, epochs=10, batch_size=32)

# Make predictions
lstm_predictions = lstm_model.predict(x_test)

# Inverse scale predictions to get actual prices
y_test_rescaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :], y_test], axis=1))[:, -1]
lstm_predictions_rescaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :], lstm_predictions], axis=1))[:, -1]

# Evaluate the model
lstm_mse = mean_squared_error(y_test_rescaled, lstm_predictions_rescaled)
print(f"LSTM Mean Squared Error: {lstm_mse}")

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label='Actual Prices', color='blue')
plt.plot(lstm_predictions_rescaled, label='LSTM Predicted Prices', color='orange')
plt.title('LSTM Predictions vs Actual Prices')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
