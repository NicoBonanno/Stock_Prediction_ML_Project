import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
lstm_model.add(Dropout(0.2))  # Add dropout to reduce overfitting
lstm_model.add(LSTM(units=100))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

# Compile the model with Adam optimizer and learning rate scheduler
optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Implement learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

# Train the LSTM model
history = lstm_model.fit(x_train, y_train, epochs=20, batch_size=32, callbacks=[reduce_lr])

# Make predictions
lstm_predictions = lstm_model.predict(x_test)

# Inverse scale predictions to get actual prices
y_test_rescaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :], y_test], axis=1))[:, -1]
lstm_predictions_rescaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :], lstm_predictions], axis=1))[:, -1]

# Evaluate the model
lstm_mse = mean_squared_error(y_test_rescaled, lstm_predictions_rescaled)
lstm_mae = mean_absolute_error(y_test_rescaled, lstm_predictions_rescaled)
lstm_r2 = r2_score(y_test_rescaled, lstm_predictions_rescaled)

print(f"LSTM Mean Squared Error: {lstm_mse}")
print(f"LSTM Mean Absolute Error: {lstm_mae}")
print(f"LSTM R-squared Value: {lstm_r2}")

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label='Actual Prices', color='blue')
plt.plot(lstm_predictions_rescaled, label='LSTM Predicted Prices', color='orange')
plt.title('LSTM Predictions vs Actual Prices')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot loss over time
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
