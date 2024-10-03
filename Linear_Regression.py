import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Import and download Nvidia stock data since 2010 into CSV file
data = yf.download('NVDA', start='2010-01-01', end='2024-12-31')
data.to_csv('nvidia_stock_data.csv')
df = pd.read_csv('nvidia_stock_data.csv')
print(df.head())

# Drop rows with missing values
data.dropna(inplace=True)

# Technical Indicators
data['MA30'] = data['Close'].rolling(window=30).mean()  # 30-day moving average
data['Prev_Close'] = data['Close'].shift(1)

# RSI calculation
delta = data['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Drop rows with NaN values (due to rolling window)
data.dropna(inplace=True)

# Normalize data
scaler = MinMaxScaler()
data[['MA30', 'Prev_Close', 'RSI']] = scaler.fit_transform(data[['MA30', 'Prev_Close', 'RSI']])

# Define Features and Target
x = data[['MA30', 'Prev_Close', 'RSI']]
y = data[['Close']]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# Plotting the historical prices after preprocessing
data['Close'].plot(title='Nvidia Stock Price After Preprocessing')
plt.show()

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_predictions = lr_model.predict(x_test)

# Calculate metrics
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
print(f"Linear Regression MSE: {lr_mse}")
print(f"Linear Regression R-squared: {lr_r2}")
print(f"Linear Regression MAE: {lr_mae}")

# Convert y_test to a pandas series for plotting
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.squeeze()
elif y_test.ndim == 2:
    y_test = y_test.flatten()

y_test_series = pd.Series(y_test.values, index=y_test.index)
lr_predictions_series = pd.Series(lr_predictions.flatten(), index=y_test.index)

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test_series.index, y_test_series, label='Actual Prices', color='blue')
plt.plot(lr_predictions_series.index, lr_predictions_series, label='Linear Regression', color='red')
plt.title('Linear Regression Predictions vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# 1. Obtain the most recent data from your dataset
recent_data = data.iloc[-30:].copy()  # Get the last 30 rows of data to calculate MA30

# 2. Simulate future dates
future_dates = [data.index[-1] + pd.DateOffset(days=i) for i in range(1, 31)]  # Next 30 days
future_dates_df = pd.DataFrame(future_dates, columns=['Date'])
future_dates_df.set_index('Date', inplace=True)

# 3. Initialize the list for future predictions
future_predictions = []

# 4. Prepare data for future predictions
for i in range(30):
    # Update the 'Prev_Close' as the most recent 'Close' price or the last predicted value
    if i == 0:
        prev_close = recent_data['Close'].values[-1]  # Use the last known closing price
    else:
        prev_close = future_predictions[-1]  # Use the last predicted price

    # Calculate the 'MA30' including the last 29 known prices and the predicted price
    if i == 0:
        ma30 = recent_data['Close'].rolling(window=30).mean().values[-1]
    else:
        new_row = pd.DataFrame({'Close': [future_predictions[-1]]}, index=[future_dates[i-1]])
        recent_data = pd.concat([recent_data, new_row])
        ma30 = recent_data['Close'].rolling(window=30).mean().values[-1]

    # RSI for future predictions (assumed static in this basic example)
    rsi = data['RSI'].iloc[-1]

    # Create feature set for prediction
    feature_array = scaler.transform([[ma30, prev_close, rsi]])  # Normalize the features

    # Predict the future price
    prediction = lr_model.predict(feature_array)[0][0]  # Predict and get the value
    future_predictions.append(prediction)  # Add the prediction to the list

# Create DataFrame for future predictions
future_predictions_df = pd.DataFrame(future_predictions, columns=['Predicted_Close'], index=future_dates)

# Combine future dates and predictions
future_df = pd.concat([future_dates_df, future_predictions_df], axis=1)

# Display future predictions
print(future_df)

# Plotting the future predictions
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Historical Prices', color='blue')
plt.plot(future_df.index, future_df['Predicted_Close'], label='Future Predictions', color='orange')
plt.title('Historical Prices vs Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
