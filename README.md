# NVIDIA Stock Price Prediction Models

This repository contains two different models for predicting NVIDIA (NVDA) stock prices using historical data: a Linear Regression model and a Long Short-Term Memory (LSTM) model. Both models utilize Python libraries such as `yfinance`, `pandas`, `scikit-learn`, and `Keras` for their implementation.

## Overview

This project aims to predict the future stock prices of NVIDIA using historical stock data. The first model implements a simple linear regression approach, while the second model leverages the LSTM architecture for time series forecasting. Both models are using historical prices and moving average as features to learn and model the historical prices.

### Features

- Download historical stock data from Yahoo Finance
- Implement a Linear Regression model for price prediction
- Implement an LSTM model for price prediction
- Visualize actual vs. predicted prices
- Calculate and display Mean Squared Error (MSE), Mean Absolute Error (MAE) and R-squared value for model evaluation 
- Use RSI, 30-day Moving Average and historical prices to help predict prices 

## Installation

To run the code, ensure you have Python installed. You can install the required libraries using pip:

```bash
pip install yfinance pandas numpy scikit-learn keras matplotlib 
