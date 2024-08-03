pip install streamlit yfinance pandas numpy scikit-learn keras matplotlib arch

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to load data
def load_data(ticker=None, file=None):
    if ticker:
        data = yf.download(ticker, period="5y")
    elif file:
        data = pd.read_csv(file)
    else:
        return None
    return data

# Function to calculate volatility using GARCH
def calculate_garch(data):
    returns = data.pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    volatility = garch_fit.conditional_volatility
    return volatility

# Function to predict momentum using LSTM
def predict_momentum(data):
    values = data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 10
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=50, batch_size=64, verbose=0)

    train_predict = model.predict(X)
    train_predict = scaler.inverse_transform(train_predict)
    return train_predict

# Streamlit UI
st.title("Time Series Analysis with GARCH and LSTM")

option = st.radio("Select Input Type", ('Yahoo Finance Ticker', 'CSV File'))

if option == 'Yahoo Finance Ticker':
    ticker = st.text_input("Enter Ticker Symbol", "AAPL")
    data_load_state = st.text("Loading data...")
    data = load_data(ticker=ticker)
    data_load_state.text("Loading data...done!")

elif option == 'CSV File':
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data_load_state = st.text("Loading data...")
        data = load_data(file=uploaded_file)
        data_load_state.text("Loading data...done!")

if 'data' in locals():
    st.subheader("Raw Data")
    st.write(data.tail())

    close_prices = data['Close']
    
    st.subheader("Volatility (GARCH)")
    volatility = calculate_garch(close_prices)
    st.line_chart(volatility)

    st.subheader("Momentum (LSTM)")
    momentum = predict_momentum(close_prices)
    momentum_series = pd.Series(momentum.flatten(), index=close_prices.index[10+1:len(momentum)+10+1])
    st.line_chart(momentum_series)

    st.subheader("Close Prices with Momentum")
    plt.figure(figsize=(14,7))
    plt.plot(close_prices, label='Close Prices')
    plt.plot(momentum_series, label='Momentum Predictions', alpha=0.7)
    plt.legend()
    st.pyplot(plt)
