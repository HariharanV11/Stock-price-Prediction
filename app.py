import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Stock Price Prediction using SVR")

company = st.selectbox(
    "Select Company",
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="1y")
    df = df[['Close']]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

df = load_data(company)

# -------------------------------
# Prepare Data
# -------------------------------
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(df[['Close']])

X, y = [], []
window_size = 5

for i in range(window_size, len(scaled_prices)):
    X.append(scaled_prices[i - window_size:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# SVR Model
# -------------------------------
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse scaling
train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------------------
# Plot (UNCHANGED)
# -------------------------------
st.subheader("📊 Training & Prediction Graph")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df.index[window_size:split + window_size], y_train_actual, label="Train Actual")
ax.plot(df.index[window_size:split + window_size], train_pred, label="Train Prediction")

ax.plot(df.index[split + window_size:], y_test_actual, label="Test Actual")
ax.plot(df.index[split + window_size:], test_pred, label="Test Prediction")

ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)

# -------------------------------
# TODAY & TOMORROW (SVR)
# -------------------------------
# Calendar dates
today_date = date.today()
tomorrow_date = today_date + timedelta(days=1)

# Last actual price (clean float)
today_price = float(df['Close'].iloc[-1])

# Predict tomorrow using last window
last_window = scaled_prices[-window_size:].reshape(1, window_size)
tomorrow_scaled = model.predict(last_window)
tomorrow_price = float(scaler.inverse_transform(tomorrow_scaled.reshape(-1, 1))[0][0])

# -------------------------------
# Display Stock Data (ONLY 2 ROWS)
# -------------------------------
st.subheader("📅 Stock Data (Today & Tomorrow)")

stock_data = pd.DataFrame({
    "Date": [today_date, tomorrow_date],
    "Close Price": [today_price, tomorrow_price]
})

st.table(stock_data)
