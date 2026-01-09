import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_lstm(df, window=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(window,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    preds = model.predict(X_test)

    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inv, preds_inv)
    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mse)

    tomorrow = scaler.inverse_transform(
        model.predict(scaled[-window:].reshape(1,window,1))
    )[0][0]

    return mse, mae, rmse, preds_inv, y_test_inv, tomorrow
