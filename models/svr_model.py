import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_svr(df, window=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = SVR(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    preds_inv = scaler.inverse_transform(preds.reshape(-1,1))

    mse = mean_squared_error(y_test_inv, preds_inv)
    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mse)

    last_window = scaled[-window:].reshape(1, window)
    tomorrow = scaler.inverse_transform(
        model.predict(last_window).reshape(-1,1)
    )[0][0]

    return mse, mae, rmse, preds_inv, y_test_inv, tomorrow
