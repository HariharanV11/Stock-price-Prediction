from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def run_sarimax(df):
    split = int(len(df) * 0.8)
    train, test = df[:split], df[split:]

    model = SARIMAX(train, order=(5,1,0), seasonal_order=(1,1,1,12))
    fit = model.fit(disp=False)

    preds = fit.forecast(len(test))

    mse = mean_squared_error(test, preds)
    mae = mean_absolute_error(test, preds)
    rmse = np.sqrt(mse)

    tomorrow = fit.forecast(1).iloc[0]

    return mse, mae, rmse, preds.values, test.values, tomorrow
