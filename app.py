import streamlit as st
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt

from utils.data_loader import load_stock_data
from models.svr_model import run_svr
from models.lstm_model import run_lstm
from models.gru_model import run_gru
from models.arima_model import run_arima
from models.sarimax_model import run_sarimax

st.title("📈 Multi-Model Stock Prediction System")

company = st.selectbox("Select Company", ["AAPL","GOOGL","MSFT","AMZN","TSLA"])

df = load_stock_data(company)

results = {}

svr = run_svr(df)
lstm = run_lstm(df)
gru = run_gru(df)
arima = run_arima(df['Close'])
sarimax = run_sarimax(df['Close'])

results["SVR"] = svr[:3]
results["LSTM"] = lstm[:3]
results["GRU"] = gru[:3]
results["ARIMA"] = arima[:3]
results["SARIMAX"] = sarimax[:3]

metrics_df = pd.DataFrame(results, index=["MSE","MAE","RMSE"]).T
st.subheader("📊 Error Comparison")
st.dataframe(metrics_df)

best_model = metrics_df["RMSE"].idxmin()
st.success(f"✅ Best Model: {best_model}")

model_map = {
    "SVR": svr,
    "LSTM": lstm,
    "GRU": gru,
    "ARIMA": arima,
    "SARIMAX": sarimax
}

preds, actuals, tomorrow_price = model_map[best_model][3:]

st.subheader("📉 Best Model Prediction Graph")
fig, ax = plt.subplots()
ax.plot(actuals, label="Actual")
ax.plot(preds, label="Predicted")
ax.legend()
st.pyplot(fig)

today_price = float(df['Close'].iloc[-1].values[0])


st.subheader("📅 Today & Tomorrow Price")
st.table(pd.DataFrame({
    "Date":[date.today(), date.today()+timedelta(days=1)],
    "Price":[today_price, tomorrow_price]
}))
