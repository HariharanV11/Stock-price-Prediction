import yfinance as yf
import pandas as pd

def load_stock_data(ticker):
    df = yf.download(ticker, period="2y")
    df = df[['Close']]
    df.dropna(inplace=True)
    return df
