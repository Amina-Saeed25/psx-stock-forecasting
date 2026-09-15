import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("psx_extended_17_with_returns.csv")
df["DATE"] = pd.to_datetime(df["DATE"])

# Test stationarity for each company's daily returns
for symbol in df["SYMBOL"].unique():
    sub = df[df["SYMBOL"] == symbol]["daily_return"].dropna()
    result = adfuller(sub)
    p_value = result[1]
    status = "Stationary" if p_value < 0.05 else "NOT Stationary"
    print(f"{symbol}: p-value = {p_value:.5f} -> {status}")