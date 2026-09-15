import pandas as pd

df = pd.read_csv("psx_extended_17_adjusted.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

# Calculate daily returns (% change) for each company separately
df["daily_return"] = df.groupby("SYMBOL")["CLOSE"].pct_change() * 100

# Quick look: average and volatility (std deviation) of returns per company
summary = df.groupby("SYMBOL")["daily_return"].agg(["mean", "std"])
summary = summary.sort_values("std", ascending=False)
print(summary)

# Save this version with returns included
df.to_csv("psx_extended_17_with_returns.csv", index=False)
print("\nSaved!")