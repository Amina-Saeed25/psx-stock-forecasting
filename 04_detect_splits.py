import pandas as pd

df = pd.read_csv("psx_extended_17.csv")
df["DATE"] = pd.to_datetime(df["DATE"])

# Sort by symbol and date (important! so pct_change compares consecutive days correctly)
df = df.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

# Calculate daily % change in CLOSE price, separately for each company
df["pct_change"] = df.groupby("SYMBOL")["CLOSE"].pct_change() * 100

# Find days where price dropped more than 40% in a single day (likely a split, not a real crash)
possible_splits = df[df["pct_change"] < -40]
print(possible_splits[["DATE", "SYMBOL", "CLOSE", "pct_change"]])