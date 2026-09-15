import pandas as pd

df = pd.read_csv("psx_extended_17.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

# Recalculate pct_change (needed to detect splits)
df["pct_change"] = df.groupby("SYMBOL")["CLOSE"].pct_change() * 100

# Detect splits: single-day drop greater than 40%
splits = df[df["pct_change"] < -40][["DATE", "SYMBOL", "pct_change"]]
print("Detected splits:")
print(splits)

# Columns that represent actual prices (need adjusting)
price_cols = ["LDCP", "OPEN", "HIGH", "LOW", "CLOSE"]

# Go through each detected split and fix the price history before it
for _, row in splits.iterrows():
    symbol = row["SYMBOL"]
    split_date = row["DATE"]
    pct = row["pct_change"]

    # Calculate the adjustment ratio
    ratio = 1 / (1 + pct / 100)

    # Mask: rows for this symbol, with dates BEFORE the split
    mask = (df["SYMBOL"] == symbol) & (df["DATE"] < split_date)

    # Divide all price columns by the ratio for those old rows
    df.loc[mask, price_cols] = df.loc[mask, price_cols] / ratio

    print(f"Adjusted {symbol}: split on {split_date.date()}, ratio = {ratio:.3f}")

# Recalculate pct_change after adjustment, to verify the fix worked
df["pct_change"] = df.groupby("SYMBOL")["CLOSE"].pct_change() * 100
remaining_big_drops = df[df["pct_change"] < -40]
print("\nAny big drops remaining after fix:")
print(remaining_big_drops[["DATE", "SYMBOL", "CLOSE", "pct_change"]])

# Save the split-adjusted dataset
df.to_csv("psx_extended_17_adjusted.csv", index=False)
print("\nSaved adjusted file!")