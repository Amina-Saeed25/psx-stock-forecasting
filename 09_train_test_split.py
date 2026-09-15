import pandas as pd

df = pd.read_csv("psx_extended_17_with_returns.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

# Define the split date: everything before 2025 = train, 2025 onwards = test
split_date = pd.to_datetime("2025-01-01")

train = df[df["DATE"] < split_date].copy()
test = df[df["DATE"] >= split_date].copy()

print(f"Total rows: {len(df)}")
print(f"Train rows: {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"Test rows: {len(test)} ({len(test)/len(df)*100:.1f}%)")
print(f"\nTrain date range: {train['DATE'].min().date()} to {train['DATE'].max().date()}")
print(f"Test date range: {test['DATE'].min().date()} to {test['DATE'].max().date()}")

# Verify each company has data in both train and test
print("\nData distribution by company:")
for symbol in sorted(df["SYMBOL"].unique()):
    train_count = len(train[train["SYMBOL"] == symbol])
    test_count = len(test[test["SYMBOL"] == symbol])
    print(f"{symbol}: train={train_count}, test={test_count}")

# Save both
train.to_csv("train_data.csv", index=False)
test.to_csv("test_data.csv", index=False)
print("\nSaved train_data.csv and test_data.csv!")