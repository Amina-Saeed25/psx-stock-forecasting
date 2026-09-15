import pandas as pd
import matplotlib.pyplot as plt

# Load the SPLIT-ADJUSTED dataset (not the original one)
df = pd.read_csv("psx_extended_17_adjusted.csv")
df["DATE"] = pd.to_datetime(df["DATE"])

# Cement sector
cement = ["LUCK", "MLCF", "DGKC", "FCCL"]

plt.figure(figsize=(12, 6))
for symbol in cement:
    sub = df[df["SYMBOL"] == symbol]
    plt.plot(sub["DATE"], sub["CLOSE"], label=symbol)

plt.title("Cement Sector — Closing Price, Split-Adjusted (2017-2025)", fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Closing Price (PKR)")
plt.legend()
plt.savefig("cement_sector_adjusted.png")
plt.show()