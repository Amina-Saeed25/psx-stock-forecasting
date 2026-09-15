import pandas as pd
import matplotlib.pyplot as plt

# Load the extended filtered dataset
df = pd.read_csv("psx_extended_17.csv")

# Convert DATE column from text to actual datetime format
df["DATE"] = pd.to_datetime(df["DATE"])



# Banking sector companies
banking = ["HBL", "UBL", "MCB", "MEBL"]

plt.figure(figsize=(12, 6))

for symbol in banking:
    sub = df[df["SYMBOL"] == symbol]
    plt.plot(sub["DATE"], sub["CLOSE"], label=symbol)

plt.title("Banking Sector — Closing Price (2017-2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (PKR)")
plt.legend()
plt.savefig("banking_sector_price.png")
plt.show()


# Cement sector
cement = ["LUCK", "MLCF", "DGKC", "FCCL"]

plt.figure(figsize=(12, 6))
for symbol in cement:
    sub = df[df["SYMBOL"] == symbol]
    plt.plot(sub["DATE"], sub["CLOSE"], label=symbol)
plt.title("Cement Sector — Closing Price (2017-2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (PKR)")
plt.legend()
plt.savefig("cement_sector_price.png")
plt.show()


# Oil & Gas sector
oil_gas = ["OGDC", "PPL", "POL", "PSO"]

plt.figure(figsize=(12, 6))
for symbol in oil_gas:
    sub = df[df["SYMBOL"] == symbol]
    plt.plot(sub["DATE"], sub["CLOSE"], label=symbol)
plt.title("Oil & Gas Sector — Closing Price (2017-2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (PKR)")
plt.legend()
plt.savefig("oil_gas_sector_price.png")
plt.show()


# Fertilizer sector
fertilizer = ["EFERT", "FFC"]

plt.figure(figsize=(12, 6))
for symbol in fertilizer:
    sub = df[df["SYMBOL"] == symbol]
    plt.plot(sub["DATE"], sub["CLOSE"], label=symbol)
plt.title("Fertilizer Sector — Closing Price (2017-2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (PKR)")
plt.legend()
plt.savefig("fertilizer_sector_price.png")
plt.show()


# Steel sector
steel = ["ASL", "ISL"]

plt.figure(figsize=(12, 6))
for symbol in steel:
    sub = df[df["SYMBOL"] == symbol]
    plt.plot(sub["DATE"], sub["CLOSE"], label=symbol)
plt.title("Steel Sector — Closing Price (2017-2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (PKR)")
plt.legend()
plt.savefig("steel_sector_price.png")
plt.show()


# Power sector (only HUBC, so no loop needed)
sub = df[df["SYMBOL"] == "HUBC"]
plt.figure(figsize=(12, 6))
plt.plot(sub["DATE"], sub["CLOSE"], label="HUBC")
plt.title("Power Sector — Closing Price (2017-2025)")
plt.xlabel("Date")
plt.ylabel("Closing Price (PKR)")
plt.legend()
plt.savefig("power_sector_price.png")
plt.show()