# import pandas as pd
#
# # Load the full PSX dataset (all 1,142 companies)
# df = pd.read_csv("compiled_psx_historical_2017_2025.csv")
#
# # Check the shape (rows, columns) and first few rows
# print(df.shape)
# print(df.head())
#


import pandas as pd

df = pd.read_csv("compiled_psx_historical_2017_2025.csv")
# Filter only the 6 companies we need
# Extended company list across 6 sectors
companies = [
    "HBL", "UBL", "MCB", "MEBL",      # Banking
    "LUCK", "MLCF", "DGKC", "FCCL",   # Cement
    "OGDC", "PPL", "POL", "PSO",      # Oil & Gas
    "EFERT", "FFC",                    # Fertilizer
    "ASL", "ISL",                      # Steel
    "HUBC"                             # Power/Energy
]

df_filtered = df[df["SYMBOL"].isin(companies)]

print(df_filtered.shape)
print(df_filtered["SYMBOL"].value_counts())

df_filtered.to_csv("psx_extended_17.csv", index=False)
print("Saved!")