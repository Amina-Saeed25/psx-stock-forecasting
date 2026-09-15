import pandas as pd
import matplotlib.pyplot as plt

# Load the extended filtered dataset
df = pd.read_csv("psx_extended_17.csv")
df["DATE"] = pd.to_datetime(df["DATE"])

# Define all sectors together
sectors = {
    "Banking": ["HBL", "UBL", "MCB", "MEBL"],
    "Cement": ["LUCK", "MLCF", "DGKC", "FCCL"],
    "Oil & Gas": ["OGDC", "PPL", "POL", "PSO"],
    "Fertilizer": ["EFERT", "FFC"],
    "Steel": ["ASL", "ISL"],
    "Power": ["HUBC"]
}

# Smaller figure size overall
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, (sector_name, symbols) in enumerate(sectors.items()):
    ax = axes[i]
    for symbol in symbols:
        sub = df[df["SYMBOL"] == symbol]
        ax.plot(sub["DATE"], sub["CLOSE"], label=symbol)
    ax.set_title(sector_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Closing Price (PKR)", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(axis='both', labelsize=8)

# Extra vertical spacing between rows so titles don't collide with the row above's x-axis
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

plt.savefig("all_sectors_overview.png", dpi=150)
plt.show()