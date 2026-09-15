import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all results
arima = pd.read_csv("model_performance.csv")
lstm = pd.read_csv("lstm_performance.csv")
ensemble = pd.read_csv("ensemble_performance.csv")
binary = pd.read_csv("binary_classification_performance.csv")

# Merge all into one dataframe for comparison
comparison = arima[["Symbol"]].copy()
comparison["ARIMA_Acc"] = arima["Directional_Accuracy"].str.rstrip('%').astype(float)
comparison["LSTM_Acc"] = lstm["LSTM_Dir_Acc"].str.rstrip('%').astype(float)
comparison["Ensemble_Acc"] = ensemble["Ensemble_Dir_Acc"].str.rstrip('%').astype(float)
comparison["Binary_Acc"] = binary["Best_Accuracy"].str.rstrip('%').astype(float)

# Calculate average accuracy per model
print("\n=== Average Accuracy by Model ===")
print(f"ARIMA: {comparison['ARIMA_Acc'].mean():.2f}%")
print(f"LSTM: {comparison['LSTM_Acc'].mean():.2f}%")
print(f"Ensemble: {comparison['Ensemble_Acc'].mean():.2f}%")
print(f"Binary Classification: {comparison['Binary_Acc'].mean():.2f}%")

# Create comparison chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Accuracy by Company (all models)
ax1 = axes[0]
x = np.arange(len(comparison))
width = 0.2

ax1.bar(x - 1.5*width, comparison["ARIMA_Acc"], width, label="ARIMA", alpha=0.8)
ax1.bar(x - 0.5*width, comparison["LSTM_Acc"], width, label="LSTM", alpha=0.8)
ax1.bar(x + 0.5*width, comparison["Ensemble_Acc"], width, label="Ensemble", alpha=0.8)
ax1.bar(x + 1.5*width, comparison["Binary_Acc"], width, label="Binary Classification", alpha=0.8)

ax1.set_xlabel("Company", fontsize=10, fontweight="bold")
ax1.set_ylabel("Directional Accuracy (%)", fontsize=10, fontweight="bold")
ax1.set_title("Model Comparison by Company", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(comparison["Symbol"], rotation=45, ha='right')
ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, label="50% (Random)", alpha=0.7)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Chart 2: Average accuracy comparison
ax2 = axes[1]
models = ["ARIMA", "LSTM", "Ensemble", "Binary\nClassification"]
avg_acc = [
    comparison["ARIMA_Acc"].mean(),
    comparison["LSTM_Acc"].mean(),
    comparison["Ensemble_Acc"].mean(),
    comparison["Binary_Acc"].mean()
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax2.bar(models, avg_acc, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, avg_acc):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label="50% (Random Baseline)", alpha=0.7)
ax2.set_ylabel("Average Directional Accuracy (%)", fontsize=10, fontweight="bold")
ax2.set_title("Average Performance Across All Companies", fontsize=12, fontweight="bold")
ax2.set_ylim([35, 60])
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
print("\nSaved: model_comparison.png")
plt.show()

# Save comparison table
comparison.to_csv("model_comparison.csv", index=False)
print("Saved: model_comparison.csv")