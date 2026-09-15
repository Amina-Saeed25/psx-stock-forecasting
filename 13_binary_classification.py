import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

train["DATE"] = pd.to_datetime(train["DATE"])
test["DATE"] = pd.to_datetime(test["DATE"])

results = []

for symbol in sorted(train["SYMBOL"].unique()):
    print(f"Binary Classification for {symbol}...")

    train_symbol = train[train["SYMBOL"] == symbol].sort_values("DATE").reset_index(drop=True)
    test_symbol = test[test["SYMBOL"] == symbol].sort_values("DATE").reset_index(drop=True)

    # Drop NaN values first
    train_symbol = train_symbol.dropna(subset=['daily_return']).reset_index(drop=True)
    test_symbol = test_symbol.dropna(subset=['daily_return']).reset_index(drop=True)

    # Create features: use last 5 days returns to predict next day
    window = 5


    def create_features(data):
        X, y = [], []
        for i in range(window, len(data)):
            # Features: last 5 days returns
            X.append(data['daily_return'].iloc[i - window:i].values)
            # Target: is next day UP (1) or DOWN (0)?
            next_return = data['daily_return'].iloc[i]
            y.append(1 if next_return > 0 else 0)
        return np.array(X), np.array(y)


    X_train, y_train = create_features(train_symbol)
    X_test, y_test = create_features(test_symbol)

    if len(X_train) < 10 or len(X_test) < 10:
        print(f"  Skipped: not enough data")
        continue

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)

    # Model 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)

    # Better of the two
    if lr_acc >= rf_acc:
        best_acc = lr_acc
        best_model = "LogisticRegression"
    else:
        best_acc = rf_acc
        best_model = "RandomForest"

    results.append({
        "Symbol": symbol,
        "LogisticReg_Acc": f"{lr_acc * 100:.1f}%",
        "RandomForest_Acc": f"{rf_acc * 100:.1f}%",
        "Best_Model": best_model,
        "Best_Accuracy": f"{best_acc * 100:.1f}%"
    })

    print(f"  LR: {lr_acc * 100:.1f}%, RF: {rf_acc * 100:.1f}% -> Best: {best_model} ({best_acc * 100:.1f}%)")

# Summary
results_df = pd.DataFrame(results)
print("\n=== Binary Classification Performance ===")
print(results_df.to_string(index=False))
results_df.to_csv("binary_classification_performance.csv", index=False)