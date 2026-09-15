import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

train["DATE"] = pd.to_datetime(train["DATE"])
test["DATE"] = pd.to_datetime(test["DATE"])

results = []

# Train ARIMA model for each company separately
for symbol in sorted(train["SYMBOL"].unique()):
    train_symbol = train[train["SYMBOL"] == symbol].sort_values("DATE")
    test_symbol = test[test["SYMBOL"] == symbol].sort_values("DATE")

    # Get the daily returns (what we're forecasting)
    train_returns = train_symbol["daily_return"].dropna().values
    test_returns = test_symbol["daily_return"].dropna().values

    # Auto-ARIMA
    print(f"Training ARIMA for {symbol}...")
    model = auto_arima(train_returns, seasonal=False, max_p=5, max_q=5,
                       max_d=2, trace=False, error_action="ignore",
                       suppress_warnings=True)

    # Make predictions on test set
    forecast_returns = model.predict(n_periods=len(test_returns))

    # Calculate RMSE (more reliable than MAPE for returns)
    rmse = np.sqrt(mean_squared_error(test_returns, forecast_returns))

    # Also calculate directional accuracy (did we predict up/down correctly?)
    actual_direction = np.sign(test_returns)
    forecast_direction = np.sign(forecast_returns)
    directional_accuracy = (actual_direction == forecast_direction).mean() * 100

    results.append({
        "Symbol": symbol,
        "ARIMA_Order": str(model.order),
        "RMSE": f"{rmse:.4f}",
        "Directional_Accuracy": f"{directional_accuracy:.1f}%"
    })

    print(f"  Order: {model.order}, RMSE: {rmse:.4f}, Dir.Acc: {directional_accuracy:.1f}%")

# Summary
results_df = pd.DataFrame(results)
print("\n=== Model Performance Summary ===")
print(results_df.to_string(index=False))
results_df.to_csv("model_performance.csv", index=False)