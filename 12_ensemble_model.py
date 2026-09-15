import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

train["DATE"] = pd.to_datetime(train["DATE"])
test["DATE"] = pd.to_datetime(test["DATE"])

results = []
seq_length = 30

for symbol in sorted(train["SYMBOL"].unique()):
    print(f"Ensemble for {symbol}...")

    train_symbol = train[train["SYMBOL"] == symbol].sort_values("DATE")
    test_symbol = test[test["SYMBOL"] == symbol].sort_values("DATE")

    train_returns = train_symbol["daily_return"].dropna().values
    test_returns = test_symbol["daily_return"].dropna().values

    # 1. ARIMA prediction
    arima_model = auto_arima(train_returns, seasonal=False, max_p=5, max_q=5,
                             max_d=2, trace=False, error_action="ignore",
                             suppress_warnings=True)
    arima_forecast = arima_model.predict(n_periods=len(test_returns))

    # 2. LSTM prediction
    train_returns_scaled = train_returns.reshape(-1, 1)
    test_returns_scaled = test_returns.reshape(-1, 1)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_returns_scaled)
    test_scaled = scaler.transform(test_returns_scaled)


    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)


    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    if len(X_train) >= 10 and len(X_test) >= 10:
        lstm_model = Sequential([LSTM(50, activation='relu', input_shape=(seq_length, 1)),
                                 Dense(25, activation='relu'), Dense(1)])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        lstm_forecast_scaled = lstm_model.predict(X_test, verbose=0)
        lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled).flatten()

        # Align lengths (LSTM produces shorter sequence due to sliding window)
        arima_aligned = arima_forecast[:len(lstm_forecast)]
        test_aligned = test_returns[:len(lstm_forecast)]

        # Ensemble: simple average
        ensemble_forecast = (arima_aligned + lstm_forecast) / 2

        # Metrics
        rmse = np.sqrt(mean_squared_error(test_aligned, ensemble_forecast))
        actual_dir = np.sign(test_aligned)
        forecast_dir = np.sign(ensemble_forecast)
        dir_acc = (actual_dir == forecast_dir).mean() * 100

        results.append({
            "Symbol": symbol,
            "Ensemble_RMSE": f"{rmse:.4f}",
            "Ensemble_Dir_Acc": f"{dir_acc:.1f}%"
        })

        print(f"  RMSE: {rmse:.4f}, Dir.Acc: {dir_acc:.1f}%")

results_df = pd.DataFrame(results)
print("\n=== Ensemble Performance ===")
print(results_df.to_string(index=False))
results_df.to_csv("ensemble_performance.csv", index=False)