import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

train["DATE"] = pd.to_datetime(train["DATE"])
test["DATE"] = pd.to_datetime(test["DATE"])

results = []

# Sequence length: use last 30 days to predict next day
seq_length = 30

for symbol in sorted(train["SYMBOL"].unique()):
    print(f"Training LSTM for {symbol}...")

    train_symbol = train[train["SYMBOL"] == symbol].sort_values("DATE")
    test_symbol = test[test["SYMBOL"] == symbol].sort_values("DATE")

    # Get returns
    train_returns = train_symbol["daily_return"].dropna().values.reshape(-1, 1)
    test_returns = test_symbol["daily_return"].dropna().values.reshape(-1, 1)

    # Normalize data (0-1 scale, required for neural networks)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_returns)
    test_scaled = scaler.transform(test_returns)


    # Create sequences: use last 30 days to predict next day
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)


    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # Skip if not enough data
    if len(X_train) < 10 or len(X_test) < 10:
        print(f"  Skipped: not enough data")
        continue

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train (quiet mode, no verbose output)
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Predict
    forecast_scaled = model.predict(X_test, verbose=0)
    forecast = scaler.inverse_transform(forecast_scaled)

    # Get actual test values (aligned with forecast length)
    y_test_actual = scaler.inverse_transform(y_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, forecast))

    # Directional accuracy
    actual_direction = np.sign(y_test_actual.flatten())
    forecast_direction = np.sign(forecast.flatten())
    directional_accuracy = (actual_direction == forecast_direction).mean() * 100

    results.append({
        "Symbol": symbol,
        "LSTM_RMSE": f"{rmse:.4f}",
        "LSTM_Dir_Acc": f"{directional_accuracy:.1f}%"
    })

    print(f"  RMSE: {rmse:.4f}, Dir.Acc: {directional_accuracy:.1f}%")

# Summary
results_df = pd.DataFrame(results)
print("\n=== LSTM Model Performance ===")
print(results_df.to_string(index=False))
results_df.to_csv("lstm_performance.csv", index=False)