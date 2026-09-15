import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
# from pmdarima import auto_arima
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="PSX Stock Forecasting", layout="wide")
st.title(" Multi-Sector PSX Stock Price Forecasting")
st.markdown("**Interactive dashboard for predicting stock price trends using ML models**")

# Load data
train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")
comparison = pd.read_csv("model_comparison.csv")

train["DATE"] = pd.to_datetime(train["DATE"])
test["DATE"] = pd.to_datetime(test["DATE"])

# Sidebar
st.sidebar.header("⚙️ Configuration")
selected_symbol = st.sidebar.selectbox("Select Company (Symbol)", sorted(train["SYMBOL"].unique()))
selected_model = st.sidebar.radio("Select View", ["Comparison", "ARIMA", "LSTM", "Binary Classification"])

# Get company performance
company_perf = comparison[comparison["Symbol"] == selected_symbol].iloc[0]

# Main content
if selected_model == "Comparison":
    st.subheader("📊 All Models Performance Comparison")
    st.dataframe(comparison.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Visual Comparison")
    st.image("model_comparison.png", use_container_width=True)

    st.markdown("""
    ### 🎯 Key Insights:
    - **Binary Classification** performs best (~52.51% average accuracy)
    - **LSTM & Ensemble** achieve ~50% (essentially random)
    - **ARIMA** performs below random (~41.27%)
    - All models reflect stock market's inherent randomness and unpredictability
    - **Conclusion**: Stock price direction is fundamentally difficult to predict
    """)

else:
    # Get data for selected company
    train_symbol = train[train["SYMBOL"] == selected_symbol].sort_values("DATE").dropna(subset=['daily_return'])
    test_symbol = test[test["SYMBOL"] == selected_symbol].sort_values("DATE").dropna(subset=['daily_return'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Company", selected_symbol)
    with col2:
        st.metric("Total Data Points", len(train_symbol) + len(test_symbol))
    with col3:
        if selected_model == "ARIMA":
            acc_val = company_perf["ARIMA_Acc"]
        elif selected_model == "LSTM":
            acc_val = company_perf["LSTM_Acc"]
        else:
            acc_val = company_perf["Binary_Acc"]
        st.metric("Model Accuracy", f"{acc_val:.2f}%")

    st.divider()

    # Price history chart
    st.subheader(f"📈 {selected_symbol} Price History (2017-2025)")
    all_data = pd.concat([train_symbol, test_symbol]).sort_values("DATE")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(all_data["DATE"], all_data["CLOSE"], linewidth=2.5, color='#1f77b4', label="Closing Price")
    ax.axvline(x=test_symbol["DATE"].min(), color='red', linestyle='--', linewidth=2, label="Train/Test Split (2025)",
               alpha=0.7)
    ax.fill_between(all_data["DATE"], all_data["CLOSE"].min(), all_data["CLOSE"].max(), alpha=0.1, color='blue')
    ax.set_xlabel("Date", fontweight="bold", fontsize=11)
    ax.set_ylabel("Closing Price (PKR)", fontweight="bold", fontsize=11)
    ax.set_title(f"{selected_symbol} - 8 Year Price Trend", fontweight="bold", fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.divider()

    # Model-specific info
    if selected_model == "ARIMA":
        st.subheader("📊 ARIMA Model")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("""
            **ARIMA** (AutoRegressive Integrated Moving Average) is a classical statistical time series forecasting model.
            - Uses historical patterns to predict future values
            - Works well for stationary data
            - Interprets relationships as linear
            """)
        with col2:
            st.info(f"**Accuracy on {selected_symbol}**\n\n{company_perf['ARIMA_Acc']:.2f}%")

    elif selected_model == "LSTM":
        st.subheader("🧠 LSTM Neural Network")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("""
            **LSTM** (Long Short-Term Memory) is a deep learning model that captures sequential patterns.
            - Can learn non-linear relationships
            - Uses 30-day sliding window
            - More complex but potentially captures subtle patterns
            """)
        with col2:
            st.info(f"**Accuracy on {selected_symbol}**\n\n{company_perf['LSTM_Acc']:.2f}%")

    else:  # Binary Classification
        st.subheader("🎯 Binary Classification (UP/DOWN Prediction)")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("""
            **Simplified Approach**: Predict only if price goes UP or DOWN next day.
            - Uses last 5 days of returns as features
            - Logistic Regression + Random Forest ensemble
            - Best performing model in this project
            """)
        with col2:
            st.info(f"**Accuracy on {selected_symbol}**\n\n{company_perf['Binary_Acc']:.2f}%")

st.divider()
st.markdown("""
## 📋 Project Details

| Aspect | Details |
|--------|---------|
| **Dataset** | PSX (Pakistan Stock Exchange) 2017-2025 |
| **Companies** | 17 companies across 6 sectors |
| **Data Points** | 37,088 daily records |
| **Data Cleaning** | Stock split detection & adjustment (LUCK, UBL) |
| **Models** | ARIMA, LSTM, Ensemble, Binary Classification |
| **Best Model** | Binary Classification (52.51% accuracy) |

### ⚠️ Important Disclaimer
- These predictions are for **educational purposes only**
- Not financial advice or investment recommendations
- Stock markets are inherently unpredictable
- Actual trading involves significant risk

### 📚 Technologies Used
Python • Pandas • Scikit-Learn • TensorFlow/Keras • Streamlit • Matplotlib
""")