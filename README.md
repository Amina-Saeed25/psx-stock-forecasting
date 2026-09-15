# Multi-Sector PSX Stock Price Forecasting

Interactive dashboard for predicting stock price trends using machine learning models.

##  Live Demo
https://psx-stock-forecasting-3avixebw9y25ej6yjdrzbr.streamlit.app/

##  Project Overview
- **Dataset**: PSX (Pakistan Stock Exchange) 2017-2025
- **Companies**: 17 companies across 6 sectors (Banking, Cement, Oil & Gas, Fertilizer, Steel, Power)
- **Models**: ARIMA, LSTM, Ensemble, Binary Classification
- **Best Model**: Binary Classification (52.51% accuracy)

##  How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Files
- `01_data_filtering.py` - Data filtering & preprocessing
- `02-14_*.py` - Data analysis & model training
- `app.py` - Streamlit web application
- `train_data.csv`, `test_data.csv` - Training & testing data
- `model_*.csv` - Pre-computed model results