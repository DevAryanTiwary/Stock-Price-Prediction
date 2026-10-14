# Stock Price Prediction Lab

A Streamlit application for educational stock-price forecasting experiments using traditional machine learning and recurrent neural networks.

## Features
- Historical OHLCV download with yfinance
- Time-series feature engineering: returns, SMA, EMA, RSI, volatility
- Chronological 80/20 train-test split
- Linear Regression, Random Forest, Gradient Boosting
- Random Forest tuning with TimeSeriesSplit + RandomizedSearchCV
- LSTM and GRU models with configurable lookback, units, dropout, learning rate and epochs
- Early stopping, MAE/RMSE/MAPE/R²
- Plotly candlestick, actual-vs-predicted and model comparison charts
- Streamlit web interface

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Use Yahoo-style ticker symbols, e.g. AAPL, MSFT, RELIANCE.NS, TCS.NS.

## Important
This is an educational forecasting project, not investment advice. Historical price prediction does not guarantee future market performance.
