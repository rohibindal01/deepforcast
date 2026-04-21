"""
utils/data_loader.py
Handles data fetching, preprocessing, and sequence creation for time series forecasting.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import joblib
import os


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period: Data period ('1y', '2y', '5y', '10y', 'max')
        interval: Data interval ('1d', '1wk', '1mo')

    Returns:
        DataFrame with OHLCV data and engineered features
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as features.
    """
    # Moving Averages
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # Exponential Moving Average
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    df["RSI"] = compute_rsi(df["Close"], window=14)

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = rolling_mean + (rolling_std * 2)
    df["BB_Lower"] = rolling_mean - (rolling_std * 2)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

    # Volume indicators
    df["Volume_MA"] = df["Volume"].rolling(window=7).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

    # Price momentum
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_21d"] = df["Close"].pct_change(21)

    # Volatility
    df["Volatility"] = df["Return_1d"].rolling(window=21).std()

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def prepare_features(df: pd.DataFrame) -> list:
    """Return list of feature columns used for training."""
    return [
        "Open", "High", "Low", "Close", "Volume",
        "MA_7", "MA_21", "MA_50",
        "EMA_12", "EMA_26", "MACD", "MACD_Signal",
        "RSI", "BB_Upper", "BB_Lower", "BB_Width",
        "Volume_MA", "Volume_Ratio",
        "Return_1d", "Return_5d", "Return_21d",
        "Volatility"
    ]


def scale_data(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_path: Optional[str] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Scale features and target separately.

    Returns:
        scaled_data: Full scaled feature array
        feature_scaler: Fitted scaler for features
        target_scaler: Fitted scaler for Close price only
    """
    feature_data = df[feature_cols].values
    target_data = df[["Close"]].values

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    if fit:
        scaled_features = feature_scaler.fit_transform(feature_data)
        target_scaler.fit(target_data)
    else:
        scaled_features = feature_scaler.transform(feature_data)

    if scaler_path and fit:
        os.makedirs(scaler_path, exist_ok=True)
        joblib.dump(feature_scaler, os.path.join(scaler_path, "feature_scaler.pkl"))
        joblib.dump(target_scaler, os.path.join(scaler_path, "target_scaler.pkl"))

    return scaled_features, feature_scaler, target_scaler


def load_scalers(scaler_path: str) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Load saved scalers."""
    feature_scaler = joblib.load(os.path.join(scaler_path, "feature_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(scaler_path, "target_scaler.pkl"))
    return feature_scaler, target_scaler


def create_sequences(
    data: np.ndarray,
    target_col_idx: int,
    seq_length: int = 60,
    forecast_horizon: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/output sequences for LSTM training.

    Args:
        data: Scaled feature array (n_samples, n_features)
        target_col_idx: Index of the Close price column
        seq_length: Number of past timesteps to use as input
        forecast_horizon: Number of future timesteps to predict

    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences, forecast_horizon)
    """
    X, y = [], []
    total = len(data)
    for i in range(seq_length, total - forecast_horizon + 1):
        X.append(data[i - seq_length : i])
        y.append(data[i : i + forecast_horizon, target_col_idx])
    return np.array(X), np.array(y)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple:
    """Split sequences into train/val/test sets (temporal order preserved)."""
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
    X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

    return X_train, X_val, X_test, y_train, y_val, y_test
