"""
predict.py
Inference module: load a trained model and predict future prices.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import Tuple, Optional

from utils.data_loader import (
    fetch_stock_data, prepare_features, scale_data,
    create_sequences, load_scalers
)
from models.forecaster import load_model


def predict_future(
    ticker: str,
    model_dir: str = "models/saved",
    period: str = "2y",
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Run inference for a ticker using a saved model.

    Returns:
        forecast_df: DataFrame with dates and predicted prices
        metrics: dict with metadata and last known price
        history_df: Recent price history for plotting
    """
    # Load metadata
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No trained model metadata found at {meta_path}. "
            "Please run `python train.py` first."
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    seq_length = metadata["seq_length"]
    horizon = metadata["forecast_horizon"]
    feature_cols = metadata["feature_cols"]

    # Load model and scalers
    model = load_model(os.path.join(model_dir, "forecaster.keras"))
    feature_scaler, target_scaler = load_scalers(os.path.join(model_dir, "scalers"))

    # Fetch latest data
    df = fetch_stock_data(ticker, period=period)
    history_df = df[["Close"]].copy()

    # Scale and build last sequence
    feature_data = df[feature_cols].values
    scaled_features = feature_scaler.transform(feature_data)

    if len(scaled_features) < seq_length:
        raise ValueError(
            f"Not enough data. Need {seq_length} rows, got {len(scaled_features)}."
        )

    last_seq = scaled_features[-seq_length:][np.newaxis, ...]  # (1, seq_len, n_feat)

    # Predict
    pred_scaled = model.predict(last_seq, verbose=0)  # (1, horizon)
    pred_prices = target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    # Build forecast dates (skip weekends naively)
    last_date = df.index[-1]
    forecast_dates = _get_future_trading_dates(last_date, horizon)

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted_Close": np.round(pred_prices, 2),
    })
    forecast_df.set_index("Date", inplace=True)

    meta_out = {
        **metadata,
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "predicted_next": round(float(pred_prices[0]), 2),
        "predicted_change_pct": round(
            (float(pred_prices[0]) - float(df["Close"].iloc[-1]))
            / float(df["Close"].iloc[-1]) * 100, 2
        ),
    }

    return forecast_df, meta_out, history_df


def _get_future_trading_dates(start_date, n_days: int) -> list:
    """Generate n_days future trading dates (Mon–Fri) after start_date."""
    dates = []
    current = start_date
    while len(dates) < n_days:
        current = current + pd.Timedelta(days=1)
        if current.weekday() < 5:  # Monday–Friday
            dates.append(current.date())
    return dates


def load_training_history(model_dir: str = "models/saved") -> Optional[pd.DataFrame]:
    """Load saved training history CSV."""
    path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
