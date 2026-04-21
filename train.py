"""
train.py
End-to-end training pipeline for the Time Series Forecaster.

Usage:
    python train.py --ticker AAPL --period 5y --seq_length 60 --horizon 7 --epochs 100
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

from utils.data_loader import (
    fetch_stock_data, prepare_features, scale_data,
    create_sequences, train_val_test_split
)
from utils.metrics import compute_all_metrics
from models.forecaster import build_model, compile_model, train_model, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Time Series Forecasting Model")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--period", type=str, default="5y", help="Historical data period")
    parser.add_argument("--seq_length", type=int, default=60, help="Input sequence length (days)")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (days)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--model_dir", type=str, default="models/saved", help="Model save directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  Time Series Forecaster — Training Pipeline")
    print(f"  Ticker: {args.ticker} | Horizon: {args.horizon}d | Seq: {args.seq_length}")
    print(f"{'='*60}\n")

    # ── 1. Fetch & preprocess data ──────────────────────────
    print("[1/5] Fetching data...")
    df = fetch_stock_data(args.ticker, period=args.period)
    print(f"      Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

    feature_cols = prepare_features(df)
    print(f"      Features: {len(feature_cols)}")

    # ── 2. Scale data ────────────────────────────────────────
    print("[2/5] Scaling data...")
    scaler_dir = os.path.join(args.model_dir, "scalers")
    scaled_data, feature_scaler, target_scaler = scale_data(
        df, feature_cols, scaler_path=scaler_dir, fit=True
    )
    target_col_idx = feature_cols.index("Close")

    # ── 3. Create sequences ──────────────────────────────────
    print("[3/5] Creating sequences...")
    X, y = create_sequences(scaled_data, target_col_idx, args.seq_length, args.horizon)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(f"      Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── 4. Build & train model ───────────────────────────────
    print("[4/5] Building model...")
    model = build_model(
        seq_length=args.seq_length,
        n_features=len(feature_cols),
        forecast_horizon=args.horizon,
    )
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    print("\n      Training...")
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ── 5. Evaluate on test set ──────────────────────────────
    print("[5/5] Evaluating...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predictions back to price space
    # Expand to full feature shape for inverse transform
    def inverse_transform_target(arr_2d, target_scaler):
        return target_scaler.inverse_transform(arr_2d.reshape(-1, 1)).reshape(arr_2d.shape)

    y_test_price = inverse_transform_target(y_test, target_scaler)
    y_pred_price = inverse_transform_target(y_pred_scaled, target_scaler)

    metrics = compute_all_metrics(y_test_price, y_pred_price)
    print("\n  ── Test Set Metrics ──")
    for k, v in metrics.items():
        print(f"     {k}: {v}")

    # Save model and metadata
    model_path = os.path.join(args.model_dir, "forecaster.keras")
    save_model(model, model_path)

    metadata = {
        "ticker": args.ticker,
        "period": args.period,
        "seq_length": args.seq_length,
        "forecast_horizon": args.horizon,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "test_metrics": metrics,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epochs_run": len(history.history["loss"]),
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(args.model_dir, "training_history.csv"), index=False)

    print(f"\n  ✓ Model saved to {model_path}")
    print(f"  ✓ Metadata saved to {args.model_dir}/metadata.json")
    print(f"  ✓ Training history saved to {args.model_dir}/training_history.csv")
    print(f"\n{'='*60}")
    print("  Training complete! Run `streamlit run app.py` to launch the app.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
