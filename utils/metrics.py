"""
utils/metrics.py
Evaluation metrics for time series forecasting.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct directional predictions."""
    actual_dir = np.sign(np.diff(y_true, axis=-1))
    pred_dir = np.sign(np.diff(y_pred, axis=-1))
    correct = np.sum(actual_dir == pred_dir)
    total = actual_dir.size
    return float(correct / total * 100) if total > 0 else 0.0


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a full suite of regression + forecasting metrics."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    return {
        "MAE": round(mean_absolute_error(y_true_flat, y_pred_flat), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)), 4),
        "MAPE (%)": round(mean_absolute_percentage_error(y_true_flat, y_pred_flat), 2),
        "R² Score": round(r2_score(y_true_flat, y_pred_flat), 4),
        "Directional Accuracy (%)": round(directional_accuracy(y_true, y_pred), 2),
    }
