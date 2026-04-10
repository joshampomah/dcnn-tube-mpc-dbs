# Canonical owner: closed-loop-dbs-bench
"""Prediction evaluation metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate prediction accuracy.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with 'mse' and 'mae' metrics.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted arrays must match")
    mse = float(np.mean(np.square(y_true - y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"mse": mse, "mae": mae}
