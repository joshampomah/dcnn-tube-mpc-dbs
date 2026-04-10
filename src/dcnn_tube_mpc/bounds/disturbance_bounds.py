"""Disturbance bounds computation from validation data."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn
    from dcnn_tube_mpc.models.dcnn_models import MultiStepDCNN


def compute_disturbance_bounds(
    predictor: "MultiStepDCNN",
    X_val: np.ndarray,
    U_val: np.ndarray,
    Y_val: np.ndarray,
    percentile: float = 80.0,
    symmetric: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Compute disturbance bounds W_i from validation prediction errors.

    Per CDC25: disturbance set W_i is the 80th percentile absolute prediction error.

    Args:
        predictor: Trained MultiStepDCNN model with N networks.
        X_val: Validation state data of shape (n_samples, n_state).
        U_val: Validation control sequences of shape (n_samples, N).
        Y_val: Validation target outputs of shape (n_samples, N).
        percentile: Percentile of absolute errors to use (default: 80.0).
        symmetric: If True, use symmetric bounds [-w, +w].
        device: Device to run predictions on.

    Returns:
        Array of shape (N, 2) where W[i] = [w_min, w_max].
    """
    import torch

    n_samples = X_val.shape[0]
    if U_val.shape[0] != n_samples or Y_val.shape[0] != n_samples:
        raise ValueError(
            f"Sample count mismatch: X_val has {n_samples}, "
            f"U_val has {U_val.shape[0]}, Y_val has {Y_val.shape[0]}"
        )

    horizon = predictor.horizon
    if U_val.shape[1] != horizon:
        raise ValueError(f"U_val has {U_val.shape[1]} steps, expected {horizon}")
    if Y_val.shape[1] != horizon:
        raise ValueError(f"Y_val has {Y_val.shape[1]} steps, expected {horizon}")

    device = torch.device(device)
    X_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    U_tensor = torch.tensor(U_val, dtype=torch.float32, device=device)

    predictor = predictor.to(device).eval()
    errors = np.zeros((n_samples, horizon))

    with torch.no_grad():
        predictions = predictor(X_tensor, U_tensor)
        for i, pred in enumerate(predictions):
            y_pred = pred.cpu().numpy().flatten()
            y_true = Y_val[:, i]
            errors[:, i] = y_pred - y_true

    W_bounds = np.zeros((horizon, 2))

    if symmetric:
        abs_errors = np.abs(errors)
        for i in range(horizon):
            p = np.percentile(abs_errors[:, i], percentile)
            W_bounds[i] = [-p, p]
    else:
        low_pct = (100 - percentile) / 2
        high_pct = 100 - low_pct
        for i in range(horizon):
            W_bounds[i, 0] = np.percentile(errors[:, i], low_pct)
            W_bounds[i, 1] = np.percentile(errors[:, i], high_pct)

    return W_bounds


def validate_disturbance_bounds(
    W_bounds: np.ndarray,
    errors: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Validate that disturbance bounds cover the specified percentage of errors."""
    n_samples, horizon = errors.shape
    per_step_coverage = np.zeros(horizon)

    for i in range(horizon):
        w_min, w_max = W_bounds[i]
        within_bounds = (errors[:, i] >= w_min) & (errors[:, i] <= w_max)
        per_step_coverage[i] = np.mean(within_bounds)

    overall_coverage = np.mean(per_step_coverage)
    return overall_coverage, per_step_coverage
