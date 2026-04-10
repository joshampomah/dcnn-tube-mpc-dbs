# Canonical owner: closed-loop-dbs-bench
"""ARX (AutoRegressive with eXogenous input) model for direct multi-step prediction.

This module implements a linear ARX model that operates in the same framework
as the DCNN multi-step predictor. Each step k has its own coefficient vector,
fitted independently via least squares (direct multi-step, not recursive).

The ARX model captures bulk linear dynamics, while a residual DC-NN captures
the nonlinear remainder. Together they form the Residual DCNN-MPC.

All operations are in log space (matching the DCNN training convention).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class ARXModel:
    """Direct multi-step ARX model with per-step coefficient vectors.

    For each step k in {1, ..., horizon}, the model predicts:
        y_hat[k] = W_z[k] @ z + W_u[k] @ u[:k] + b[k]

    where z is the state vector (past y and u values) and u[:k] are the
    first k future control inputs.
    """

    def __init__(self, n_state: int, horizon: int = 5):
        self.n_state = n_state
        self.horizon = horizon
        self.coefficients: Dict[int, Dict] = {}
        self._is_fitted = False

    def fit(
        self,
        x: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        k: int,
        regularization: float = 0.0,
    ) -> Dict[str, float]:
        """Fit ARX model for step k using least squares."""
        regressor = np.hstack([x, u[:, :k]])
        target = y[:, k - 1]

        n_features = regressor.shape[1]
        regressor_bias = np.hstack([regressor, np.ones((len(regressor), 1))])

        if regularization > 0:
            A = regressor_bias.T @ regressor_bias
            A += regularization * np.eye(A.shape[0])
            A[-1, -1] -= regularization
            b = regressor_bias.T @ target
            w = np.linalg.solve(A, b)
        else:
            w, residuals, rank, sv = np.linalg.lstsq(regressor_bias, target, rcond=None)

        self.coefficients[k] = {
            "W": w[:n_features].copy(),
            "b": float(w[n_features]),
        }

        y_pred = regressor_bias @ w
        ss_res = np.sum((target - y_pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        mse = np.mean((target - y_pred) ** 2)

        if len(self.coefficients) == self.horizon:
            self._is_fitted = True

        return {"r2": float(r2), "mse": float(mse), "n_samples": len(target)}

    def fit_all(
        self,
        x: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        regularization: float = 0.0,
    ) -> List[Dict[str, float]]:
        """Fit ARX model for all steps k=1..horizon."""
        metrics = []
        for k in range(1, self.horizon + 1):
            m = self.fit(x, u, y, k, regularization=regularization)
            metrics.append(m)
        return metrics

    def predict(self, z: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
        """Predict output at step k."""
        if k not in self.coefficients:
            raise ValueError(f"Step k={k} not fitted. Fitted steps: {list(self.coefficients.keys())}")

        coeff = self.coefficients[k]
        W = coeff["W"]
        b = coeff["b"]

        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
            u = u.reshape(1, -1)

        regressor = np.hstack([z, u[:, :k]])
        result = regressor @ W + b

        return result[0] if single else result

    def predict_all(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict outputs for all steps k=1..horizon."""
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
            u = u.reshape(1, -1)

        N = z.shape[0]
        preds = np.zeros((N, self.horizon))
        for k in range(1, self.horizon + 1):
            preds[:, k - 1] = self.predict(z, u, k)

        return preds[0] if single else preds

    def get_u_jacobian(self, k: int) -> np.ndarray:
        """Get the u-coefficients (Jacobian w.r.t. u) for step k."""
        if k not in self.coefficients:
            raise ValueError(f"Step k={k} not fitted")

        W = self.coefficients[k]["W"]
        u_coeffs = W[self.n_state:self.n_state + k]
        return u_coeffs.reshape(1, k)

    def save(self, path: Path) -> None:
        """Save ARX model coefficients to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_state": self.n_state,
            "horizon": self.horizon,
            "coefficients": {},
        }
        for k, coeff in self.coefficients.items():
            data["coefficients"][str(k)] = {
                "W": coeff["W"].tolist(),
                "b": coeff["b"],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ARXModel":
        """Load ARX model from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        model = cls(n_state=data["n_state"], horizon=data["horizon"])
        for k_str, coeff in data["coefficients"].items():
            k = int(k_str)
            model.coefficients[k] = {
                "W": np.array(coeff["W"]),
                "b": coeff["b"],
            }

        if len(model.coefficients) == model.horizon:
            model._is_fitted = True

        return model

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
