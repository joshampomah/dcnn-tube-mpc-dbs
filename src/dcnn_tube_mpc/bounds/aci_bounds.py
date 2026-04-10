"""Adaptive Conformal Inference (ACI) bounds for DCNN-MPC.

Implements the ACI method of Gibbs & Candes (2021) for online prediction
interval adaptation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ACIConfig:
    """Configuration for ACI bounds tracker."""

    alpha: float = 0.05
    gamma: float = 0.005
    min_samples: int = 50
    warmup_strategy: str = "replace"
    horizon: int = 5
    theta_init: str = "offline"

    def __post_init__(self):
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")
        if self.warmup_strategy not in ("max", "replace", "min"):
            raise ValueError(
                f"warmup_strategy must be 'max', 'replace', or 'min', "
                f"got '{self.warmup_strategy}'"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.theta_init not in ("offline", "zero"):
            raise ValueError(
                f"theta_init must be 'offline' or 'zero', got '{self.theta_init}'"
            )

    @classmethod
    def from_dkw_equivalent(
        cls,
        delta: float = 1e-6,
        window_size: int = 500,
        horizon: int = 5,
        **kwargs,
    ) -> "ACIConfig":
        """Create ACI config with equivalent coverage to DKW."""
        alpha = np.sqrt(np.log(1.0 / delta) / (2 * window_size))
        gamma = 2 * np.log(2) / (alpha * window_size)

        return cls(
            alpha=float(alpha),
            gamma=float(gamma),
            horizon=horizon,
            **kwargs,
        )


class ACIBoundsTracker:
    """Tracks prediction errors online and computes ACI bounds."""

    def __init__(self, config: ACIConfig, offline_bounds: np.ndarray):
        self.config = config
        self.offline_bounds = np.asarray(offline_bounds).copy()

        N = config.horizon
        if self.offline_bounds.shape[0] < N:
            raise ValueError(
                f"offline_bounds has {self.offline_bounds.shape[0]} steps, "
                f"need at least {N}"
            )
        self.offline_bounds = self.offline_bounds[:N]

        if config.theta_init == "offline":
            self._theta_init = self.offline_bounds[:, 1].copy()
        else:
            self._theta_init = np.zeros(N)

        self.thetas = self._theta_init.copy()

        self.n_samples = np.zeros(N, dtype=int)
        self.n_misses = np.zeros(N, dtype=int)

        self.error_histories: List[List[float]] = [[] for _ in range(N)]

        self._pred_buffer: deque = deque(maxlen=N + 1)

    def record_prediction(self, step: int, y_nominal: np.ndarray) -> None:
        """Record DCNN prediction after SCP solve."""
        self._pred_buffer.append((step, y_nominal.copy()))

    def record_observation(self, step: int, y_true: float) -> None:
        """Record true observation and update ACI thetas."""
        N = self.config.horizon
        alpha = self.config.alpha
        gamma = self.config.gamma

        for pred_step, y_preds in self._pred_buffer:
            for k in range(min(N, len(y_preds))):
                if pred_step + k + 1 == step:
                    error = y_true - y_preds[k]
                    self.error_histories[k].append(error)
                    self.n_samples[k] += 1

                    miss = 1.0 if abs(error) > self.thetas[k] else 0.0
                    if miss > 0:
                        self.n_misses[k] += 1
                    self.thetas[k] += gamma * (miss - alpha)
                    self.thetas[k] = max(self.thetas[k], 0.0)

    def get_current_bounds(self) -> np.ndarray:
        """Compute current disturbance bounds using ACI thresholds."""
        N = self.config.horizon
        bounds = np.zeros((N, 2))

        for k in range(N):
            if self.n_samples[k] < self.config.min_samples:
                bounds[k] = self.offline_bounds[k]
                continue

            aci_bound = np.array([-self.thetas[k], self.thetas[k]])

            offline = self.offline_bounds[k]
            if self.config.warmup_strategy == "max":
                bounds[k, 0] = min(offline[0], aci_bound[0])
                bounds[k, 1] = max(offline[1], aci_bound[1])
            elif self.config.warmup_strategy == "replace":
                bounds[k] = aci_bound
            elif self.config.warmup_strategy == "min":
                bounds[k, 0] = max(offline[0], aci_bound[0])
                bounds[k, 1] = min(offline[1], aci_bound[1])

        return bounds

    def compute_coverage(self, k: int) -> float:
        """Compute empirical coverage for step k."""
        if self.n_samples[k] == 0:
            return 1.0
        return 1.0 - self.n_misses[k] / self.n_samples[k]

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the ACI tracker state."""
        N = self.config.horizon
        active = [int(self.n_samples[k]) >= self.config.min_samples for k in range(N)]

        current = self.get_current_bounds()
        offline = self.offline_bounds

        current_widths = current[:, 1] - current[:, 0]
        offline_widths = offline[:, 1] - offline[:, 0]
        width_ratios = np.where(
            offline_widths > 0,
            current_widths / offline_widths,
            np.ones(N),
        )

        return {
            "n_samples": [int(n) for n in self.n_samples],
            "active": active,
            "current_bounds": current,
            "offline_bounds": offline,
            "bound_width_ratio": width_ratios.tolist(),
            "thetas": self.thetas.copy().tolist(),
            "cumulative_coverage": [self.compute_coverage(k) for k in range(N)],
            "n_misses": [int(n) for n in self.n_misses],
        }

    def reset(self) -> None:
        """Reset all error histories, thetas, and prediction buffer."""
        N = self.config.horizon
        self.thetas = self._theta_init.copy()
        self.n_samples = np.zeros(N, dtype=int)
        self.n_misses = np.zeros(N, dtype=int)
        self.error_histories = [[] for _ in range(N)]
        self._pred_buffer.clear()
