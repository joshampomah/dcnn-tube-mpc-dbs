"""Online DKW disturbance bounds for DCNN-MPC."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DKWConfig:
    """Configuration for online DKW bounds tracker."""

    delta: float = 1e-6
    alpha: Optional[float] = None
    min_samples: int = 50
    warmup_strategy: str = "replace"
    horizon: int = 5
    window_size: Optional[int] = None

    def __post_init__(self):
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.alpha is not None and (self.alpha <= 0 or self.alpha >= 1):
            raise ValueError(f"alpha must be in (0, 1) or None, got {self.alpha}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")
        if self.warmup_strategy not in ("max", "replace", "min"):
            raise ValueError(
                f"warmup_strategy must be 'max', 'replace', or 'min', "
                f"got '{self.warmup_strategy}'"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.window_size is not None and self.window_size < self.min_samples:
            raise ValueError(
                f"window_size must be >= min_samples ({self.min_samples}), "
                f"got {self.window_size}"
            )


class DKWBoundsTracker:
    """Tracks prediction errors online and computes DKW bounds."""

    def __init__(self, config: DKWConfig, offline_bounds: np.ndarray):
        self.config = config
        self.offline_bounds = np.asarray(offline_bounds).copy()

        N = config.horizon
        if self.offline_bounds.shape[0] < N:
            raise ValueError(
                f"offline_bounds has {self.offline_bounds.shape[0]} steps, "
                f"need at least {N}"
            )
        self.offline_bounds = self.offline_bounds[:N]

        self.error_histories: List[List[float]] = [[] for _ in range(N)]
        self._pred_buffer: deque = deque(maxlen=N + 1)

    def record_prediction(self, step: int, y_nominal: np.ndarray) -> None:
        """Record DCNN prediction after SCP solve."""
        self._pred_buffer.append((step, y_nominal.copy()))

    def record_observation(self, step: int, y_true: float) -> None:
        """Record true observation and compute prediction errors."""
        N = self.config.horizon
        for pred_step, y_preds in self._pred_buffer:
            for k in range(min(N, len(y_preds))):
                if pred_step + k + 1 == step:
                    error = y_true - y_preds[k]
                    self.error_histories[k].append(error)

    def compute_epsilon(self, n: int) -> float:
        """Compute DKW epsilon for n samples."""
        if n <= 0:
            return 1.0
        return np.sqrt(np.log(1.0 / self.config.delta) / (2 * n))

    def get_current_bounds(self) -> np.ndarray:
        """Compute current disturbance bounds using DKW inequality."""
        N = self.config.horizon
        bounds = np.zeros((N, 2))

        for k in range(N):
            n = len(self.error_histories[k])

            if n < self.config.min_samples:
                bounds[k] = self.offline_bounds[k]
                continue

            errors = np.array(self.error_histories[k])
            if self.config.window_size is not None:
                errors = errors[-self.config.window_size:]
            abs_errors = np.abs(errors)

            if self.config.alpha is not None:
                dkw_width = float(np.quantile(abs_errors, 1 - self.config.alpha))
            else:
                dkw_width = float(np.max(abs_errors))

            dkw_bound = np.array([-dkw_width, dkw_width])

            offline = self.offline_bounds[k]
            if self.config.warmup_strategy == "max":
                bounds[k, 0] = min(offline[0], dkw_bound[0])
                bounds[k, 1] = max(offline[1], dkw_bound[1])
            elif self.config.warmup_strategy == "replace":
                bounds[k] = dkw_bound
            elif self.config.warmup_strategy == "min":
                bounds[k, 0] = max(offline[0], dkw_bound[0])
                bounds[k, 1] = min(offline[1], dkw_bound[1])

        return bounds

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the DKW tracker state."""
        N = self.config.horizon
        n_total = [len(self.error_histories[k]) for k in range(N)]
        if self.config.window_size is not None:
            n_samples = [min(nt, self.config.window_size) for nt in n_total]
        else:
            n_samples = list(n_total)
        epsilons = [self.compute_epsilon(n) for n in n_samples]
        active = [n >= self.config.min_samples for n in n_samples]

        alpha = self.config.alpha or 0.0
        exceedance = [alpha + eps for eps in epsilons]

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
            "n_samples": n_samples,
            "epsilon": epsilons,
            "exceedance_bound": exceedance,
            "active": active,
            "current_bounds": current,
            "offline_bounds": offline,
            "bound_width_ratio": width_ratios.tolist(),
            "alpha": self.config.alpha,
        }

    def reset(self) -> None:
        """Reset all error histories and prediction buffer."""
        N = self.config.horizon
        self.error_histories = [[] for _ in range(N)]
        self._pred_buffer.clear()
