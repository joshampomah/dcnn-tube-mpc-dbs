"""Configuration dataclass for SCP (Sequential Convex Programming) solver.

This module provides configuration parameters for the SCP-based MPC controller
as described in Algorithm 1 of the CDC25 paper.

Device/physical parameters (u_max, delta_u_max, etc.) are loaded from the
central device_params.json for consistency across all code.

Example:
    >>> from dcnn_tube_mpc.controllers.scp_config import SCPConfig
    >>> config = SCPConfig(Q=50000.0, R=1.0, maxiters=10)
    >>> config.prediction_horizon
    5
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dcnn_tube_mpc.config.device_config import get_device_config

# Load device config once at module level
_DEVICE_CONFIG = get_device_config()


@dataclass
class SCPConfig:
    """Configuration for SCP solver and DCNN Tube MPC."""

    # State dimensions
    n_state_y: int = 15
    n_state_u: int = 5

    # Horizon settings
    prediction_horizon: int = 5
    control_horizon: int = 5
    freeze_bounds: bool = True
    decimation: int = 1

    # SCP algorithm parameters
    maxiters: int = 10
    delta_J_min: float = 1e-5
    delta_u_tol: float = 0.0

    # Cost function weights
    Q: float = 50000.0
    R: float = 1.0
    R_delta: float = 0.0

    # Persistent Excitation
    pe_gamma: float = 0.0
    pe_drift_triggered: bool = True
    pe_adapt_window: int = 250
    pe_dither_amplitude: float = 0.0
    pe_dither_type: str = "prbs"

    # Beta threshold
    beta_0: float = field(
        default_factory=lambda: _DEVICE_CONFIG.beta.default_threshold
    )

    # Input constraints (loaded from central device config)
    u_min: float = field(default_factory=lambda: _DEVICE_CONFIG.constraints.u_min)
    u_max: float = field(default_factory=lambda: _DEVICE_CONFIG.constraints.u_max)
    delta_u_max: float = field(default_factory=lambda: _DEVICE_CONFIG.constraints.delta_u_max)

    # Output constraints (soft)
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    # Warm-starting
    use_warm_start: bool = True

    # Solver settings
    solver: str = "MOSEK"
    solver_verbose: bool = False

    # Feasibility handling
    constraint_softening: bool = True

    # QP solver selection
    qp_solver_type: str = "direct"

    # Disturbance bounds
    W_bounds: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.prediction_horizon < 1:
            raise ValueError(f"prediction_horizon must be >= 1, got {self.prediction_horizon}")
        if self.control_horizon < 1:
            raise ValueError(f"control_horizon must be >= 1, got {self.control_horizon}")
        if self.control_horizon > self.prediction_horizon:
            raise ValueError(
                f"control_horizon ({self.control_horizon}) must be <= "
                f"prediction_horizon ({self.prediction_horizon})"
            )
        if self.maxiters < 1:
            raise ValueError(f"maxiters must be >= 1, got {self.maxiters}")
        if self.delta_J_min <= 0:
            raise ValueError(f"delta_J_min must be > 0, got {self.delta_J_min}")
        if self.Q < 0:
            raise ValueError(f"Q must be >= 0, got {self.Q}")
        if self.R < 0:
            raise ValueError(f"R must be >= 0, got {self.R}")
        if self.u_min > self.u_max:
            raise ValueError(f"u_min ({self.u_min}) must be <= u_max ({self.u_max})")
        if self.delta_u_max <= 0:
            raise ValueError(f"delta_u_max must be > 0, got {self.delta_u_max}")
        if self.pe_gamma < 0:
            raise ValueError(f"pe_gamma must be >= 0, got {self.pe_gamma}")
        if self.pe_dither_amplitude < 0:
            raise ValueError(f"pe_dither_amplitude must be >= 0, got {self.pe_dither_amplitude}")
        if self.pe_dither_amplitude > self.u_max / 2:
            raise ValueError(
                f"pe_dither_amplitude ({self.pe_dither_amplitude}) too large "
                f"relative to u_max ({self.u_max})"
            )

        # Scale Q, R, R_delta, pe_gamma so that R*u_max^2 is at least MIN_R_COST
        MIN_R_COST = 0.01
        r_cost = self.R * self.u_max ** 2
        if r_cost > 0 and r_cost < MIN_R_COST:
            scale = MIN_R_COST / r_cost
            self.Q *= scale
            self.R *= scale
            self.R_delta *= scale
            self.pe_gamma *= scale

        if self.pe_gamma >= self.R / 2:
            raise ValueError(
                f"pe_gamma must be < R/2 for PSD Hessian: "
                f"pe_gamma={self.pe_gamma}, R/2={self.R/2}"
            )

    def with_updates(self, **changes) -> "SCPConfig":
        """Create a new config with updated fields."""
        return dataclasses.replace(self, **changes)

    def with_disturbance_bounds(self, W_bounds: np.ndarray) -> "SCPConfig":
        """Create a new config with disturbance bounds set."""
        W_bounds = np.asarray(W_bounds)

        if W_bounds.shape[0] > self.prediction_horizon and W_bounds.shape[1] == 2:
            W_bounds = W_bounds[:self.prediction_horizon]

        if W_bounds.shape == (self.control_horizon, 2) and self.uses_extended_horizon:
            extended_bounds = np.zeros((self.prediction_horizon, 2))
            extended_bounds[:self.control_horizon] = W_bounds
            if self.freeze_bounds:
                for i in range(self.control_horizon, self.prediction_horizon):
                    extended_bounds[i] = W_bounds[-1].copy()
            else:
                base_bound = abs(W_bounds[-1, 1])
                for i in range(self.control_horizon, self.prediction_horizon):
                    w_bound = base_bound * np.sqrt((i + 1) / self.control_horizon)
                    extended_bounds[i] = [-w_bound, w_bound]
            W_bounds = extended_bounds

        if W_bounds.shape != (self.prediction_horizon, 2):
            raise ValueError(
                f"W_bounds must have shape ({self.prediction_horizon}, 2) or "
                f"({self.control_horizon}, 2), got {W_bounds.shape}"
            )
        return self.with_updates(W_bounds=W_bounds)

    def get_default_W_bounds(self) -> np.ndarray:
        """Get default disturbance bounds if not set from data."""
        if self.W_bounds is not None:
            return self.W_bounds

        calibrated_bounds = np.array([
            [-0.13, 0.13],
            [-0.30, 0.30],
            [-0.45, 0.45],
            [-0.57, 0.57],
            [-0.64, 0.64],
        ])

        bounds = np.zeros((self.prediction_horizon, 2))

        n_calibrated = min(self.control_horizon, 5)
        bounds[:n_calibrated] = calibrated_bounds[:n_calibrated]

        if self.control_horizon > 5:
            for i in range(5, self.control_horizon):
                w_bound = 0.64 * np.sqrt((i + 1) / 5)
                bounds[i] = [-w_bound, w_bound]

        if self.prediction_horizon > self.control_horizon:
            if self.freeze_bounds:
                frozen_bound = bounds[self.control_horizon - 1]
                for i in range(self.control_horizon, self.prediction_horizon):
                    bounds[i] = frozen_bound.copy()
            else:
                base_bound = abs(bounds[self.control_horizon - 1, 1])
                base_step = self.control_horizon
                for i in range(self.control_horizon, self.prediction_horizon):
                    w_bound = base_bound * np.sqrt((i + 1) / base_step)
                    bounds[i] = [-w_bound, w_bound]

        return bounds

    @property
    def uses_extended_horizon(self) -> bool:
        """Check if extended horizon (move blocking) is enabled."""
        return self.prediction_horizon > self.control_horizon
