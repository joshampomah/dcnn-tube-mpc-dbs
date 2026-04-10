"""High-level SCP Controller for DC-NN Tube MPC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from dcnn_tube_mpc.models.dcnn_models import MultiStepDCNN
    from dcnn_tube_mpc.controllers.scp_config import SCPConfig

from dcnn_tube_mpc.controllers.scp_algorithm import SCPResult, solve_scp, create_warm_start


@dataclass
class ControllerState:
    """Internal state of the SCP controller."""

    last_result: Optional[SCPResult] = None
    last_u_applied: float = 0.0
    step_count: int = 0
    total_solve_time: float = 0.0


class SCPController:
    """SCP-based MPC controller for DC-NN predictions.

    Example:
        >>> predictor = load_trained_model()
        >>> config = SCPConfig(Q=50000, R=1)
        >>> W = compute_disturbance_bounds(predictor, X_val, U_val, Y_val)
        >>> controller = SCPController(predictor, config, W)

        >>> y_history = get_initial_history()
        >>> u_prev = 0.0
        >>> for k in range(1000):
        ...     u_k, result = controller.compute_control(y_history, u_history, u_prev)
        ...     y_new = simulate_step(u_k)
    """

    def __init__(
        self,
        predictor: "MultiStepDCNN",
        config: "SCPConfig",
        W_bounds: Optional[np.ndarray] = None,
        device: str = "cpu",
        dkw_config=None,
        aci_config=None,
        pac_config=None,
        cusum_config=None,
        online_arx_config=None,
        stim_gain_config=None,
    ):
        """Initialize the SCP controller."""
        n_online = sum(x is not None for x in [dkw_config, aci_config, pac_config])
        if n_online > 1:
            raise ValueError(
                "Choose at most one of dkw_config, aci_config, pac_config"
            )

        self.predictor = predictor
        self.config = config
        self.device = torch.device(device)
        self.state = ControllerState()

        self.bounds_tracker = None
        self._dkw_config = dkw_config
        self._aci_config = aci_config
        self._pac_config = pac_config
        self.drift_detector = None
        self.arx_adapter = None
        self._cusum_config = cusum_config
        self._online_arx_config = online_arx_config

        self._pe_active = not config.pe_drift_triggered
        self._pe_remaining = 0
        self._pe_dither_rng = np.random.RandomState(42)

        self.stim_gain_estimator = None

        if W_bounds is not None:
            self.config = self.config.with_disturbance_bounds(W_bounds)
        elif self.config.W_bounds is None:
            self.config = self.config.with_updates(
                W_bounds=self.config.get_default_W_bounds()
            )

        if self._dkw_config is not None:
            from dcnn_tube_mpc.bounds.dkw_bounds import DKWBoundsTracker
            self.bounds_tracker = DKWBoundsTracker(
                self._dkw_config, self.config.get_default_W_bounds()
            )
        elif self._aci_config is not None:
            from dcnn_tube_mpc.bounds.aci_bounds import ACIBoundsTracker
            self.bounds_tracker = ACIBoundsTracker(
                self._aci_config, self.config.get_default_W_bounds()
            )

        expected_horizon = config.control_horizon
        if predictor.horizon < expected_horizon:
            raise ValueError(
                f"Predictor horizon ({predictor.horizon}) is less than "
                f"config control_horizon ({expected_horizon})"
            )

        self.predictor = self.predictor.to(self.device)
        self.predictor.eval()

    def apply_dither(self, u_mpc: float) -> float:
        """Apply additive dither for persistent excitation."""
        if self.config.pe_dither_amplitude <= 0:
            return u_mpc
        if not self._pe_active:
            return u_mpc
        amp = self.config.pe_dither_amplitude
        if self.config.pe_dither_type == "prbs":
            d = amp * (2 * self._pe_dither_rng.randint(0, 2) - 1)
        else:
            d = self._pe_dither_rng.uniform(-amp, amp)
        return float(np.clip(u_mpc + d, self.config.u_min, self.config.u_max))

    def compute_control(
        self,
        y_history: np.ndarray,
        u_history: np.ndarray,
        u_prev: float,
    ) -> Tuple[float, SCPResult]:
        """Compute optimal control for current state.

        Args:
            y_history: Past beta values of shape (n_state_y=15,).
                ORDERING: Newest first [y(k), y(k-1), ..., y(k-14)].
            u_history: Past control inputs of shape (n_state_u,).
                ORDERING: Newest first [u(k-1), u(k-2), ..., u(k-n)].
            u_prev: Previous applied control u(k-1).

        Returns:
            Tuple (u_k, result).
        """
        if y_history.shape != (self.config.n_state_y,):
            raise ValueError(
                f"y_history shape mismatch: expected ({self.config.n_state_y},), "
                f"got {y_history.shape}"
            )
        if u_history.shape != (self.config.n_state_u,):
            raise ValueError(
                f"u_history shape mismatch: expected ({self.config.n_state_u},), "
                f"got {u_history.shape}"
            )

        if abs(u_history[0] - u_prev) > 1e-6:
            u_history = u_history.copy()
            u_history[0] = u_prev

        if self.state.step_count > 0:
            y_true = y_history[0]

            if self.bounds_tracker is not None:
                self.bounds_tracker.record_observation(self.state.step_count, y_true)
                self.config = self.config.with_disturbance_bounds(
                    self.bounds_tracker.get_current_bounds()
                )

        y_past = y_history[::-1]
        u_past = u_history[::-1]
        z_k = np.concatenate([y_past.flatten(), u_past.flatten()])

        if self.state.last_result is not None and self.config.use_warm_start:
            u_initial = create_warm_start(self.state.last_result.u_optimal)
            u_initial = np.clip(u_initial, self.config.u_min, self.config.u_max)
        else:
            u_initial = np.full(self.config.prediction_horizon, u_prev)

        gain_scale = 1.0

        result = solve_scp(
            z_k=z_k,
            u_prev=u_prev,
            u_initial=u_initial,
            predictor=self.predictor,
            config=self.config,
            device=self.device,
            gain_scale=gain_scale,
        )

        u_out = self.apply_dither(result.u_optimal[0])

        self.state.last_result = result
        self.state.last_u_applied = u_out
        self.state.step_count += 1
        self.state.total_solve_time += sum(result.iteration_times)

        self.state.last_u_seq = result.u_optimal.copy()
        self.state.last_y_nominal = result.y_nominal.copy()
        self.state.last_cost = result.J_optimal

        step_idx = self.state.step_count - 1
        if self.bounds_tracker is not None:
            self.bounds_tracker.record_prediction(step_idx, result.y_nominal)

        return u_out, result

    def compute_control_sequence(
        self,
        y_history: np.ndarray,
        u_history: np.ndarray,
        u_prev: float,
    ) -> Tuple[np.ndarray, SCPResult]:
        """Compute full optimal control sequence."""
        _, result = self.compute_control(y_history, u_history, u_prev)
        return result.u_optimal, result

    def reset(self) -> None:
        """Reset controller state."""
        self.state = ControllerState()
        if self.bounds_tracker is not None:
            self.bounds_tracker.reset()

    def get_average_solve_time(self) -> float:
        """Get average solve time per control step."""
        if self.state.step_count == 0:
            return 0.0
        return self.state.total_solve_time / self.state.step_count

    def get_statistics(self) -> dict:
        """Get controller performance statistics."""
        stats = {
            "step_count": self.state.step_count,
            "total_solve_time": self.state.total_solve_time,
            "avg_solve_time": self.get_average_solve_time(),
            "last_converged": (
                self.state.last_result.converged
                if self.state.last_result else None
            ),
            "last_iterations": (
                self.state.last_result.n_iterations
                if self.state.last_result else None
            ),
        }
        if self.bounds_tracker is not None:
            stats["bounds"] = self.bounds_tracker.get_diagnostics()
        return stats


def create_controller(
    predictor: "MultiStepDCNN",
    Q: float = 50000.0,
    R: float = 1.0,
    beta_0: float = 2.3,
    W_bounds: Optional[np.ndarray] = None,
    device: str = "cpu",
    qp_solver_type: str = "direct",
    dkw_config=None,
    aci_config=None,
    **config_kwargs,
) -> SCPController:
    """Factory function to create SCP controller with common parameters."""
    from dcnn_tube_mpc.controllers.scp_config import SCPConfig

    control_horizon = config_kwargs.pop("control_horizon", predictor.horizon)
    prediction_horizon = config_kwargs.pop("prediction_horizon", predictor.horizon)

    if "n_state_y" not in config_kwargs and hasattr(predictor, "n_state_y"):
        config_kwargs["n_state_y"] = predictor.n_state_y
    if "n_state_u" not in config_kwargs and hasattr(predictor, "n_state_u"):
        config_kwargs["n_state_u"] = predictor.n_state_u
    elif "n_state_u" not in config_kwargs and hasattr(predictor, "n_state"):
        default_n_state_y = config_kwargs.get("n_state_y", 15)
        config_kwargs["n_state_u"] = predictor.n_state - default_n_state_y

    config = SCPConfig(
        control_horizon=control_horizon,
        prediction_horizon=prediction_horizon,
        Q=Q,
        R=R,
        beta_0=beta_0,
        qp_solver_type=qp_solver_type,
        **config_kwargs,
    )

    return SCPController(
        predictor, config, W_bounds, device,
        dkw_config=dkw_config, aci_config=aci_config,
    )
