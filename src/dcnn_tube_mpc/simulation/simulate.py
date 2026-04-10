# Canonical owner: closed-loop-dbs-bench
"""Minimal simulation harness for DC-NN Tube MPC.

External controllers (e.g. SCPController) can be plugged in via the
ControllerProtocol interface defined below.

Example:
    >>> from dcnn_tube_mpc.simulation.simulate import simulate_trial, PatientData
    >>> from dcnn_tube_mpc.synthetic.data_generator import generate_demo_patient
    >>> patient = generate_demo_patient()
    >>> result = simulate_trial("open-loop", patient, duration=10.0)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Controller Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ControllerProtocol(Protocol):
    """Interface for external controllers.

    Any object with compute_control and reset satisfies this protocol.
    """

    def compute_control(self, y: float, **kwargs) -> float: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of a closed-loop simulation trial.

    Attributes:
        time: Time vector of shape (n_steps,).
        y: Output (beta power) of shape (n_steps,).
        u: Control input (stimulation) of shape (n_steps,).
        y_ref: Reference threshold (beta_0).
        metrics: Performance metrics dictionary.
        controller_type: Type of controller used.
        solver_info: Optional solver diagnostics.
        params: Controller parameters used.
        eta: Optional stimulation effect trajectory.
    """
    time: np.ndarray
    y: np.ndarray
    u: np.ndarray
    y_ref: float
    metrics: Dict[str, float]
    controller_type: str
    solver_info: Optional[Dict] = None
    params: Dict = field(default_factory=dict)
    eta: Optional[np.ndarray] = None
    final_y_history: Optional[np.ndarray] = None
    final_u_history: Optional[np.ndarray] = None
    final_u_prev: Optional[float] = None
    final_sim_state: Optional[Dict] = None

    @property
    def n_steps(self) -> int:
        return len(self.time)

    @property
    def duration(self) -> float:
        return self.time[-1] - self.time[0] if len(self.time) > 1 else 0.0


@dataclass
class PatientData:
    """Data describing patient initial conditions and plant parameters.

    Attributes:
        y_history: Initial state history of shape (n_state_y,).
        u_history: Initial control history of shape (n_state_u,).
        stim_gain: Stimulation gain (nominal: 62.11).
        stim_tau1: Time constant 1 (nominal: 0.05).
        stim_tau2: Time constant 2 (nominal: 0.25).
        beta_ar_coeffs: AR coefficients for beta dynamics.
        noise_std: Process noise standard deviation.
    """
    y_history: np.ndarray
    u_history: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    stim_gain: float = 62.11
    stim_tau1: float = 0.05
    stim_tau2: float = 0.25
    beta_ar_coeffs: Tuple[float, ...] = (0.35, -0.08, 0.04)
    noise_std: float = 0.0012

    @classmethod
    def create_default(cls, n_state_y: int = 15, initial_beta: float = 2.5) -> "PatientData":
        """Create default patient data with given initial conditions."""
        y_history = np.full(n_state_y, initial_beta, dtype=np.float32)
        return cls(y_history=y_history)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class BetaSimulator:
    """Simulates beta power dynamics with stimulation effect (synthetic).

    Uses an AR process for natural beta and a 2nd-order ZOH state-space
    model for the stimulation attenuation effect.
    """

    def __init__(
        self,
        stim_gain: float = 62.11,
        stim_tau1: float = 0.05,
        stim_tau2: float = 0.25,
        beta_ar_coeffs: Tuple[float, ...] = (0.35, -0.08, 0.04),
        noise_std: float = 0.0012,
        dt: float = 0.02,
    ):
        self.stim_gain = stim_gain
        self.stim_tau1 = stim_tau1
        self.stim_tau2 = stim_tau2
        self.beta_ar_coeffs = beta_ar_coeffs
        self.noise_std = noise_std
        self.dt = dt
        self._build_matrices()
        self._stim_state = np.zeros(2, dtype=np.float64)
        self._natural_history = np.zeros(len(beta_ar_coeffs), dtype=np.float64)

    def _build_matrices(self) -> None:
        g, t1, t2, dt = self.stim_gain, self.stim_tau1, self.stim_tau2, self.dt
        e1 = math.exp(-dt / t1)
        e2 = math.exp(-dt / t2)
        inv_diff = 1.0 / (1.0 / t2 - 1.0 / t1)
        ad10 = (g / t2) * (e1 - e2) * inv_diff
        bd0 = 1.0 - e1
        bd1 = g * (1.0 - e1) - (t2 / t1) * ad10
        self.Ad = np.array([[e1, 0.0], [ad10, e2]], dtype=np.float64)
        self.Bd = np.array([bd0, bd1], dtype=np.float64)
        self.Cd = np.array([0.0, 1.0], dtype=np.float64)

    def initialize(self, y_history: np.ndarray) -> None:
        """Initialize simulator with historical beta observations."""
        p = len(self.beta_ar_coeffs)
        hist_len = min(p, len(y_history))
        self._natural_history = np.zeros(p, dtype=np.float64)
        self._natural_history[:hist_len] = y_history[-hist_len:][::-1]
        self._stim_state = np.zeros(2, dtype=np.float64)

    def step(self, u: float, rng: np.random.Generator = None) -> Tuple[float, float]:
        """Simulate one step.

        Args:
            u: Stimulation input.
            rng: Optional random generator for noise.

        Returns:
            Tuple of (y_observed, eta_step).
        """
        eta = float(self.Cd @ self._stim_state)
        self._stim_state = self.Ad @ self._stim_state + self.Bd * float(u)

        p = len(self.beta_ar_coeffs)
        y_natural = sum(
            self.beta_ar_coeffs[j] * self._natural_history[j]
            for j in range(p)
        )
        if rng is not None:
            y_natural += rng.standard_normal() * self.noise_std

        y_obs = float(y_natural - eta)

        # Update history
        self._natural_history = np.roll(self._natural_history, 1)
        self._natural_history[0] = y_obs + eta  # natural beta
        return y_obs, eta


def simulate_trial(
    controller: Union[str, ControllerProtocol],
    patient: PatientData,
    duration: float = 30.0,
    dt: float = 0.02,
    beta_0: float = 2.3,
    seed: int = 42,
    verbose: bool = False,
) -> SimulationResult:
    """Run a closed-loop simulation trial.

    Args:
        controller: Either a ControllerProtocol object or a string naming
            a simple baseline ("open-loop", "bang-bang").
        patient: Patient initial conditions and parameters.
        duration: Trial duration in seconds.
        dt: Sample period in seconds.
        beta_0: Pathological threshold.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        SimulationResult with trajectories and metrics.
    """
    n_steps = int(round(duration / dt))
    rng = np.random.default_rng(seed)

    sim = BetaSimulator(
        stim_gain=patient.stim_gain,
        stim_tau1=patient.stim_tau1,
        stim_tau2=patient.stim_tau2,
        beta_ar_coeffs=patient.beta_ar_coeffs,
        noise_std=patient.noise_std,
        dt=dt,
    )
    sim.initialize(patient.y_history)

    y_buf = list(patient.y_history.copy())
    u_buf = list(patient.u_history.copy())

    y_traj = np.empty(n_steps, dtype=np.float32)
    u_traj = np.empty(n_steps, dtype=np.float32)
    eta_traj = np.empty(n_steps, dtype=np.float32)
    time_vec = np.arange(n_steps, dtype=np.float32) * dt

    u_prev = float(patient.u_history[-1]) if len(patient.u_history) > 0 else 0.0

    for t in range(n_steps):
        y_now = float(y_buf[-1]) if y_buf else 2.5

        # Determine control action
        if isinstance(controller, str):
            if controller == "open-loop":
                u_now = 0.0
            elif controller == "bang-bang":
                u_now = 0.015 if y_now > beta_0 else 0.0
            else:
                raise ValueError(f"Unknown controller string: {controller!r}")
        else:
            n_state_y = len(patient.y_history)
            n_state_u = len(patient.u_history)
            z_k = np.array(y_buf[-n_state_y:] + u_buf[-n_state_u:], dtype=np.float32)
            u_now = float(controller.compute_control(y_now, z_k=z_k, u_prev=u_prev))

        y_obs, eta = sim.step(u_now, rng=rng)

        y_traj[t] = y_obs
        u_traj[t] = u_now
        eta_traj[t] = eta

        y_buf.append(y_obs)
        u_buf.append(u_now)
        u_prev = u_now

    excess = np.maximum(y_traj - beta_0, 0.0)
    metrics = {
        "mean_excess": float(np.mean(excess)),
        "time_above": float(np.mean(y_traj > beta_0)),
        "mean_u": float(np.mean(u_traj)),
        "mean_y": float(np.mean(y_traj)),
    }

    ctrl_type = controller if isinstance(controller, str) else type(controller).__name__

    return SimulationResult(
        time=time_vec,
        y=y_traj,
        u=u_traj,
        y_ref=beta_0,
        metrics=metrics,
        controller_type=ctrl_type,
        eta=eta_traj,
        final_y_history=np.array(y_buf[-len(patient.y_history):], dtype=np.float32),
        final_u_history=np.array(u_buf[-len(patient.u_history):], dtype=np.float32),
        final_u_prev=u_prev,
    )
