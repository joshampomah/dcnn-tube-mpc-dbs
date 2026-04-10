"""Synthetic modulation for CDC25-style training data generation.

This module implements the stimulation modulation described in CDC25 Section IV-A:
- PRBS stimulation with incremental (random walk) signal
- Modulation formula: y = y_beta * exp(-eta(u))

By default, training data is generated with fixed nominal eta parameters
(k, tau1, tau2), since the DCNN should learn the nominal plant dynamics.
Time-varying random-walk parameters are available via ``use_fixed_params=False``
for robustness evaluation or domain-randomisation experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import cont2discrete


@dataclass
class ModulationConfig:
    """Configuration for synthetic modulation.

    Note: These defaults match central device_params.json.
    When used in training, values are typically overridden from DCNNConfig.
    """

    # PRBS parameters
    delta_u_max: float = 0.0024  # Rate limit: 0.25s ramp (12.5 steps @ 20ms)
    u_max: float = 0.030  # Max stimulation
    prbs_freq: float = 50.0  # Switching frequency (Hz)

    # Random walk parameters (CDC25 Section IV-A)
    param_max_step_deviation: float = 0.025  # 2.5% per step
    param_max_total_deviation: float = 0.40  # 40% total

    # Nominal eta parameters
    k_nominal: float = 62.11
    tau1_nominal: float = 0.05
    tau2_nominal: float = 0.25

    # Time step
    Ts: float = 0.02  # 50 Hz sampling


def generate_parameter_random_walk(
    n_steps: int,
    nominal_value: float,
    max_step_dev: float = 0.025,
    max_total_dev: float = 0.40,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate time-varying parameter via random walk.

    Per CDC25 Section IV-A: parameters vary slowly via random walk with
    constraints on per-step and total deviation from nominal.

    Args:
        n_steps: Number of time steps
        nominal_value: Nominal parameter value
        max_step_dev: Maximum relative deviation per step (default: 2.5%)
        max_total_dev: Maximum total relative deviation (default: 40%)
        seed: Random seed for reproducibility

    Returns:
        param_trajectory: Array of shape (n_steps,) with parameter values
    """
    rng = np.random.default_rng(seed)

    trajectory = np.empty(n_steps, dtype=np.float32)
    current = nominal_value

    min_val = nominal_value * (1.0 - max_total_dev)
    max_val = nominal_value * (1.0 + max_total_dev)
    step_size = nominal_value * max_step_dev

    for i in range(n_steps):
        trajectory[i] = current
        delta = rng.uniform(-step_size, step_size)
        current = np.clip(current + delta, min_val, max_val)

    return trajectory


def generate_all_parameter_trajectories(
    n_steps: int,
    config: ModulationConfig,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate time-varying k, tau1, tau2 via random walk.

    Args:
        n_steps: Number of time steps
        config: Modulation configuration with nominal parameters
        seed: Base random seed (each parameter uses seed+offset)

    Returns:
        Dict with keys 'k', 'tau1', 'tau2', each containing trajectory array
    """
    k_seed = seed
    tau1_seed = None if seed is None else seed + 1000
    tau2_seed = None if seed is None else seed + 2000

    return {
        "k": generate_parameter_random_walk(
            n_steps,
            config.k_nominal,
            config.param_max_step_deviation,
            config.param_max_total_deviation,
            k_seed,
        ),
        "tau1": generate_parameter_random_walk(
            n_steps,
            config.tau1_nominal,
            config.param_max_step_deviation,
            config.param_max_total_deviation,
            tau1_seed,
        ),
        "tau2": generate_parameter_random_walk(
            n_steps,
            config.tau2_nominal,
            config.param_max_step_deviation,
            config.param_max_total_deviation,
            tau2_seed,
        ),
    }


def discretize_eta_params(
    k: float, tau1: float, tau2: float, Ts: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize continuous-time eta model for given parameters.

    Continuous system (CDC24 Equation 2):
        xdot_c = A_c x_c + B_c u
        eta = C_c x_c

    Args:
        k: Gain parameter
        tau1: First time constant
        tau2: Second time constant
        Ts: Sampling period

    Returns:
        A_eta, B_eta, C_eta: Discrete-time system matrices
    """
    A_c = np.array([[-1.0 / tau1, 0.0], [1.0 / tau2, -1.0 / tau2]], dtype=np.float32)
    B_c = np.array([[k / tau1], [0.0]], dtype=np.float32)
    C_c = np.array([[0.0, 1.0]], dtype=np.float32)
    D_c = np.array([[0.0]], dtype=np.float32)

    sys_discrete = cont2discrete((A_c, B_c, C_c, D_c), Ts, method="zoh")
    A_eta = np.asarray(sys_discrete[0], dtype=np.float32)
    B_eta = np.asarray(sys_discrete[1], dtype=np.float32)
    C_eta = np.asarray(sys_discrete[2], dtype=np.float32)

    return A_eta, B_eta, C_eta


def simulate_eta_time_varying(
    stim: np.ndarray,
    param_trajectories: Dict[str, np.ndarray],
    Ts: float,
) -> np.ndarray:
    """Simulate eta with time-varying parameters.

    At each time step, the parameters (k, tau1, tau2) can change,
    so we need to update the discretized system matrices.

    Args:
        stim: Stimulation signal array of shape (n_steps,)
        param_trajectories: Dict with 'k', 'tau1', 'tau2' arrays
        Ts: Sampling period

    Returns:
        eta: Stimulation effect array of shape (n_steps,)
    """
    n_steps = stim.shape[0]
    k_traj = param_trajectories["k"]
    tau1_traj = param_trajectories["tau1"]
    tau2_traj = param_trajectories["tau2"]

    x = np.zeros(2, dtype=np.float32)
    eta = np.zeros(n_steps, dtype=np.float32)

    for i in range(n_steps):
        A_eta, B_eta, C_eta = discretize_eta_params(
            k_traj[i], tau1_traj[i], tau2_traj[i], Ts
        )
        eta[i] = float((C_eta @ x[:, None])[0, 0])
        x = (A_eta @ x[:, None] + B_eta * stim[i])[:, 0]

    return eta


def simulate_eta_fixed(
    stim: np.ndarray,
    config: ModulationConfig,
) -> np.ndarray:
    """Simulate eta with fixed (nominal) parameters.

    Args:
        stim: Stimulation signal array of shape (n_steps,)
        config: Modulation configuration with nominal parameters

    Returns:
        eta: Stimulation effect array of shape (n_steps,)
    """
    A_eta, B_eta, C_eta = discretize_eta_params(
        config.k_nominal,
        config.tau1_nominal,
        config.tau2_nominal,
        config.Ts,
    )

    n_steps = stim.shape[0]
    x = np.zeros(2, dtype=np.float32)
    eta = np.zeros(n_steps, dtype=np.float32)

    for i in range(n_steps):
        eta[i] = float((C_eta @ x[:, None])[0, 0])
        x = (A_eta @ x[:, None] + B_eta * stim[i])[:, 0]

    return eta


def generate_prbs_stimulation(
    n_steps: int,
    config: ModulationConfig,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate PRBS stimulation signal.

    Generates a rate-limited random binary stimulation sequence that
    changes by at most delta_u_max per step.

    Args:
        n_steps: Number of time steps
        config: Modulation configuration
        seed: Random seed for reproducibility

    Returns:
        stim: Stimulation signal of shape (n_steps,)
    """
    rng = np.random.default_rng(seed)
    stim = np.zeros(n_steps, dtype=np.float32)
    current = config.u_max / 2.0

    for i in range(n_steps):
        delta = rng.uniform(-config.delta_u_max, config.delta_u_max)
        current = float(np.clip(current + delta, 0.0, config.u_max))
        stim[i] = current

    return stim


def generate_modulated_output(
    y_beta: np.ndarray,
    stim: np.ndarray,
    config: ModulationConfig,
    use_fixed_params: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply stimulation modulation to natural beta trajectory.

    Computes: y_modulated = y_beta - eta(stim)

    Args:
        y_beta: Natural beta trajectory (n_steps,)
        stim: Stimulation signal (n_steps,)
        config: Modulation configuration
        use_fixed_params: If True, uses nominal parameters (faster)
        seed: Random seed for parameter trajectories

    Returns:
        y_modulated: Modulated beta of shape (n_steps,)
    """
    if use_fixed_params:
        eta = simulate_eta_fixed(stim, config)
    else:
        param_traj = generate_all_parameter_trajectories(
            len(stim), config, seed=seed
        )
        eta = simulate_eta_time_varying(stim, param_traj, config.Ts)

    return y_beta - eta


def generate_single_augmentation(
    y_beta: np.ndarray,
    config: ModulationConfig,
    seed: Optional[int] = None,
    use_fixed_params: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a single augmented dataset from a beta trajectory.

    Args:
        y_beta: Natural beta trajectory (n_steps,)
        config: Modulation configuration
        seed: Random seed
        use_fixed_params: Use fixed nominal parameters

    Returns:
        Dict with 'stim' and 'y_modulated' arrays
    """
    stim = generate_prbs_stimulation(len(y_beta), config, seed=seed)
    y_modulated = generate_modulated_output(y_beta, stim, config, use_fixed_params, seed)

    return {
        "stim": stim,
        "y_modulated": y_modulated,
    }


def generate_multiple_augmentations(
    y_beta: np.ndarray,
    config: ModulationConfig,
    n_augmentations: int = 1,
    base_seed: Optional[int] = 42,
    use_fixed_params: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Generate multiple augmented datasets from a beta trajectory.

    Args:
        y_beta: Natural beta trajectory (n_steps,)
        config: Modulation configuration
        n_augmentations: Number of augmentations to generate
        base_seed: Base random seed (each augmentation uses base_seed + i)
        use_fixed_params: Use fixed nominal parameters

    Returns:
        List of dicts, each with 'stim' and 'y_modulated' arrays
    """
    augmentations = []
    for i in range(n_augmentations):
        seed = None if base_seed is None else base_seed + i * 100
        aug = generate_single_augmentation(y_beta, config, seed, use_fixed_params)
        augmentations.append(aug)

    return augmentations
