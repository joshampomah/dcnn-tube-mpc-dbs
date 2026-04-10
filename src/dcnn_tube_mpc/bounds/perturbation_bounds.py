"""Perturbation bounds computation for DC-NN Tube MPC."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn


def compute_jacobians_for_trajectory(
    predictor,
    z_k: np.ndarray,
    u_nominal: np.ndarray,
    device: str = "cpu",
    cached_weights_f1: list = None,
    cached_weights_f2: list = None,
) -> Tuple[list, list]:
    """Compute Jacobians of f1 and f2 for all prediction steps.

    Args:
        predictor: MultiStepDCNN model.
        z_k: Current state vector of shape (n_state,).
        u_nominal: Nominal control sequence of shape (N_ctrl,).
        device: Device for computation.
        cached_weights_f1: Pre-extracted weights for f1 networks.
        cached_weights_f2: Pre-extracted weights for f2 networks.

    Returns:
        Tuple (jacobians_f1, jacobians_f2), each list of arrays.
    """
    N = len(u_nominal)
    jacobians_f1 = []
    jacobians_f2 = []

    for i in range(N):
        n_u = i + 1
        network = predictor.networks[i]

        weights_f1 = cached_weights_f1[i] if cached_weights_f1 is not None else None
        weights_f2 = cached_weights_f2[i] if cached_weights_f2 is not None else None

        jac_f1 = _compute_component_jacobian(
            network.f1, z_k, u_nominal[:n_u], device, weights=weights_f1
        )
        jacobians_f1.append(jac_f1)

        jac_f2 = _compute_component_jacobian(
            network.f2, z_k, u_nominal[:n_u], device, weights=weights_f2
        )
        jacobians_f2.append(jac_f2)

    return jacobians_f1, jacobians_f2


def _compute_component_jacobian(
    model,
    x: np.ndarray,
    u: np.ndarray,
    device: str = "cpu",
    weights: list = None,
) -> np.ndarray:
    """Compute Jacobian of a single convex component w.r.t. control inputs."""
    from dcnn_tube_mpc.analysis.jacobian import compute_component_jacobian_analytical

    return compute_component_jacobian_analytical(model, x, u, weights=weights)


def compute_single_step_bounds(
    f1,
    f2,
    z_k: np.ndarray,
    u_nominal: np.ndarray,
    u_candidate: np.ndarray,
    y_nominal: float,
    W_i: Tuple[float, float],
    jacobian_f1: np.ndarray,
    jacobian_f2: np.ndarray,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Compute perturbation bounds for a single prediction step (CDC25 eqs. 6-7)."""
    import torch

    device = torch.device(device)
    w_min, w_max = W_i

    v = u_candidate - u_nominal

    input_nominal = np.hstack([z_k, u_nominal]).reshape(1, -1)
    input_candidate = np.hstack([z_k, u_candidate]).reshape(1, -1)

    input_nominal_t = torch.tensor(input_nominal, dtype=torch.float32, device=device)
    input_candidate_t = torch.tensor(input_candidate, dtype=torch.float32, device=device)

    f1 = f1.to(device).eval()
    f2 = f2.to(device).eval()

    with torch.no_grad():
        f1_at_u = f1(input_candidate_t).cpu().numpy()[0, 0]
        f2_at_u = f2(input_candidate_t).cpu().numpy()[0, 0]
        f1_at_u0 = f1(input_nominal_t).cpu().numpy()[0, 0]
        f2_at_u0 = f2(input_nominal_t).cpu().numpy()[0, 0]

    jac_f1_v = float((jacobian_f1 @ v).item() if hasattr(jacobian_f1 @ v, 'item') else (jacobian_f1 @ v)[0])
    jac_f2_v = float((jacobian_f2 @ v).item() if hasattr(jacobian_f2 @ v, 'item') else (jacobian_f2 @ v)[0])

    s_max = f1_at_u - f2_at_u0 - jac_f2_v + w_max - y_nominal
    s_min = -f2_at_u + f1_at_u0 + jac_f1_v + w_min - y_nominal

    return float(s_max), float(s_min)
