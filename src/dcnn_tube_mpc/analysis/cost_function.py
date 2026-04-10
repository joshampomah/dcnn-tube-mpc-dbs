# Canonical owner: closed-loop-dbs-bench
"""Asymmetric cost function for DC-NN Tube MPC.

This module implements equation (8) from the CDC25 paper - the worst-case
MPC cost with asymmetric tracking penalty that only penalizes beta activity
exceeding the pathological threshold.

The cost function is:
    J = sum_i max_{s in S_i} (Q * [y^0_i + s - beta_0]_{>=0}^2 + R * u_i^2)

where [x]_{>=0} = max(x, 0) applies the asymmetric penalty.

Example:
    >>> from dcnn_tube_mpc.analysis.cost_function import compute_worst_case_cost
    >>> J = compute_worst_case_cost(y_nominal, s_bounds, u, Q=50000, R=1, beta_0=2.3)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import cvxpy as cp


def asymmetric_penalty(y: float, beta_0: float) -> float:
    """Compute asymmetric penalty [y - beta_0]_{>=0}.

    Only penalizes y values exceeding the pathological threshold beta_0.

    Args:
        y: Beta activity value.
        beta_0: Pathological threshold.

    Returns:
        max(y - beta_0, 0)

    Example:
        >>> asymmetric_penalty(2.5, 2.3)  # Above threshold
        0.2
        >>> asymmetric_penalty(2.0, 2.3)  # Below threshold
        0.0
    """
    return max(y - beta_0, 0.0)


def compute_step_cost(
    y: float,
    u: float,
    Q: float,
    R: float,
    beta_0: float,
) -> float:
    """Compute cost for a single time step.

    Cost = Q * [y - beta_0]_{>=0}^2 + R * u^2

    Args:
        y: Beta activity value.
        u: Control input (stimulation).
        Q: Tracking weight.
        R: Control effort weight.
        beta_0: Pathological threshold.

    Returns:
        Scalar cost value.
    """
    penalty = asymmetric_penalty(y, beta_0)
    return Q * penalty**2 + R * u**2


def compute_worst_case_step_cost(
    y_nominal: float,
    s_min: float,
    s_max: float,
    u: float,
    Q: float,
    R: float,
    beta_0: float,
) -> float:
    """Compute worst-case cost for a single step given perturbation bounds.

    Implements: max_{s in [s_min, s_max]} (Q * [y^0 + s - beta_0]_{>=0}^2 + R * u^2)

    Args:
        y_nominal: Nominal prediction y^0.
        s_min: Lower perturbation bound.
        s_max: Upper perturbation bound.
        u: Control input.
        Q: Tracking weight.
        R: Control effort weight.
        beta_0: Pathological threshold.

    Returns:
        Worst-case cost for this step.
    """
    y_worst = y_nominal + s_max
    penalty_worst = asymmetric_penalty(y_worst, beta_0)
    control_cost = R * u**2
    return Q * penalty_worst**2 + control_cost


def compute_worst_case_cost(
    y_nominal: np.ndarray,
    s_bounds: np.ndarray,
    u: np.ndarray,
    Q: float,
    R: float,
    beta_0: float,
) -> float:
    """Compute total worst-case cost over prediction horizon.

    Implements equation (8) from CDC25:
        J = sum_{i=0}^{N-1} max_{s in S_i} (Q*[y^0_i + s - beta_0]_{>=0}^2 + R*u_i^2)

    Args:
        y_nominal: Nominal predictions of shape (N,).
        s_bounds: Perturbation bounds of shape (N, 2) where s_bounds[i] = [s_min, s_max].
        u: Control sequence of shape (N,).
        Q: Tracking weight.
        R: Control effort weight.
        beta_0: Pathological threshold.

    Returns:
        Total worst-case cost (scalar).
    """
    N = len(y_nominal)
    total_cost = 0.0

    for i in range(N):
        s_min, s_max = s_bounds[i]
        step_cost = compute_worst_case_step_cost(
            y_nominal[i], s_min, s_max, u[i], Q, R, beta_0
        )
        total_cost += step_cost

    return total_cost


# =============================================================================
# CVXPY Expression Builders for QP Optimization
# =============================================================================


def build_tracking_cost_expression(
    y_nominal: np.ndarray,
    s_max_vars: "cp.Variable",
    Q: float,
    beta_0: float,
) -> "cp.Expression":
    """Build CVXPY expression for tracking cost.

    Args:
        y_nominal: Nominal predictions of shape (N,).
        s_max_vars: CVXPY Variable for upper perturbation bounds (N,).
        Q: Tracking weight.
        beta_0: Pathological threshold.

    Returns:
        CVXPY expression for tracking cost.
    """
    import cvxpy as cp

    N = len(y_nominal)
    tracking_cost = 0

    for i in range(N):
        deviation = y_nominal[i] + s_max_vars[i] - beta_0
        tracking_cost += Q * cp.pos(deviation) ** 2

    return tracking_cost


def build_control_cost_expression(
    u_vars: "cp.Variable",
    R: float,
) -> "cp.Expression":
    """Build CVXPY expression for control effort cost.

    Args:
        u_vars: CVXPY Variable for control inputs (N,).
        R: Control effort weight.

    Returns:
        CVXPY expression: R * sum(u^2)
    """
    import cvxpy as cp

    return R * cp.sum_squares(u_vars)


def build_total_cost_expression(
    y_nominal: np.ndarray,
    s_max_vars: "cp.Variable",
    u_vars: "cp.Variable",
    Q: float,
    R: float,
    beta_0: float,
) -> "cp.Expression":
    """Build complete CVXPY cost expression for MPC optimization.

    Args:
        y_nominal: Nominal predictions of shape (N,).
        s_max_vars: CVXPY Variable for upper perturbation bounds (N,).
        u_vars: CVXPY Variable for control inputs (N,).
        Q: Tracking weight.
        R: Control effort weight.
        beta_0: Pathological threshold.

    Returns:
        CVXPY expression for total cost.
    """
    tracking = build_tracking_cost_expression(y_nominal, s_max_vars, Q, beta_0)
    control = build_control_cost_expression(u_vars, R)
    return tracking + control


def compute_cost_gradient(
    y_nominal: np.ndarray,
    s_bounds: np.ndarray,
    u: np.ndarray,
    Q: float,
    R: float,
    beta_0: float,
) -> np.ndarray:
    """Compute gradient of worst-case cost w.r.t. control inputs.

    Args:
        y_nominal: Nominal predictions of shape (N,).
        s_bounds: Perturbation bounds of shape (N, 2).
        u: Control sequence of shape (N,).
        Q: Tracking weight.
        R: Control effort weight.
        beta_0: Pathological threshold.

    Returns:
        Gradient array of shape (N,).
    """
    N = len(u)
    grad = np.zeros(N)

    for i in range(N):
        grad[i] = 2 * R * u[i]

    return grad
