"""SCP Algorithm for DC-NN Tube MPC.

This module implements Algorithm 1 from the CDC25 paper - the Sequential
Convex Programming solver that iteratively solves convex QP subproblems
to find the optimal control sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from dcnn_tube_mpc.models.dcnn_models import MultiStepDCNN
    from dcnn_tube_mpc.controllers.scp_config import SCPConfig


@dataclass
class SCPResult:
    """Result of SCP algorithm."""

    u_optimal: np.ndarray
    y_nominal: np.ndarray
    u_nominal: np.ndarray
    J_optimal: float
    n_iterations: int
    converged: bool
    iteration_costs: List[float] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    status: str = "SUCCESS"
    s_max_final: Optional[np.ndarray] = None
    s_min_final: Optional[np.ndarray] = None


def solve_scp(
    z_k: np.ndarray,
    u_prev: float,
    u_initial: np.ndarray,
    predictor: "MultiStepDCNN",
    config: "SCPConfig",
    device: str = "cpu",
    gain_scale: float = 1.0,
) -> SCPResult:
    """Solve the SCP optimization problem (Algorithm 1 from CDC25).

    Args:
        z_k: Current state vector of shape (n_state,).
        u_prev: Previous control input u_{k-1}.
        u_initial: Initial guess for control sequence of shape (N_ctrl,).
        predictor: Trained MultiStepDCNN model.
        config: SCP configuration.
        device: Device for neural network computation.
        gain_scale: Scale factor for Jacobians (for online gain estimation).

    Returns:
        SCPResult with optimal control sequence and convergence info.
    """
    from dcnn_tube_mpc.analysis.jacobian import extract_weights_from_convex_nn
    from dcnn_tube_mpc.bounds.perturbation_bounds import compute_jacobians_for_trajectory

    N_pred = config.prediction_horizon
    N_ctrl = getattr(config, 'control_horizon', N_pred)
    uses_extended_horizon = N_pred > N_ctrl

    W_bounds = config.W_bounds
    if W_bounds is None:
        W_bounds = config.get_default_W_bounds()

    has_linear = hasattr(predictor, 'get_linear_jacobians')

    n_networks = min(N_ctrl, len(predictor.networks))
    cached_weights_f1 = [
        extract_weights_from_convex_nn(predictor.networks[i].f1)
        for i in range(n_networks)
    ]
    cached_weights_f2 = [
        extract_weights_from_convex_nn(predictor.networks[i].f2)
        for i in range(n_networks)
    ]

    linear_jacobians = None
    if has_linear:
        linear_jacobians = [
            predictor.get_linear_jacobians(i + 1)
            for i in range(N_ctrl)
        ]

    u_initial_ctrl = u_initial[:N_ctrl] if len(u_initial) >= N_ctrl else u_initial
    u_nominal_ctrl = np.clip(u_initial_ctrl.copy(), config.u_min, config.u_max)

    y_nominal, f1_nominal, f2_nominal = _compute_nominal_predictions_dc_extended(
        predictor, z_k, u_nominal_ctrl, N_pred, device, has_linear=has_linear
    )

    solver_type = config.qp_solver_type
    use_direct = solver_type in ("direct", "osqp", "piqp", "proxqp",
                                  "stable-neuron", "stable-neuron-proxqp", "stable-neuron-daqp")
    qp = None

    if solver_type == "direct":
        from dcnn_tube_mpc.solvers.direct_qp_solver import create_direct_solver
        qp = create_direct_solver(predictor, config)
    elif solver_type == "osqp":
        from dcnn_tube_mpc.solvers.osqp_solver import create_osqp_solver
        qp = create_osqp_solver(predictor, config)
    elif solver_type == "piqp":
        from dcnn_tube_mpc.solvers.piqp_solver import create_piqp_solver
        qp = create_piqp_solver(predictor, config)

    if use_direct and qp is None:
        use_direct = False

    if not use_direct:
        from dcnn_tube_mpc.solvers.qp_solver import QPSubproblem
        qp = QPSubproblem(N=N_pred, config=config, predictor=predictor)

    j = 1
    delta_J = np.inf
    J_prev = np.inf
    iteration_costs = []
    iteration_times = []
    last_qp_solution = None

    while j <= config.maxiters and delta_J > config.delta_J_min:
        jacobians_f1, jacobians_f2 = compute_jacobians_for_trajectory(
            predictor, z_k, u_nominal_ctrl, device,
            cached_weights_f1=cached_weights_f1,
            cached_weights_f2=cached_weights_f2,
        )

        for i in range(N_ctrl):
            assert jacobians_f1[i].shape == (1, i + 1), \
                f"Jacobian f1 step {i} shape mismatch: {jacobians_f1[i].shape} != (1, {i+1})"
            assert jacobians_f2[i].shape == (1, i + 1), \
                f"Jacobian f2 step {i} shape mismatch: {jacobians_f2[i].shape} != (1, {i+1})"

        if gain_scale != 1.0:
            jacobians_f1 = [J * gain_scale for J in jacobians_f1]
            jacobians_f2 = [J * gain_scale for J in jacobians_f2]
            if linear_jacobians is not None:
                linear_jacobians_scaled = [J * gain_scale for J in linear_jacobians]
            else:
                linear_jacobians_scaled = None
        else:
            linear_jacobians_scaled = linear_jacobians

        if use_direct:
            qp_solution = qp.solve(
                z_k=z_k,
                y_nominal=y_nominal,
                u_nominal=u_nominal_ctrl,
                u_prev=u_prev,
                jacobians_f1=jacobians_f1,
                jacobians_f2=jacobians_f2,
                W_bounds=W_bounds,
                f1_nominal=f1_nominal,
                f2_nominal=f2_nominal,
                linear_jacobians=linear_jacobians_scaled,
            )
        else:
            qp_solution = qp.solve(
                z_k=z_k,
                y_nominal=y_nominal,
                u_nominal=u_nominal_ctrl,
                u_prev=u_prev,
                jacobians_f1=jacobians_f1,
                jacobians_f2=jacobians_f2,
                W_bounds=W_bounds,
                f1_nominal=f1_nominal,
                f2_nominal=f2_nominal,
                device=device,
                force_rebuild=(j == 1),
            )

        iteration_times.append(qp_solution.solve_time)
        last_qp_solution = qp_solution

        if not qp_solution.is_feasible:
            u_safe = np.full(N_pred, np.clip(u_prev, config.u_min, config.u_max), dtype=np.float32)
            return SCPResult(
                u_optimal=u_safe,
                y_nominal=y_nominal,
                u_nominal=u_safe,
                J_optimal=np.inf,
                n_iterations=j,
                converged=False,
                iteration_costs=iteration_costs,
                iteration_times=iteration_times,
                status=f"INFEASIBLE at iteration {j}: {qp_solution.status}",
            )

        J_j = qp_solution.cost
        u_optimal_full = np.asarray(qp_solution.u_optimal, dtype=np.float32)

        if j == 1 and config.delta_u_tol > 0:
            max_delta_u = float(np.max(np.abs(u_optimal_full[:N_ctrl] - u_nominal_ctrl)))
            if max_delta_u <= config.delta_u_tol:
                iteration_costs.append(J_j)
                u_nominal_ctrl = u_optimal_full[:N_ctrl].copy()
                y_nominal, f1_nominal, f2_nominal = _compute_nominal_predictions_dc_extended(
                    predictor, z_k, u_nominal_ctrl, N_pred, device, has_linear=has_linear
                )
                delta_J = 0.0
                break

        if J_prev == np.inf:
            delta_J = np.inf
        else:
            delta_J = abs(J_j - J_prev)

        iteration_costs.append(J_j)

        u_nominal_ctrl = u_optimal_full[:N_ctrl].copy()
        y_nominal, f1_nominal, f2_nominal = _compute_nominal_predictions_dc_extended(
            predictor, z_k, u_nominal_ctrl, N_pred, device, has_linear=has_linear
        )

        J_prev = J_j
        j += 1

    converged = delta_J <= config.delta_J_min or (config.delta_u_tol > 0 and delta_J == 0.0)

    u_optimal_full = np.zeros(N_pred, dtype=np.float32)
    u_optimal_full[:N_ctrl] = u_nominal_ctrl
    if uses_extended_horizon:
        u_optimal_full[N_ctrl:] = u_nominal_ctrl[-1]

    s_max_final = None
    s_min_final = None
    if last_qp_solution is not None and hasattr(last_qp_solution, 's_max_optimal'):
        s_max_final = np.asarray(last_qp_solution.s_max_optimal, dtype=np.float32)
        s_min_final = np.asarray(last_qp_solution.s_min_optimal, dtype=np.float32)

    return SCPResult(
        u_optimal=u_optimal_full,
        y_nominal=y_nominal,
        u_nominal=u_optimal_full,
        J_optimal=J_prev,
        n_iterations=j - 1,
        converged=converged,
        iteration_costs=iteration_costs,
        iteration_times=iteration_times,
        status="CONVERGED" if converged else "MAX_ITERATIONS",
        s_max_final=s_max_final,
        s_min_final=s_min_final,
    )


def _compute_nominal_predictions_dc_extended(
    predictor,
    z_k: np.ndarray,
    u_nominal_ctrl: np.ndarray,
    N_pred: int,
    device: str = "cpu",
    has_linear: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute nominal predictions for extended horizon."""
    import torch

    N_ctrl = len(u_nominal_ctrl)

    if hasattr(predictor, 'networks') and len(predictor.networks) > 0:
        try:
            param = next(predictor.networks[0].parameters())
            torch_device = param.device
        except StopIteration:
            torch_device = torch.device(device)
    else:
        torch_device = torch.device(device)

    y_nominal = np.zeros(N_pred, dtype=np.float32)
    f1_nominal = np.zeros(N_ctrl, dtype=np.float32)
    f2_nominal = np.zeros(N_ctrl, dtype=np.float32)

    with torch.no_grad():
        for i in range(N_ctrl):
            n_u = i + 1
            inputs = np.hstack([z_k, u_nominal_ctrl[:n_u]])
            inputs_tensor = torch.tensor(
                inputs.reshape(1, -1), dtype=torch.float32, device=torch_device
            )

            network = predictor.networks[i]
            f1_val = network.f1(inputs_tensor).cpu().numpy()[0, 0]
            f2_val = network.f2(inputs_tensor).cpu().numpy()[0, 0]

            f1_nominal[i] = f1_val
            f2_nominal[i] = f2_val

            if has_linear:
                linear_pred = predictor.predict_linear(z_k, u_nominal_ctrl, i + 1)
                y_nominal[i] = linear_pred + f1_val - f2_val
            else:
                y_nominal[i] = f1_val - f2_val

    if N_pred > N_ctrl:
        y_nominal[N_ctrl:] = y_nominal[N_ctrl - 1]

    return y_nominal, f1_nominal, f2_nominal


def create_warm_start(
    previous_u_optimal: np.ndarray,
    N_ctrl: int = None,
    default_value: Optional[float] = None,
) -> np.ndarray:
    """Create warm-start initial guess from previous solution."""
    if N_ctrl is None:
        N_ctrl = len(previous_u_optimal)

    u_ctrl = previous_u_optimal[:N_ctrl]

    if default_value is None:
        default_value = u_ctrl[-1]

    return np.append(u_ctrl[1:], default_value)


def solve_scp_with_warm_start(
    z_k: np.ndarray,
    u_prev: float,
    previous_result: Optional[SCPResult],
    predictor: "MultiStepDCNN",
    config: "SCPConfig",
    device: str = "cpu",
) -> SCPResult:
    """Solve SCP with automatic warm-starting from previous solution."""
    N_ctrl = getattr(config, 'control_horizon', config.prediction_horizon)

    warm_start_u = None
    if previous_result is not None and config.use_warm_start:
        warm_start_u = previous_result.u_optimal

    if warm_start_u is not None:
        u_initial = create_warm_start(warm_start_u, N_ctrl=N_ctrl)
        u_initial = np.clip(u_initial, config.u_min, config.u_max)
    else:
        u_initial = np.full(N_ctrl, u_prev, dtype=np.float32)

    return solve_scp(z_k, u_prev, u_initial, predictor, config, device)
