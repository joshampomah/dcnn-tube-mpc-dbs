"""QP formulation for SCP subproblems in DCNN Tube MPC."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import cvxpy as cp

if TYPE_CHECKING:
    from dcnn_tube_mpc.controllers.scp_config import SCPConfig
    from dcnn_tube_mpc.models.dcnn_models import MultiStepDCNN


@dataclass
class QPSolution:
    """Result of QP subproblem solve."""

    u_optimal: np.ndarray
    s_max_optimal: np.ndarray
    s_min_optimal: np.ndarray
    cost: float
    status: str
    solve_time: float
    is_feasible: bool


class QPSubproblem:
    """DC-Convex QP solver for SCP iterations (CDC25 formulation)."""

    def __init__(
        self,
        N: int,
        config: "SCPConfig",
        predictor: "MultiStepDCNN" = None,
        weights_f1: List[List[np.ndarray]] = None,
        weights_f2: List[List[np.ndarray]] = None,
    ):
        self.N = N
        self.N_pred = N
        self.N_ctrl = getattr(config, 'control_horizon', N)
        self.config = config

        self.build_count = 0
        self.solve_count = 0

        self.uses_extended_horizon = self.N_pred > self.N_ctrl

        if predictor is not None:
            from dcnn_tube_mpc.analysis.jacobian import extract_weights_from_convex_nn
            n_networks = min(self.N_ctrl, len(predictor.networks))
            self.weights_f1 = [
                extract_weights_from_convex_nn(predictor.networks[i].f1)
                for i in range(n_networks)
            ]
            self.weights_f2 = [
                extract_weights_from_convex_nn(predictor.networks[i].f2)
                for i in range(n_networks)
            ]
            self.n_state = predictor.n_state
        elif weights_f1 is not None and weights_f2 is not None:
            self.weights_f1 = weights_f1
            self.weights_f2 = weights_f2
            self.n_state = weights_f1[0][0].shape[1] - 1
        else:
            self.weights_f1 = None
            self.weights_f2 = None
            self.n_state = getattr(config, 'n_state_y', 15) + getattr(config, 'n_state_u', 1)

        self.use_dc_mode = self.weights_f1 is not None

        self._problem = None
        self._current_z_k = None

    def solve(
        self,
        z_k: np.ndarray,
        y_nominal: np.ndarray,
        u_nominal: np.ndarray,
        u_prev: float,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        W_bounds: np.ndarray,
        f1_nominal: np.ndarray = None,
        f2_nominal: np.ndarray = None,
        device: str = "cpu",
        force_rebuild: bool = False,
        linear_jacobians: List[np.ndarray] = None,
    ) -> QPSolution:
        """Solve the QP subproblem."""
        start_time = time.time()

        needs_rebuild = (
            self._problem is None
            or force_rebuild
            or not np.allclose(z_k, self._current_z_k, rtol=1e-10)
        )
        if needs_rebuild:
            self._build_problem(z_k)

        self.param_y_nominal.value = y_nominal
        self.param_u_nominal.value = u_nominal[:self.N_ctrl]
        self.param_u_prev.value = u_prev
        self.param_w_min.value = W_bounds[:, 0]
        self.param_w_max.value = W_bounds[:, 1]

        for i in range(self.N_ctrl):
            self.params_J_f1[i].value = np.asarray(jacobians_f1[i]).reshape(1, i + 1)
            self.params_J_f2[i].value = np.asarray(jacobians_f2[i]).reshape(1, i + 1)

            if self.use_dc_mode:
                if f1_nominal is None or f2_nominal is None:
                    raise ValueError("f1_nominal and f2_nominal required for DC mode")
                self.params_f1_nom[i].value = f1_nominal[i]
                self.params_f2_nom[i].value = f2_nominal[i]
            else:
                J_diff = jacobians_f1[i] - jacobians_f2[i]
                self.params_J[i].value = np.asarray(J_diff).reshape(1, i + 1)

        if self.uses_extended_horizon:
            for i in range(self.N_ctrl, self.N_pred):
                self.params_y_nom_ext[i - self.N_ctrl].value = y_nominal[i]

        HARD_PENALTY = 1e9
        SOFT_PENALTY = 1e4

        self.param_slack_penalty.value = HARD_PENALTY

        try:
            self._problem.solve(
                solver=getattr(cp, self.config.solver, cp.CLARABEL),
                warm_start=True,
                verbose=self.config.solver_verbose
            )

            total_slack = np.sum(self.slack_max.value) + np.sum(self.slack_min.value)

            if self._problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or total_slack > 1e-3:
                if self.config.constraint_softening:
                    self.param_slack_penalty.value = SOFT_PENALTY
                    self._problem.solve(
                        solver=getattr(cp, self.config.solver, cp.CLARABEL),
                        warm_start=True,
                        verbose=self.config.solver_verbose
                    )

            solve_time = time.time() - start_time
            self.solve_count += 1

            is_feasible = self._problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

            if is_feasible:
                u_ctrl = self.u.value
                if self.uses_extended_horizon:
                    u_extended = np.zeros(self.N_pred, dtype=np.float32)
                    u_extended[:self.N_ctrl] = u_ctrl
                    u_extended[self.N_ctrl:] = u_ctrl[-1]
                else:
                    u_extended = np.asarray(u_ctrl, dtype=np.float32)

                return QPSolution(
                    u_optimal=u_extended,
                    s_max_optimal=np.asarray(self.s_max.value, dtype=np.float32),
                    s_min_optimal=np.asarray(self.s_min.value, dtype=np.float32),
                    cost=self._problem.value,
                    status=self._problem.status,
                    solve_time=solve_time,
                    is_feasible=True
                )
            else:
                return self._failure_result(u_nominal, solve_time, self._problem.status)

        except Exception as e:
            solve_time = time.time() - start_time
            return self._failure_result(u_nominal, solve_time, f"ERROR: {str(e)}")

    def _failure_result(self, u_nominal, solve_time, status):
        if len(u_nominal) < self.N_pred:
            u_extended = np.zeros(self.N_pred, dtype=np.float32)
            u_extended[:len(u_nominal)] = u_nominal
            u_extended[len(u_nominal):] = u_nominal[-1]
        else:
            u_extended = np.asarray(u_nominal[:self.N_pred], dtype=np.float32)

        return QPSolution(
            u_optimal=u_extended,
            s_max_optimal=np.zeros(self.N_pred, dtype=np.float32),
            s_min_optimal=np.zeros(self.N_pred, dtype=np.float32),
            cost=np.inf,
            status=status,
            solve_time=solve_time,
            is_feasible=False
        )

    def _build_problem(self, z_k: np.ndarray):
        """Construct the CVXPY problem graph."""
        self.build_count += 1
        self._current_z_k = z_k.copy()

        self.param_u_prev = cp.Parameter(name="u_prev")
        self.param_u_nominal = cp.Parameter(self.N_ctrl, name="u_nominal")
        self.param_y_nominal = cp.Parameter(self.N_pred, name="y_nominal")
        self.param_w_max = cp.Parameter(self.N_pred, name="w_max")
        self.param_w_min = cp.Parameter(self.N_pred, name="w_min")

        if self.uses_extended_horizon:
            self.params_y_nom_ext = [
                cp.Parameter(name=f"y_nom_ext_{i}")
                for i in range(self.N_pred - self.N_ctrl)
            ]
        else:
            self.params_y_nom_ext = []

        self.u = cp.Variable(self.N_ctrl, name="u")
        self.s_max = cp.Variable(self.N_pred, name="s_max")
        self.s_min = cp.Variable(self.N_pred, name="s_min")
        self.slack_max = cp.Variable(self.N_pred, nonneg=True, name="slack_max")
        self.slack_min = cp.Variable(self.N_pred, nonneg=True, name="slack_min")
        self.param_slack_penalty = cp.Parameter(nonneg=True, name="slack_penalty")

        constraints = []

        constraints.append(self.u >= self.config.u_min)
        constraints.append(self.u <= self.config.u_max)

        constraints.append(self.u[0] - self.param_u_prev <= self.config.delta_u_max)
        constraints.append(self.u[0] - self.param_u_prev >= -self.config.delta_u_max)
        for i in range(1, self.N_ctrl):
            constraints.append(self.u[i] - self.u[i-1] <= self.config.delta_u_max)
            constraints.append(self.u[i] - self.u[i-1] >= -self.config.delta_u_max)

        if self.use_dc_mode:
            constraints.extend(self._build_dc_constraints(z_k))
        else:
            constraints.extend(self._build_linearized_constraints())

        if self.uses_extended_horizon:
            constraints.extend(self._build_extended_constraints())

        constraints.append(self.s_max >= 0)
        constraints.append(self.s_min <= 0)
        constraints.append(self.s_max >= self.s_min)

        if self.config.y_max is not None:
            constraints.append(
                self.param_y_nominal + self.s_max <= self.config.y_max + self.slack_max
            )
        if self.config.y_min is not None:
            constraints.append(
                self.param_y_nominal + self.s_min >= self.config.y_min - self.slack_min
            )

        Q = self.config.Q
        R = self.config.R
        R_delta = getattr(self.config, "R_delta", 0.0)
        beta_0 = self.config.beta_0
        gamma = getattr(self.config, "tube_weight", 0.0)

        tracking_error = self.param_y_nominal + self.s_max - beta_0
        tracking_cost = Q * cp.sum_squares(cp.pos(tracking_error))

        control_cost = R * cp.sum_squares(self.u)

        if self.uses_extended_horizon:
            n_extended = self.N_pred - self.N_ctrl
            control_cost += R * n_extended * cp.square(self.u[-1])

        if R_delta > 0:
            delta_u = cp.diff(self.u)
            delta_u_0 = self.u[0] - self.param_u_prev
            rate_cost = R_delta * (cp.sum_squares(delta_u_0) + cp.sum_squares(delta_u))
        else:
            rate_cost = 0

        pe_gamma_val = getattr(self.config, "pe_gamma", 0.0)
        if pe_gamma_val > 0:
            delta_u_pe = cp.diff(self.u)
            delta_u_0_pe = self.u[0] - self.param_u_prev
            pe_cost = -pe_gamma_val * (cp.sum_squares(delta_u_0_pe) + cp.sum_squares(delta_u_pe))
        else:
            pe_cost = 0

        tube_cost = gamma * cp.sum_squares(self.s_max - self.s_min)
        slack_cost = self.param_slack_penalty * (cp.sum(self.slack_max) + cp.sum(self.slack_min))

        objective = cp.Minimize(tracking_cost + control_cost + rate_cost + pe_cost + tube_cost + slack_cost)

        self._problem = cp.Problem(objective, constraints)

    def _build_dc_constraints(self, z_k: np.ndarray) -> List:
        """Build DC constraints using CVXPY ICNN expressions."""
        from dcnn_tube_mpc.analysis.jacobian import forward_from_weights_cvxpy, build_icnn_cvxpy_params

        constraints = []

        self.params_f1_nom = [cp.Parameter(name=f"f1_nom_{i}") for i in range(self.N_ctrl)]
        self.params_f2_nom = [cp.Parameter(name=f"f2_nom_{i}") for i in range(self.N_ctrl)]
        self.params_J_f1 = [cp.Parameter((1, i + 1), name=f"J_f1_{i}") for i in range(self.N_ctrl)]
        self.params_J_f2 = [cp.Parameter((1, i + 1), name=f"J_f2_{i}") for i in range(self.N_ctrl)]

        z_k_const = z_k.astype(np.float64)

        self._nonneg_params_f1 = []
        self._nonneg_params_f2 = []

        for i in range(self.N_ctrl):
            n_u = i + 1

            v = self.u[:n_u] - self.param_u_nominal[:n_u]

            z_k_param = cp.Parameter(self.n_state, name=f"z_k_{i}")
            z_k_param.value = z_k_const

            nonneg_f1 = build_icnn_cvxpy_params(self.weights_f1[i], f"f1_{i}")
            nonneg_f2 = build_icnn_cvxpy_params(self.weights_f2[i], f"f2_{i}")
            self._nonneg_params_f1.append(nonneg_f1)
            self._nonneg_params_f2.append(nonneg_f2)

            f1_expr = forward_from_weights_cvxpy(
                z_k_param, self.u[:n_u], self.weights_f1[i], nonneg_f1
            )
            f2_expr = forward_from_weights_cvxpy(
                z_k_param, self.u[:n_u], self.weights_f2[i], nonneg_f2
            )

            upper_bound = f1_expr - self.params_f1_nom[i] - self.params_J_f2[i] @ v + self.param_w_max[i]
            constraints.append(self.s_max[i] >= upper_bound)

            lower_bound = -(f2_expr - self.params_f2_nom[i]) + self.params_J_f1[i] @ v + self.param_w_min[i]
            constraints.append(self.s_min[i] <= lower_bound)

        return constraints

    def _build_linearized_constraints(self) -> List:
        """Build fully linearized constraints (fallback mode)."""
        constraints = []

        self.params_J = [cp.Parameter((1, i + 1), name=f"J_{i}") for i in range(self.N_ctrl)]
        self.params_J_f1 = self.params_J
        self.params_J_f2 = [cp.Parameter((1, i + 1), name=f"J_f2_{i}") for i in range(self.N_ctrl)]
        self.params_f1_nom = [cp.Parameter(name=f"f1_nom_{i}") for i in range(self.N_ctrl)]
        self.params_f2_nom = [cp.Parameter(name=f"f2_nom_{i}") for i in range(self.N_ctrl)]

        for i in range(self.N_ctrl):
            n_u = i + 1
            v = self.u[:n_u] - self.param_u_nominal[:n_u]
            linearized_delta = self.params_J[i] @ v

            constraints.append(self.s_max[i] >= linearized_delta + self.param_w_max[i])
            constraints.append(self.s_min[i] <= linearized_delta + self.param_w_min[i])

        return constraints

    def _build_extended_constraints(self) -> List:
        """Build constraints for extended horizon steps."""
        constraints = []

        for i in range(self.N_ctrl, self.N_pred):
            constraints.append(self.s_max[i] >= self.param_w_max[i])
            constraints.append(self.s_min[i] <= self.param_w_min[i])

        return constraints


def create_qp_subproblem(
    config: "SCPConfig",
    predictor: "MultiStepDCNN" = None,
) -> QPSubproblem:
    """Factory function to create QP subproblem."""
    return QPSubproblem(
        N=config.prediction_horizon,
        config=config,
        predictor=predictor,
    )
