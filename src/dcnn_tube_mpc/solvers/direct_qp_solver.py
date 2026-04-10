"""Direct QP solver for DC-MPC using CLARABEL."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import sparse


@dataclass
class DirectQPSolution:
    """Result of direct QP solve."""
    u_optimal: np.ndarray
    s_max_optimal: np.ndarray
    s_min_optimal: np.ndarray
    cost: float
    status: str
    solve_time: float
    is_feasible: bool
    setup_time: float = 0.0
    update_time: float = 0.0


class QPMatrixBuilder:
    """Builds and updates QP matrices for DC-MPC."""

    def __init__(
        self,
        N: int,
        n_state: int,
        weights_f1: List[List[np.ndarray]],
        weights_f2: List[List[np.ndarray]],
        Q: float = 50000.0,
        R: float = 1.0,
        R_delta: float = 0.0,
        tube_weight: float = 0.0,
        beta_0: float = 2.3,
        u_min: float = 0.0,
        u_max: float = 3.0,
        delta_u_max: float = 0.5,
        decimation: int = 1,
        pe_gamma: float = 0.0,
    ):
        self.N = N
        self.n_state = n_state
        self.weights_f1 = weights_f1
        self.weights_f2 = weights_f2

        self.Q = Q
        self.R = R
        self.R_delta = R_delta
        self.pe_gamma = pe_gamma
        self.tube_weight = tube_weight
        self.beta_0 = beta_0
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.decimation = decimation

        self.n_hidden = weights_f1[0][0].shape[0]
        self.n_layers = 1 + (len(weights_f1[0]) - 4) // 4

        self.n_aux_per_net = self.n_hidden * self.n_layers
        self.n_aux_total = 2 * N * self.n_aux_per_net
        self.n_vars = N + N + N + N + N + N + self.n_aux_total

        self.idx_u = slice(0, N)
        self.idx_s_max = slice(N, 2*N)
        self.idx_s_min = slice(2*N, 3*N)
        self.idx_t = slice(3*N, 4*N)
        self.idx_y_f1 = slice(4*N, 5*N)
        self.idx_y_f2 = slice(5*N, 6*N)
        self.idx_aux = slice(6*N, self.n_vars)

        self._setup_done = False

    def setup(self):
        """Build the QP structure (one-time cost)."""
        self.P = self._build_cost_matrix()

        (self.A_eq, self.b_eq_template,
         self.A_ineq, self.h_template,
         self.n_eq, self.n_ineq) = self._build_constraints()

        self._setup_done = True

    def _build_cost_matrix(self) -> sparse.csc_matrix:
        """Build the quadratic cost matrix P."""
        N = self.N
        rows, cols, data = [], [], []

        for i in range(N):
            t_idx = 3*N + i
            rows.append(t_idx)
            cols.append(t_idx)
            data.append(2 * self.Q)

        for i in range(N):
            rows.append(i)
            cols.append(i)
            data.append(2 * self.R)

        for i in range(N):
            s_max_idx = N + i
            s_min_idx = 2*N + i
            rows.extend([s_max_idx, s_min_idx, s_max_idx, s_min_idx])
            cols.extend([s_max_idx, s_min_idx, s_min_idx, s_max_idx])
            data.extend([2 * self.tube_weight, 2 * self.tube_weight,
                        -2 * self.tube_weight, -2 * self.tube_weight])

        if self.R_delta > 0:
            for i in range(1, N):
                rows.extend([i, i-1, i, i-1])
                cols.extend([i, i-1, i-1, i])
                data.extend([2*self.R_delta, 2*self.R_delta,
                           -2*self.R_delta, -2*self.R_delta])

        if self.pe_gamma > 0:
            for i in range(1, N):
                rows.extend([i, i-1, i, i-1])
                cols.extend([i, i-1, i-1, i])
                data.extend([-2*self.pe_gamma, -2*self.pe_gamma,
                            2*self.pe_gamma, 2*self.pe_gamma])

        P = sparse.csc_matrix((data, (rows, cols)), shape=(self.n_vars, self.n_vars))
        return P

    def _build_constraints(self):
        """Build constraint matrices."""
        N = self.N
        n_hidden = self.n_hidden

        eq_rows, eq_cols, eq_data = [], [], []
        eq_b = []
        eq_row_idx = 0

        ineq_rows, ineq_cols, ineq_data = [], [], []
        ineq_h = []
        ineq_row_idx = 0

        def add_equality(coeffs, rhs):
            nonlocal eq_row_idx
            for idx, coef in coeffs:
                eq_rows.append(eq_row_idx)
                eq_cols.append(idx)
                eq_data.append(coef)
            eq_b.append(rhs)
            eq_row_idx += 1

        def add_inequality(coeffs, rhs):
            nonlocal ineq_row_idx
            for idx, coef in coeffs:
                ineq_rows.append(ineq_row_idx)
                ineq_cols.append(idx)
                ineq_data.append(coef)
            ineq_h.append(rhs)
            ineq_row_idx += 1

        aux_offset = 6 * N
        for step in range(N):
            for net_idx, (weights, y_idx_base) in enumerate([
                (self.weights_f1[step], 4*N),
                (self.weights_f2[step], 5*N),
            ]):
                net_aux_start = aux_offset + (step * 2 + net_idx) * self.n_aux_per_net
                W_out = np.maximum(weights[-2], 0)
                b_out = float(weights[-1][0])
                y_idx = y_idx_base + step
                h_last_start = net_aux_start + (self.n_layers - 1) * n_hidden

                coeffs = [(y_idx, 1.0)]
                for k in range(n_hidden):
                    if abs(W_out[0, k]) > 1e-10:
                        coeffs.append((h_last_start + k, -W_out[0, k]))
                add_equality(coeffs, b_out)

        if self.decimation > 1:
            for i in range(1, N):
                if i % self.decimation != 0:
                    add_equality([(i, 1.0), (i - 1, -1.0)], 0.0)

        self._u_bound_start = ineq_row_idx
        for i in range(N):
            add_inequality([(i, -1.0)], -self.u_min)
            add_inequality([(i, 1.0)], self.u_max)

        self._rate_idx = ineq_row_idx
        add_inequality([(0, 1.0)], 0.0)
        add_inequality([(0, -1.0)], 0.0)

        for i in range(1, N):
            add_inequality([(i, 1.0), (i-1, -1.0)], self.delta_u_max)
            add_inequality([(i, -1.0), (i-1, 1.0)], self.delta_u_max)

        for i in range(N):
            add_inequality([(N + i, -1.0)], 0.0)
            add_inequality([(2*N + i, 1.0)], 0.0)

        for i in range(N):
            add_inequality([(2*N + i, 1.0), (N + i, -1.0)], 0.0)

        self._t_ineq_start = ineq_row_idx
        for i in range(N):
            add_inequality([(N + i, 1.0), (3*N + i, -1.0)], 0.0)

        for i in range(N):
            add_inequality([(3*N + i, -1.0)], 0.0)

        self._icnn_ineq_start = ineq_row_idx
        for step in range(N):
            n_u = step + 1

            for net_idx, weights in enumerate([self.weights_f1[step], self.weights_f2[step]]):
                net_aux_start = aux_offset + (step * 2 + net_idx) * self.n_aux_per_net

                W0 = weights[0]
                W0_u = W0[:, self.n_state:self.n_state + n_u]

                for j in range(n_hidden):
                    h_idx = net_aux_start + j
                    coeffs = [(h_idx, -1.0)]
                    for k in range(n_u):
                        if abs(W0_u[j, k]) > 1e-10:
                            coeffs.append((k, W0_u[j, k]))
                    add_inequality(coeffs, 0.0)

                for j in range(n_hidden):
                    add_inequality([(net_aux_start + j, -1.0)], 0.0)

                n_internal = (len(weights) - 4) // 4
                h_prev_start = net_aux_start

                for layer in range(n_internal):
                    h_curr_start = net_aux_start + (layer + 1) * n_hidden

                    Wx = np.maximum(weights[2 + layer*4], 0)
                    W0_layer = weights[2 + layer*4 + 2]
                    W0_u_layer = W0_layer[:, self.n_state:self.n_state + n_u]

                    for j in range(n_hidden):
                        h_curr_idx = h_curr_start + j
                        coeffs = [(h_curr_idx, -1.0)]

                        for k in range(n_hidden):
                            if abs(Wx[j, k]) > 1e-10:
                                coeffs.append((h_prev_start + k, Wx[j, k]))

                        for k in range(n_u):
                            if abs(W0_u_layer[j, k]) > 1e-10:
                                coeffs.append((k, W0_u_layer[j, k]))

                        add_inequality(coeffs, 0.0)

                    for j in range(n_hidden):
                        add_inequality([(h_curr_start + j, -1.0)], 0.0)

                    h_prev_start = h_curr_start

        self._dc_upper_start = ineq_row_idx
        self._dc_upper_u_indices = []
        for step in range(N):
            n_u = step + 1
            y_f1_idx = 4*N + step
            s_max_idx = N + step

            coeffs = [(y_f1_idx, 1.0), (s_max_idx, -1.0)]
            u_start_idx = len(ineq_data)
            for k in range(n_u):
                coeffs.append((k, 0.0))
            self._dc_upper_u_indices.append((ineq_row_idx, u_start_idx + 2, n_u))
            add_inequality(coeffs, 0.0)

        self._dc_lower_start = ineq_row_idx
        self._dc_lower_u_indices = []
        for step in range(N):
            n_u = step + 1
            y_f2_idx = 5*N + step
            s_min_idx = 2*N + step

            coeffs = [(s_min_idx, 1.0), (y_f2_idx, 1.0)]
            u_start_idx = len(ineq_data)
            for k in range(n_u):
                coeffs.append((k, 0.0))
            self._dc_lower_u_indices.append((ineq_row_idx, u_start_idx + 2, n_u))
            add_inequality(coeffs, 0.0)

        n_eq = eq_row_idx
        n_ineq = ineq_row_idx

        A_eq = sparse.csc_matrix((eq_data, (eq_rows, eq_cols)),
                                  shape=(n_eq, self.n_vars)) if n_eq > 0 else None
        b_eq = np.array(eq_b) if n_eq > 0 else None

        A_ineq = sparse.csc_matrix((ineq_data, (ineq_rows, ineq_cols)),
                                    shape=(n_ineq, self.n_vars))
        h = np.array(ineq_h)

        self._ineq_data = np.array(ineq_data, dtype=np.float64)
        self._A_ineq_rows = ineq_rows
        self._A_ineq_cols = ineq_cols

        return A_eq, b_eq, A_ineq, h, n_eq, n_ineq

    def compute_linear_cost(self, y_nominal: np.ndarray, u_prev: float) -> np.ndarray:
        """Compute the linear cost vector q."""
        q = np.zeros(self.n_vars)
        if self.R_delta > 0:
            q[0] -= 2 * self.R_delta * u_prev
        if self.pe_gamma > 0:
            q[0] += 2 * self.pe_gamma * u_prev
        return q

    def update_constraints(
        self,
        z_k: np.ndarray,
        u_prev: float,
        u_nominal: np.ndarray,
        y_nominal: np.ndarray,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        f1_nominal: np.ndarray,
        f2_nominal: np.ndarray,
        W_bounds: np.ndarray,
        linear_jacobians: List[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, sparse.csc_matrix]:
        """Update constraint RHS and Jacobian entries."""
        N = self.N
        n_hidden = self.n_hidden

        h = self.h_template.copy()
        ineq_data = self._ineq_data.copy()

        h[self._rate_idx] = u_prev + self.delta_u_max
        h[self._rate_idx + 1] = -u_prev + self.delta_u_max

        for i in range(N):
            h[self._t_ineq_start + i] = self.beta_0 - y_nominal[i]

        ineq_idx = self._icnn_ineq_start
        aux_offset = 6 * N

        for step in range(N):
            for net_idx, weights in enumerate([self.weights_f1[step], self.weights_f2[step]]):
                W0 = weights[0]
                b0 = weights[1]
                W0_z = W0[:, :self.n_state]
                c0 = W0_z @ z_k + b0

                for j in range(n_hidden):
                    h[ineq_idx] = -c0[j]
                    ineq_idx += 1

                ineq_idx += n_hidden

                n_internal = (len(weights) - 4) // 4
                for layer in range(n_internal):
                    bx = weights[2 + layer*4 + 1]
                    W0_layer = weights[2 + layer*4 + 2]
                    b0_layer = weights[2 + layer*4 + 3]
                    W0_z_layer = W0_layer[:, :self.n_state]
                    c_layer = W0_z_layer @ z_k + bx + b0_layer

                    for j in range(n_hidden):
                        h[ineq_idx] = -c_layer[j]
                        ineq_idx += 1

                    ineq_idx += n_hidden

        for step in range(N):
            n_u = step + 1
            J_f2 = jacobians_f2[step].flatten()

            if linear_jacobians is not None:
                J_linear = linear_jacobians[step].flatten()
                J_f2_eff = J_f2 - J_linear
            else:
                J_f2_eff = J_f2

            w_max = W_bounds[step, 1]

            row_idx, data_start, count = self._dc_upper_u_indices[step]
            for k in range(count):
                ineq_data[data_start + k] = -J_f2_eff[k]

            h[row_idx] = f1_nominal[step] - np.dot(J_f2_eff, u_nominal[:n_u]) - w_max

        for step in range(N):
            n_u = step + 1
            J_f1 = jacobians_f1[step].flatten()

            if linear_jacobians is not None:
                J_linear = linear_jacobians[step].flatten()
                J_f1_eff = J_f1 + J_linear
            else:
                J_f1_eff = J_f1

            w_min = W_bounds[step, 0]

            row_idx, data_start, count = self._dc_lower_u_indices[step]
            for k in range(count):
                ineq_data[data_start + k] = -J_f1_eff[k]

            h[row_idx] = f2_nominal[step] - np.dot(J_f1_eff, u_nominal[:n_u]) + w_min

        A_ineq = sparse.csc_matrix(
            (ineq_data, (self._A_ineq_rows, self._A_ineq_cols)),
            shape=(len(h), self.n_vars)
        )

        return self.b_eq_template, h, A_ineq


class DirectQPSolver:
    """Direct CLARABEL solver for DC-MPC without CVXPY overhead."""

    def __init__(self, builder: QPMatrixBuilder):
        self._builder = builder
        self._setup_done = False

        self.N = builder.N
        self.idx_u = builder.idx_u
        self.idx_s_max = builder.idx_s_max
        self.idx_s_min = builder.idx_s_min

    def setup(self):
        """Build the QP structure (one-time cost)."""
        import clarabel

        start = time.perf_counter()

        if not self._builder._setup_done:
            self._builder.setup()

        cones = [clarabel.NonnegativeConeT(self._builder.n_ineq)]
        if self._builder.n_eq > 0:
            cones.insert(0, clarabel.ZeroConeT(self._builder.n_eq))
        self._cone_dims = cones

        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self._settings.max_iter = 100
        self._settings.tol_gap_abs = 1e-6
        self._settings.tol_gap_rel = 1e-6

        self._setup_done = True
        self._setup_time = time.perf_counter() - start

    def solve(
        self,
        z_k: np.ndarray,
        u_prev: float,
        u_nominal: np.ndarray,
        y_nominal: np.ndarray,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        f1_nominal: np.ndarray,
        f2_nominal: np.ndarray,
        W_bounds: np.ndarray,
        linear_jacobians: List[np.ndarray] = None,
    ) -> DirectQPSolution:
        """Solve the QP with updated parameters."""
        import clarabel

        if not self._setup_done:
            self.setup()

        start = time.perf_counter()

        q = self._builder.compute_linear_cost(y_nominal, u_prev)
        b_eq, h, A_ineq = self._builder.update_constraints(
            z_k, u_prev, u_nominal, y_nominal,
            jacobians_f1, jacobians_f2, f1_nominal, f2_nominal, W_bounds,
            linear_jacobians=linear_jacobians,
        )

        update_time = time.perf_counter() - start

        if self._builder.A_eq is not None:
            A = sparse.vstack([self._builder.A_eq, A_ineq]).tocsc()
            b_combined = np.concatenate([b_eq, h])
        else:
            A = A_ineq.tocsc()
            b_combined = h

        solve_start = time.perf_counter()
        solver = clarabel.DefaultSolver(
            self._builder.P, q, A, b_combined, self._cone_dims, self._settings
        )
        solution = solver.solve()

        total_time = time.perf_counter() - start

        is_feasible = solution.status == clarabel.SolverStatus.Solved

        if is_feasible:
            x = np.array(solution.x)
            return DirectQPSolution(
                u_optimal=np.asarray(x[self.idx_u], dtype=np.float32),
                s_max_optimal=np.asarray(x[self.idx_s_max], dtype=np.float32),
                s_min_optimal=np.asarray(x[self.idx_s_min], dtype=np.float32),
                cost=solution.obj_val,
                status=str(solution.status),
                solve_time=total_time,
                is_feasible=True,
                setup_time=getattr(self, '_setup_time', 0),
                update_time=update_time,
            )
        else:
            return DirectQPSolution(
                u_optimal=np.asarray(u_nominal, dtype=np.float32),
                s_max_optimal=np.zeros(self.N, dtype=np.float32),
                s_min_optimal=np.zeros(self.N, dtype=np.float32),
                cost=np.inf,
                status=str(solution.status),
                solve_time=total_time,
                is_feasible=False,
                update_time=update_time,
            )


def create_direct_solver(predictor, config):
    """Factory function to create DirectQPSolver from predictor and config."""
    builder = _create_builder(predictor, config)
    if builder is None:
        return None
    return DirectQPSolver(builder)


def _create_builder(predictor, config) -> "QPMatrixBuilder | None":
    """Shared factory for all direct solver backends."""
    from dcnn_tube_mpc.analysis.jacobian import extract_weights_from_convex_nn

    N_ctrl = getattr(config, 'control_horizon', config.prediction_horizon)
    N_pred = config.prediction_horizon
    if N_ctrl < N_pred:
        import warnings
        warnings.warn(
            f"Extended horizon (N_ctrl={N_ctrl} < N_pred={N_pred}) not supported "
            "by direct solver. Falling back to CVXPY solver."
        )
        return None

    N = config.prediction_horizon
    n_state = predictor.n_state

    weights_f1 = [
        extract_weights_from_convex_nn(predictor.networks[i].f1)
        for i in range(N)
    ]
    weights_f2 = [
        extract_weights_from_convex_nn(predictor.networks[i].f2)
        for i in range(N)
    ]

    return QPMatrixBuilder(
        N=N,
        n_state=n_state,
        weights_f1=weights_f1,
        weights_f2=weights_f2,
        Q=config.Q,
        R=config.R,
        R_delta=getattr(config, 'R_delta', 0.0),
        tube_weight=getattr(config, 'tube_weight', 0.0),
        beta_0=config.beta_0,
        u_min=config.u_min,
        u_max=config.u_max,
        delta_u_max=config.delta_u_max,
        decimation=getattr(config, 'decimation', 1),
        pe_gamma=getattr(config, 'pe_gamma', 0.0),
    )
