"""Tests for SCP core: SCPConfig and SCP algorithm with a mock predictor."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_convex_nn(n_input: int, n_hidden: int = 8) -> nn.Module:
    """Build a minimal ConvexNN-compatible mock module."""
    from dcnn_tube_mpc.models.dcnn_models import ConvexNN
    return ConvexNN(n_input=n_input, n_hidden=n_hidden, n_layers=1)


def _make_dcnn_model(n_state: int, n_input: int, n_hidden: int = 8) -> nn.Module:
    """Build a minimal DCNNModel."""
    from dcnn_tube_mpc.models.dcnn_models import DCNNModel
    return DCNNModel(n_input=n_state + n_input, n_hidden=n_hidden, n_layers=1)


def _make_multi_step_dcnn(n_state: int = 30, horizon: int = 3) -> nn.Module:
    """Build a minimal MultiStepDCNN."""
    from dcnn_tube_mpc.models.dcnn_models import MultiStepDCNN
    return MultiStepDCNN(
        n_state=n_state,
        n_input=1,
        n_hidden=8,
        n_layers=1,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# SCPConfig tests
# ---------------------------------------------------------------------------

class TestSCPConfig:
    def test_default_instantiation(self):
        from dcnn_tube_mpc.controllers.scp_config import SCPConfig
        cfg = SCPConfig()
        assert cfg.prediction_horizon >= 1
        assert cfg.Q > 0
        assert cfg.R > 0
        assert cfg.maxiters >= 1

    def test_custom_values(self):
        from dcnn_tube_mpc.controllers.scp_config import SCPConfig
        # Use u_max=0.03, R=2.0 -> R*u_max^2=0.0018 < MIN_R_COST=0.01 triggers scale
        # Use a large enough R so no scaling: R=50 -> R*0.03^2=0.045 > 0.01
        cfg = SCPConfig(prediction_horizon=5, Q=10000.0, R=50.0, maxiters=20)
        assert cfg.prediction_horizon == 5
        assert cfg.Q == 10000.0
        assert cfg.R == 50.0
        assert cfg.maxiters == 20

    def test_invalid_horizon_raises(self):
        from dcnn_tube_mpc.controllers.scp_config import SCPConfig
        with pytest.raises((ValueError, Exception)):
            SCPConfig(prediction_horizon=0)

    def test_invalid_Q_raises(self):
        from dcnn_tube_mpc.controllers.scp_config import SCPConfig
        with pytest.raises((ValueError, Exception)):
            SCPConfig(Q=-1.0)

    def test_solver_field_present(self):
        from dcnn_tube_mpc.controllers.scp_config import SCPConfig
        cfg = SCPConfig()
        assert hasattr(cfg, "solver")


# ---------------------------------------------------------------------------
# SCP algorithm tests (with mock predictor)
# ---------------------------------------------------------------------------

class TestSCPAlgorithm:
    @pytest.fixture
    def setup(self):
        """Create a small DCNN and config for SCP tests."""
        from dcnn_tube_mpc.controllers.scp_config import SCPConfig

        n_state = 30
        horizon = 3
        predictor = _make_multi_step_dcnn(n_state=n_state, horizon=horizon)
        predictor.eval()

        cfg = SCPConfig(
            prediction_horizon=horizon,
            control_horizon=horizon,
            n_state_y=15,
            n_state_u=15,
            Q=50000.0,
            R=1.0,
            maxiters=3,
            solver="CLARABEL",
            u_min=0.0,
            u_max=0.030,
        )

        rng = np.random.default_rng(0)
        z_k = (rng.standard_normal(n_state) * 0.05 + 2.3).astype(np.float32)
        u0 = np.full(horizon, 0.01, dtype=np.float32)
        u_prev = 0.01

        return predictor, cfg, z_k, u0, u_prev

    def test_solve_scp_returns_result(self, setup):
        """SCP solver should return an SCPResult without error."""
        from dcnn_tube_mpc.controllers.scp_algorithm import solve_scp

        predictor, cfg, z_k, u0, u_prev = setup

        result = solve_scp(
            z_k=z_k,
            u_prev=u_prev,
            u_initial=u0,
            predictor=predictor,
            config=cfg,
        )

        assert result is not None
        assert hasattr(result, "u_optimal")
        assert len(result.u_optimal) == cfg.control_horizon

    def test_u_optimal_within_bounds(self, setup):
        """Optimal control should respect [u_min, u_max] bounds."""
        from dcnn_tube_mpc.controllers.scp_algorithm import solve_scp

        predictor, cfg, z_k, u0, u_prev = setup

        result = solve_scp(
            z_k=z_k,
            u_prev=u_prev,
            u_initial=u0,
            predictor=predictor,
            config=cfg,
        )

        u_opt = result.u_optimal
        assert np.all(u_opt >= cfg.u_min - 1e-6), f"u below u_min: {u_opt}"
        assert np.all(u_opt <= cfg.u_max + 1e-6), f"u above u_max: {u_opt}"

    def test_iterations_converge(self, setup):
        """SCP should complete (converge or reach maxiters) without error."""
        from dcnn_tube_mpc.controllers.scp_algorithm import solve_scp

        predictor, cfg, z_k, u0, u_prev = setup
        result = solve_scp(
            z_k=z_k,
            u_prev=u_prev,
            u_initial=u0,
            predictor=predictor,
            config=cfg,
        )

        assert hasattr(result, "n_iterations")
        assert result.n_iterations >= 1
        assert result.n_iterations <= cfg.maxiters + 1

    def test_warm_start(self, setup):
        """Calling solve_scp twice should not error (warm start path)."""
        from dcnn_tube_mpc.controllers.scp_algorithm import solve_scp

        predictor, cfg, z_k, u0, u_prev = setup
        result1 = solve_scp(z_k=z_k, u_prev=u_prev, u_initial=u0, predictor=predictor, config=cfg)
        result2 = solve_scp(z_k=z_k, u_prev=u_prev, u_initial=result1.u_optimal, predictor=predictor, config=cfg)
        assert result2 is not None


# ---------------------------------------------------------------------------
# Disturbance bounds tests
# ---------------------------------------------------------------------------

class TestDisturbanceBounds:
    def test_compute_disturbance_bounds_shape(self):
        from dcnn_tube_mpc.bounds.disturbance_bounds import compute_disturbance_bounds

        n_state = 30
        horizon = 3
        n_samples = 200

        predictor = _make_multi_step_dcnn(n_state=n_state, horizon=horizon)
        predictor.eval()

        rng = np.random.default_rng(1)
        X_val = rng.standard_normal((n_samples, n_state)).astype(np.float32)
        U_val = rng.uniform(0, 0.03, (n_samples, horizon)).astype(np.float32)
        Y_val = rng.standard_normal((n_samples, horizon)).astype(np.float32)

        W = compute_disturbance_bounds(predictor, X_val, U_val, Y_val, percentile=80.0)

        assert W.shape == (horizon, 2)
        assert np.all(W[:, 0] <= 0)
        assert np.all(W[:, 1] >= 0)
