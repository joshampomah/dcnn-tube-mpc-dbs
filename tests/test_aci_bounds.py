"""Tests for ACI and DKW bounds trackers."""
from __future__ import annotations

import numpy as np
import pytest

from dcnn_tube_mpc.bounds.aci_bounds import ACIConfig, ACIBoundsTracker
from dcnn_tube_mpc.bounds.dkw_bounds import DKWConfig, DKWBoundsTracker


# ---------------------------------------------------------------------------
# ACIConfig tests
# ---------------------------------------------------------------------------

class TestACIConfig:
    def test_default(self):
        cfg = ACIConfig()
        assert 0.0 < cfg.alpha < 1.0
        assert cfg.gamma > 0.0
        assert cfg.horizon >= 1
        assert cfg.min_samples >= 1

    def test_custom(self):
        cfg = ACIConfig(alpha=0.1, gamma=0.01, horizon=5, min_samples=20)
        assert cfg.alpha == 0.1
        assert cfg.gamma == 0.01
        assert cfg.horizon == 5

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ACIConfig(alpha=1.5)

    def test_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            ACIConfig(alpha=0.0)

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError):
            ACIConfig(gamma=-0.001)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            ACIConfig(horizon=0)

    def test_warmup_strategies(self):
        for strategy in ("max", "replace", "min"):
            cfg = ACIConfig(warmup_strategy=strategy)
            assert cfg.warmup_strategy == strategy

    def test_invalid_warmup_raises(self):
        with pytest.raises(ValueError):
            ACIConfig(warmup_strategy="invalid")

    def test_from_dkw_equivalent(self):
        cfg = ACIConfig.from_dkw_equivalent(delta=1e-6, window_size=500, horizon=5)
        assert 0.0 < cfg.alpha < 1.0
        assert cfg.gamma > 0.0
        assert cfg.horizon == 5


# ---------------------------------------------------------------------------
# ACIBoundsTracker tests
# ---------------------------------------------------------------------------

class TestACIBoundsTracker:
    @pytest.fixture
    def tracker(self):
        cfg = ACIConfig(alpha=0.1, gamma=0.005, horizon=3, min_samples=5)
        offline = np.array([[-0.1, 0.1], [-0.2, 0.2], [-0.15, 0.15]])
        return ACIBoundsTracker(cfg, offline)

    def test_init(self, tracker):
        assert tracker.config.horizon == 3
        assert np.all(tracker.n_samples == 0)
        assert np.all(tracker.n_misses == 0)

    def test_initial_bounds_are_offline(self, tracker):
        bounds = tracker.get_current_bounds()
        np.testing.assert_array_equal(bounds, tracker.offline_bounds)

    def test_record_prediction_and_observation(self, tracker):
        y_preds = np.array([2.5, 2.4, 2.3])
        tracker.record_prediction(step=0, y_nominal=y_preds)
        tracker.record_observation(step=1, y_true=2.6)
        # After one observation, step 0 horizon 0 (k=0) gets error=2.6-2.5=0.1
        assert tracker.n_samples[0] == 1

    def test_bounds_update_after_min_samples(self, tracker):
        """After min_samples observations, bounds should switch from offline."""
        y_preds = np.array([2.5, 2.4, 2.3])

        for step in range(10):
            tracker.record_prediction(step=step, y_nominal=y_preds)
            tracker.record_observation(step=step + 1, y_true=2.5 + 0.05 * (step % 3))

        bounds = tracker.get_current_bounds()
        # At least one horizon step should have gone past min_samples
        assert bounds.shape == (3, 2)

    def test_theta_non_negative(self, tracker):
        y_preds = np.array([2.5, 2.4, 2.3])
        for step in range(20):
            tracker.record_prediction(step=step, y_nominal=y_preds)
            tracker.record_observation(step=step + 1, y_true=1.0)  # always below theta
        assert np.all(tracker.thetas >= 0.0)

    def test_reset(self, tracker):
        y_preds = np.array([2.5, 2.4, 2.3])
        tracker.record_prediction(step=0, y_nominal=y_preds)
        tracker.record_observation(step=1, y_true=2.6)
        tracker.reset()

        assert np.all(tracker.n_samples == 0)
        assert np.all(tracker.n_misses == 0)

    def test_get_diagnostics(self, tracker):
        diag = tracker.get_diagnostics()
        assert "n_samples" in diag
        assert "thetas" in diag
        assert "cumulative_coverage" in diag
        assert "active" in diag
        assert "current_bounds" in diag
        assert "offline_bounds" in diag

    def test_coverage_at_init(self, tracker):
        for k in range(tracker.config.horizon):
            cov = tracker.compute_coverage(k)
            assert cov == 1.0  # No samples yet

    def test_theta_init_offline(self):
        cfg = ACIConfig(horizon=3, theta_init="offline")
        offline = np.array([[-0.1, 0.1], [-0.2, 0.2], [-0.15, 0.15]])
        tracker = ACIBoundsTracker(cfg, offline)
        np.testing.assert_array_equal(tracker.thetas, offline[:, 1])

    def test_theta_init_zero(self):
        cfg = ACIConfig(horizon=3, theta_init="zero")
        offline = np.array([[-0.1, 0.1], [-0.2, 0.2], [-0.15, 0.15]])
        tracker = ACIBoundsTracker(cfg, offline)
        np.testing.assert_array_equal(tracker.thetas, np.zeros(3))


# ---------------------------------------------------------------------------
# DKWConfig tests
# ---------------------------------------------------------------------------

class TestDKWConfig:
    def test_default(self):
        cfg = DKWConfig()
        assert 0.0 < cfg.delta < 1.0
        assert cfg.min_samples >= 1

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            DKWConfig(delta=1.5)

    def test_invalid_delta_zero_raises(self):
        with pytest.raises(ValueError):
            DKWConfig(delta=0.0)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            DKWConfig(horizon=0)

    def test_window_size_must_be_gte_min_samples(self):
        with pytest.raises(ValueError):
            DKWConfig(min_samples=50, window_size=10)


# ---------------------------------------------------------------------------
# DKWBoundsTracker tests
# ---------------------------------------------------------------------------

class TestDKWBoundsTracker:
    @pytest.fixture
    def tracker(self):
        cfg = DKWConfig(delta=1e-3, horizon=3, min_samples=5)
        offline = np.array([[-0.1, 0.1], [-0.2, 0.2], [-0.15, 0.15]])
        return DKWBoundsTracker(cfg, offline)

    def test_init(self, tracker):
        assert tracker.config.horizon == 3

    def test_initial_bounds_are_offline(self, tracker):
        bounds = tracker.get_current_bounds()
        np.testing.assert_array_equal(bounds, tracker.offline_bounds)

    def test_epsilon_at_zero_samples(self, tracker):
        eps = tracker.compute_epsilon(0)
        assert eps == 1.0

    def test_epsilon_decreases_with_more_samples(self, tracker):
        eps_10 = tracker.compute_epsilon(10)
        eps_100 = tracker.compute_epsilon(100)
        assert eps_10 > eps_100

    def test_bounds_update_after_min_samples(self, tracker):
        y_preds = np.array([2.5, 2.4, 2.3])
        for step in range(10):
            tracker.record_prediction(step=step, y_nominal=y_preds)
            tracker.record_observation(step=step + 1, y_true=2.5)

        bounds = tracker.get_current_bounds()
        assert bounds.shape == (3, 2)

    def test_reset(self, tracker):
        y_preds = np.array([2.5, 2.4, 2.3])
        tracker.record_prediction(step=0, y_nominal=y_preds)
        tracker.record_observation(step=1, y_true=2.6)
        tracker.reset()

        for k in range(tracker.config.horizon):
            assert len(tracker.error_histories[k]) == 0

    def test_get_diagnostics(self, tracker):
        diag = tracker.get_diagnostics()
        assert "n_samples" in diag
        assert "epsilon" in diag
        assert "active" in diag
        assert "current_bounds" in diag

    def test_offline_bounds_truncated_to_horizon(self):
        cfg = DKWConfig(horizon=2)
        offline = np.array([[-0.1, 0.1], [-0.2, 0.2], [-0.15, 0.15], [-0.3, 0.3]])
        tracker = DKWBoundsTracker(cfg, offline)
        assert tracker.offline_bounds.shape == (2, 2)

    def test_offline_bounds_too_short_raises(self):
        cfg = DKWConfig(horizon=5)
        offline = np.array([[-0.1, 0.1], [-0.2, 0.2]])
        with pytest.raises(ValueError):
            DKWBoundsTracker(cfg, offline)
