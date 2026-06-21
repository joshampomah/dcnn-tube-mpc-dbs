"""Integration tests for the local simulation harness."""
from __future__ import annotations

import numpy as np

from dcnn_tube_mpc.simulation.simulate import simulate_trial
from dcnn_tube_mpc.synthetic.data_generator import generate_demo_patient


def test_simulate_trial_passes_histories_to_controller():
    """History-aware controllers receive newest-first histories."""

    class HistoryController:
        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def compute_control(self, y_history, u_history, u_prev):
            self.calls += 1
            assert y_history.shape == (15,)
            assert u_history.shape == (15,)
            assert np.isclose(u_history[0], u_prev)
            return 0.0, {"call": self.calls}

    patient = generate_demo_patient(n_state_y=15, n_state_u=15, seed=1)
    ctrl = HistoryController()
    result = simulate_trial(ctrl, patient, duration=0.1)
    assert result.n_steps == 5
    assert ctrl.calls == 5
