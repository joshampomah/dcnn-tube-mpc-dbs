"""Tests for custom training-data loading."""
import json

import numpy as np


def _write_patient_folder(root, name, beta, stim):
    patient_dir = root / name
    patient_dir.mkdir(parents=True)
    np.savetxt(patient_dir / "beta_causal_RMS.csv", beta, delimiter=",")
    np.savetxt(patient_dir / "stimulation.csv", stim, delimiter=",")
    return patient_dir


def test_csv_loader_downsamples_stimulation_and_windows(tmp_path):
    from dcnn_tube_mpc.training.train_predictor import _load_or_generate_data

    beta = np.linspace(1.0, 2.0, 80, dtype=np.float32)
    stim_50hz = np.linspace(0.0, 0.03, 80, dtype=np.float32)
    patient_dir = _write_patient_folder(
        tmp_path,
        "patient_direct",
        beta,
        np.repeat(stim_50hz, 4),
    )

    x, u, y = _load_or_generate_data(
        patient_dir,
        n_state_y=5,
        n_state_u=5,
        horizon=3,
        input_space="linear",
    )

    assert x.shape == (72, 10)
    assert u.shape == (72, 3)
    assert y.shape == (72, 3)
    assert np.all(np.isfinite(x[:, :5]))
    assert np.isclose(u[0, 0], stim_50hz[5])


def test_selected_patients_role_filter(tmp_path):
    from dcnn_tube_mpc.training.train_predictor import _load_or_generate_data

    beta = np.linspace(1.0, 2.0, 50, dtype=np.float32)
    stim = np.zeros(50, dtype=np.float32)
    _write_patient_folder(tmp_path, "patient_training", beta, stim)
    _write_patient_folder(tmp_path, "patient_refinement", beta, stim)

    (tmp_path / "selected_patients.json").write_text(
        json.dumps(
            {
                "patients": [
                    {"directory": "patient_training", "role": "training"},
                    {"directory": "patient_refinement", "role": "refinement"},
                ]
            }
        )
    )

    x, _, _ = _load_or_generate_data(
        tmp_path,
        n_state_y=5,
        n_state_u=5,
        horizon=3,
        patient_role="training",
    )

    assert x.shape[0] == 42
