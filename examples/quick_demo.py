#!/usr/bin/env python
"""Quick demonstration of DC-NN Tube MPC on synthetic DBS data.

This example:
1. Generates synthetic beta/stimulation data (no patient data required)
2. Trains a small DC-NN predictor on synthetic data
3. Computes disturbance bounds from validation data
4. Runs a closed-loop simulation with the SCP controller
5. Reports performance metrics

Run with:
    python examples/quick_demo.py
"""
from __future__ import annotations

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== DC-NN Tube MPC Quick Demo ===\n")

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    logger.info("Step 1: Generating synthetic training data...")

    from dcnn_tube_mpc.synthetic.data_generator import (
        generate_modulated_beta,
        generate_demo_patient,
    )

    n_state_y = 15
    n_state_u = 15
    n_state = n_state_y + n_state_u
    horizon = 3  # small for speed
    n_train = 3000
    n_val = 500

    beta, stim = generate_modulated_beta(n_train + n_state + horizon + n_val + 10, seed=42)

    X_list, U_list, Y_list = [], [], []
    for t in range(n_state, len(beta) - horizon):
        x_t = np.hstack([beta[t - n_state_y:t], stim[t - n_state_u:t]])
        X_list.append(x_t)
        U_list.append(stim[t:t + horizon])
        Y_list.append(beta[t:t + horizon])
        if len(X_list) >= n_train + n_val:
            break

    X = np.array(X_list, dtype=np.float32)
    U = np.array(U_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    X_train, U_train, Y_train = X[:n_train], U[:n_train], Y[:n_train]
    X_val, U_val, Y_val = X[n_train:], U[n_train:], Y[n_train:]
    logger.info(f"  Training samples: {n_train}, Validation samples: {n_val}")

    # ------------------------------------------------------------------
    # 2. Train DC-NN predictor
    # ------------------------------------------------------------------
    logger.info("\nStep 2: Training DC-NN predictor (small, fast)...")

    from dcnn_tube_mpc.training.train_predictor import MultiStepPredictor

    predictor_wrapper = MultiStepPredictor(
        n_state=n_state,
        n_hidden=16,
        n_layers=1,
        horizon=horizon,
        n_state_y=n_state_y,
    )

    train_summary = predictor_wrapper.train(
        x=X_train,
        u=U_train,
        y=Y_train,
        epochs=20,
        batch_size=256,
        lr=0.001,
        val_split=0.2,
        num_workers=0,
        pin_memory=False,
        verbose=1,
        early_stopping=True,
        early_stopping_patience=5,
    )
    logger.info(f"  Training complete in {train_summary['elapsed_time']:.1f}s")

    predictor = predictor_wrapper.model
    predictor.eval()

    # ------------------------------------------------------------------
    # 3. Compute disturbance bounds
    # ------------------------------------------------------------------
    logger.info("\nStep 3: Computing disturbance bounds...")

    from dcnn_tube_mpc.bounds.disturbance_bounds import compute_disturbance_bounds

    W_bounds = compute_disturbance_bounds(
        predictor=predictor,
        X_val=X_val,
        U_val=U_val,
        Y_val=Y_val,
        percentile=80.0,
    )
    logger.info(f"  W_bounds (horizon={horizon}):")
    for k, (w_min, w_max) in enumerate(W_bounds):
        logger.info(f"    Step {k+1}: [{w_min:.4f}, {w_max:.4f}]")

    # ------------------------------------------------------------------
    # 4. Build controller and simulate
    # ------------------------------------------------------------------
    logger.info("\nStep 4: Setting up SCP controller...")

    from dcnn_tube_mpc.controllers.scp_config import SCPConfig
    from dcnn_tube_mpc.controllers.scp_controller import SCPController

    cfg = SCPConfig(
        horizon=horizon,
        n_state=n_state,
        Q=50000.0,
        R=1.0,
        max_iter=3,
        solver="cvxpy",
        u_min=0.0,
        u_max=0.030,
        beta_0=2.3,
    )

    controller = SCPController(predictor=predictor, config=cfg, W_bounds=W_bounds)

    # ------------------------------------------------------------------
    # 5. Simulate
    # ------------------------------------------------------------------
    logger.info("\nStep 5: Running closed-loop simulation (30s)...")

    patient = generate_demo_patient(n_state_y=n_state_y, n_state_u=n_state_u, seed=42)

    from dcnn_tube_mpc.simulation.simulate import simulate_trial

    result = simulate_trial(
        controller=controller,
        patient=patient,
        duration=30.0,
        beta_0=2.3,
        seed=42,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # 6. Baseline comparison (bang-bang)
    # ------------------------------------------------------------------
    result_bb = simulate_trial(
        controller="bang-bang",
        patient=patient,
        duration=30.0,
        beta_0=2.3,
        seed=42,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Simulation Results (30s)")
    print("=" * 50)
    print(f"{'Metric':<25} {'DC-NN MPC':>12} {'Bang-Bang':>12}")
    print("-" * 50)
    for key in ["mean_y", "mean_excess", "time_above", "mean_u"]:
        v_mpc = result.metrics[key]
        v_bb = result_bb.metrics[key]
        if key == "time_above":
            print(f"  {key:<23} {v_mpc*100:>11.1f}% {v_bb*100:>11.1f}%")
        else:
            print(f"  {key:<23} {v_mpc:>12.4f} {v_bb:>12.4f}")
    print("=" * 50)
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
