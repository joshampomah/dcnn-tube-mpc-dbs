#!/usr/bin/env python
"""Run DC-NN Tube MPC on synthetic data.

Example:
    python scripts/run_dcnn_mpc.py --duration 60 --model-dir models/dcnn
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run DC-NN Tube MPC simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to saved predictor model directory")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n-state-y", type=int, default=15)
    parser.add_argument("--n-state-u", type=int, default=15)
    parser.add_argument("--beta-0", type=float, default=2.3,
                        help="Pathological threshold")
    parser.add_argument("--Q", type=float, default=50000.0)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--solver", default="clarabel",
                        choices=["clarabel", "osqp", "piqp", "cvxpy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    n_state = args.n_state_y + args.n_state_u

    # Generate synthetic patient data
    logger.info("Generating synthetic patient data (seed=%d)", args.seed)
    from dcnn_tube_mpc.synthetic.data_generator import generate_demo_patient
    patient = generate_demo_patient(
        n_state_y=args.n_state_y,
        n_state_u=args.n_state_u,
        seed=args.seed,
    )

    # Build or load predictor
    if args.model_dir is not None and args.model_dir.exists():
        logger.info("Loading model from %s", args.model_dir)
        from dcnn_tube_mpc.training.train_predictor import MultiStepPredictor
        predictor_wrapper = MultiStepPredictor(
            n_state=n_state,
            horizon=args.horizon,
            n_state_y=args.n_state_y,
        )
        predictor_wrapper.load_models(args.model_dir)
        predictor = predictor_wrapper.model
    else:
        logger.warning("No model dir provided or not found — using untrained random weights")
        from dcnn_tube_mpc.models.dcnn_models import MultiStepDCNN
        predictor = MultiStepDCNN(
            n_state=n_state,
            n_input=1,
            n_hidden=32,
            n_layers=1,
            horizon=args.horizon,
        )

    predictor.eval()

    # Compute disturbance bounds from a small synthetic validation set
    logger.info("Computing disturbance bounds from synthetic validation data")
    from dcnn_tube_mpc.synthetic.data_generator import generate_modulated_beta
    import torch

    n_val = 500
    beta_val, stim_val = generate_modulated_beta(n_val + n_state + args.horizon + 5, seed=args.seed + 1)
    X_val, U_val, Y_val = [], [], []
    for t in range(n_state, len(beta_val) - args.horizon):
        X_val.append(np.hstack([beta_val[t - args.n_state_y:t], stim_val[t - args.n_state_u:t]]))
        U_val.append(stim_val[t:t + args.horizon])
        Y_val.append(beta_val[t:t + args.horizon])
        if len(X_val) >= n_val:
            break
    X_val = np.array(X_val, dtype=np.float32)
    U_val = np.array(U_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)

    from dcnn_tube_mpc.bounds.disturbance_bounds import compute_disturbance_bounds
    W_bounds = compute_disturbance_bounds(predictor, X_val, U_val, Y_val)
    logger.info("Disturbance bounds: %s", W_bounds)

    # Build SCP config
    from dcnn_tube_mpc.controllers.scp_config import SCPConfig
    cfg = SCPConfig(
        horizon=args.horizon,
        n_state=n_state,
        Q=args.Q,
        R=args.R,
        max_iter=args.max_iter,
        solver=args.solver,
        u_min=0.0,
        u_max=0.030,
        beta_0=args.beta_0,
    )

    # Build controller
    from dcnn_tube_mpc.controllers.scp_controller import SCPController
    controller = SCPController(predictor=predictor, config=cfg, W_bounds=W_bounds)

    # Run simulation
    logger.info("Running simulation: duration=%.1fs", args.duration)
    from dcnn_tube_mpc.simulation.simulate import simulate_trial
    result = simulate_trial(
        controller=controller,
        patient=patient,
        duration=args.duration,
        beta_0=args.beta_0,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Report metrics
    print("\n=== Simulation Results ===")
    print(f"  Controller:      {result.controller_type}")
    print(f"  Duration:        {result.duration:.1f} s ({result.n_steps} steps)")
    print(f"  Mean y:          {result.metrics['mean_y']:.4f}")
    print(f"  Mean excess:     {result.metrics['mean_excess']:.4f}")
    print(f"  Time above beta: {result.metrics['time_above']*100:.1f}%")
    print(f"  Mean u:          {result.metrics['mean_u']:.4f}")
    print("=" * 26)


if __name__ == "__main__":
    main()
