# dcnn-tube-mpc-dbs

DC neural network tube MPC for closed-loop deep brain stimulation (DBS).

This repository contains the neural-network MPC method code from the project:
multi-step DCNN predictors, sequential convex programming (SCP), QP solver
backends, and uncertainty/tube-bound utilities. It is intended for researchers
who want to inspect or extend the DCNN controller family, not just run the
baseline benchmark.

This code is a research prototype. It is not a medical device and must not be
used for clinical decision-making or patient treatment. See
[DISCLAIMER.md](DISCLAIMER.md).

## Repository Set

The project is split by responsibility:

| Repository | Purpose |
|---|---|
| [closed-loop-dbs-bench](https://github.com/joshampomah/closed-loop-dbs-bench) | Shared benchmark, synthetic DBS plant, metrics, plotting utilities, bang-bang/PI/linear baselines |
| [dcnn-tube-mpc-dbs](https://github.com/joshampomah/dcnn-tube-mpc-dbs) | DC neural network tube MPC method: predictor, SCP controller, uncertainty bounds, synthetic training/demo code |
| [koopman-mpc-dbs](https://github.com/joshampomah/koopman-mpc-dbs) | Koopman MPC method: lifted-linear predictor, dense QP builder, OLS training/demo code |
| [embedded-stable-neuron-mpc](https://github.com/joshampomah/embedded-stable-neuron-mpc) | C++/STM32 implementation of the stable-neuron and Koopman QP solvers, plus the final report PDF |

Use `closed-loop-dbs-bench` for a common comparison harness. Use this repo for
the DCNN-specific model, training, and controller implementation.

## What Is In This Repo

- `src/dcnn_tube_mpc/models/`: DCNN/ICNN model definitions, spectral
  normalization helpers, ensemble predictor support, and an ARX utility model.
- `src/dcnn_tube_mpc/training/`: synthetic/public-safe training pipeline for
  multi-step predictors.
- `src/dcnn_tube_mpc/controllers/`: SCP controller, SCP configuration, and SCP
  algorithm implementation.
- `src/dcnn_tube_mpc/solvers/`: QP backends used inside SCP, including direct
  CLARABEL-style assembly, OSQP, PIQP, and CVXPY-backed paths.
- `src/dcnn_tube_mpc/bounds/`: disturbance bounds, DKW bounds, ACI bounds, and
  perturbation/Jacobian-based utilities.
- `src/dcnn_tube_mpc/synthetic/`: synthetic beta/stimulation generation and
  modulation helpers.
- `src/dcnn_tube_mpc/simulation/`: a small method-local simulation harness for
  quick demos.
- `scripts/run_dcnn_mpc.py`: command-line synthetic run for a DCNN controller.
- `examples/quick_demo.py`: end-to-end demo that trains a small synthetic
  predictor, computes bounds, and runs closed-loop simulation.
- `tests/`: lightweight public-safe tests for the model/controller utilities.

## What Is Not In This Repo

- No patient recordings.
- No patient-trained model checkpoints.
- No private experiment archive or report-writing material.
- No STM32 firmware. Embedded deployment lives in `embedded-stable-neuron-mpc`.

The included demos use synthetic data. If no model directory is supplied,
`scripts/run_dcnn_mpc.py` falls back to random untrained weights, which is useful
only for exercising the code path.

## Installation

Requires Python 3.10-3.12.

```bash
pip install -e ".[dev]"
```

Optional solver backends may require their own platform-specific wheels. The CI
workflow installs the default development dependencies and runs the tests on
Python 3.10, 3.11, and 3.12.

## Quick Start

Run the end-to-end synthetic demo:

```bash
python examples/quick_demo.py
```

Run the command-line synthetic controller path:

```bash
python scripts/run_dcnn_mpc.py --duration 30 --solver direct
```

Use a saved predictor directory if you have one:

```bash
python scripts/run_dcnn_mpc.py --model-dir models/dcnn --duration 60
```

## Using Your Own Data

For the 4YP DCNN experiments, the larger private data files were raw MATLAB
recordings from the Cambium/MRC BNDU dataset
[STN local field potential recordings from awake patients with Parkinson's, ON
and OFF meds, and during 130 Hz DBS](https://data.mrc.ox.ac.uk/stn-lfp-on-off-and-dbs).
Registered/logged-in users can download or request access to the raw data from
that page. The raw `.mat` files were first processed into one folder per
recording:

```text
private_data/processed/aperiodic/
├── patient_001/
│   ├── beta_causal_RMS.csv
│   ├── stimulation.csv
│   └── metadata.json
├── patient_002/
│   └── ...
└── selected_patients.json
```

The raw `.mat` input used by the 4YP processing scripts had a `SmrData` struct
with `Fs`, `WvData`, and `WvTits`. Processing selected the STN LFP channel,
extracted a causal 13-30 Hz beta RMS envelope, resampled it to 50 Hz, and wrote
`beta_causal_RMS.csv`. For resting-state recordings, `stimulation.csv` is the
same length and contains zeros.

Train directly from that processed root:

```bash
python -m dcnn_tube_mpc.training.train_predictor \
  --data-dir ../private_data/processed/aperiodic \
  --input-space linear \
  --patient-role training \
  --synthetic-stim \
  --horizon 5 \
  --save-dir models/dcnn_custom
```

`--synthetic-stim` overlays PRBS stimulation on autonomous/resting-state beta,
matching the 4YP DCNN training setup. Omit it if `stimulation.csv` already
contains applied stimulation. `--input-space linear` is correct for the
processed `beta_causal_RMS.csv`; use `--input-space log` only if your CSV has
already been log-transformed.

You can also point `--data-dir` at a single folder containing
`beta_causal_RMS.csv` and `stimulation.csv`, such as the original
`nndbs/original_js` stimulation pair. If the stimulation trace is sampled at an
obvious integer multiple of the beta trace, the loader downsamples it before
windowing.

Cached `.npz` files with `x`, `u`, and `y` arrays are still supported as an
optional private preprocessing format, but they are not required.

At runtime, `SCPController.compute_control(...)` expects `y_history` and
`u_history` newest-first. The training windows use oldest-to-newest histories
because they represent fixed supervised regressors. The benchmark repo has a
longer [DATA.md](https://github.com/joshampomah/closed-loop-dbs-bench/blob/master/DATA.md)
with a raw-trace-to-window example.

## Main Programming Interface

The high-level controller is `SCPController`:

```python
from dcnn_tube_mpc.controllers.scp_config import SCPConfig
from dcnn_tube_mpc.controllers.scp_controller import SCPController

cfg = SCPConfig(
    prediction_horizon=5,
    control_horizon=5,
    n_state_y=15,
    n_state_u=15,
    qp_solver_type="direct",
    beta_0=2.3,
)

ctrl = SCPController(predictor=predictor, config=cfg, W_bounds=W_bounds)
u, info = ctrl.compute_control(y_history, u_history, u_prev)
```

`y_history` and `u_history` are newest-first arrays. `compute_control` returns
the first control action and an `SCPResult` with solver diagnostics.

## Using With The Benchmark Repo

Install the benchmark repo alongside this repo:

```bash
pip install -e ../closed-loop-dbs-bench
pip install -e ".[dev]"
```

Then pass an `SCPController` directly into the benchmark runner:

```python
from dbs_bench.simulation.simulate import SimulationRunner
from dbs_bench.synthetic.data_generator import generate_demo_patient

patient = generate_demo_patient(n_state_y=15)
runner = SimulationRunner(patient, dt=0.02, beta_0=2.3)

result = runner.run(ctrl, duration=60.0, controller_type="dcnn-mpc")
print(result.metrics)
```

## Method Summary

The predictor is a multi-step difference-of-convex neural network:

```text
f(z, u) = f1(z, u) - f2(z, u)
```

where the convex sub-networks allow each MPC step to be approximated by a
sequence of convex QP subproblems. The SCP controller repeatedly linearizes the
concave part, solves the QP, and applies the first control input. Disturbance
and tube-bound modules provide public-safe versions of the robustness machinery
used in the project.

## Tests

```bash
pytest tests/ -v
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
