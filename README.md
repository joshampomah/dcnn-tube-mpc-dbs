# dcnn-tube-mpc-dbs

DC Neural Network Tube MPC for Closed-Loop Deep Brain Stimulation.

Research code accompanying the CDC25 paper "Deep Learning Model Predictive Control for Deep Brain Stimulation in Parkinson's Disease".

## Overview

This package implements a robust tube MPC controller based on Difference-of-Convex Neural Networks (DC-NN). Key components:

- **DC-NN predictor**: Multi-step predictor using DC decomposition f = f1 - f2
- **SCP algorithm**: Sequential Convex Programming for solving the tube MPC
- **Disturbance bounds**: DKW and ACI methods for online bound adaptation
- **Synthetic data**: Generate benchmark datasets without patient recordings

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from dcnn_tube_mpc.synthetic.data_generator import generate_synthetic_beta, generate_synthetic_stimulation

beta = generate_synthetic_beta(n_steps=5000, seed=42)
stim = generate_synthetic_stimulation(n_steps=5000, seed=42)
```

See `examples/quick_demo.py` for a complete demo.

## Disclaimer

See [DISCLAIMER.md](DISCLAIMER.md). This is a research prototype — not a medical device.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT License. See [LICENSE](LICENSE).
