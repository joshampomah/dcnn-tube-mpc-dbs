"""Ensemble DCNN predictor with conformal prediction tubes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .dcnn_models import MultiStepDCNN


class EnsembleDCNN(nn.Module):
    """Ensemble of M MultiStepDCNN models."""

    def __init__(
        self,
        members: List[MultiStepDCNN],
        conformal_quantiles: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.n_members = len(members)

        m0 = members[0]
        self.n_state = m0.n_state
        self.n_input = m0.n_input
        self.n_hidden = m0.n_hidden
        self.n_layers = m0.n_layers
        self.horizon = m0.horizon

        if conformal_quantiles is not None:
            self.register_buffer(
                "conformal_quantiles",
                torch.tensor(conformal_quantiles, dtype=torch.float32),
            )
        else:
            self.conformal_quantiles = None

    def forward_k(self, x: torch.Tensor, u: torch.Tensor, k: int) -> torch.Tensor:
        preds = torch.stack([m.forward_k(x, u, k) for m in self.members])
        return preds.mean(dim=0)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        all_preds = [m(x, u) for m in self.members]
        means = []
        for k in range(self.horizon):
            stacked = torch.stack([p[k] for p in all_preds])
            means.append(stacked.mean(dim=0))
        return tuple(means)

    def forward_with_spread(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        all_preds = [m(x, u) for m in self.members]
        means, stds = [], []
        for k in range(self.horizon):
            stacked = torch.stack([p[k] for p in all_preds])
            means.append(stacked.mean(dim=0))
            stds.append(stacked.std(dim=0))
        return tuple(means), tuple(stds)

    def get_conformal_bounds(self) -> Optional[np.ndarray]:
        if self.conformal_quantiles is None:
            return None
        q = self.conformal_quantiles.numpy()
        return np.column_stack([-q, q])

    def enforce_constraints(self):
        for m in self.members:
            m.enforce_constraints()


def load_ensemble(
    ensemble_dir: str | Path,
    device: str = "cpu",
) -> EnsembleDCNN:
    """Load a trained ensemble from disk."""
    ensemble_dir = Path(ensemble_dir)

    with open(ensemble_dir / "ensemble_config.json") as f:
        meta = json.load(f)

    n_members = meta["n_members"]
    members = []

    for i in range(n_members):
        member_dir = Path(meta["members"][i])
        with open(member_dir / "config.json") as f:
            cfg = json.load(f)

        model = MultiStepDCNN(
            n_state=cfg.get("n_state", 30),
            n_input=cfg.get("n_input", 1),
            horizon=cfg.get("horizon", 5),
            n_hidden=cfg.get("n_hidden", 32),
            n_layers=cfg.get("n_layers", 1),
            use_spectral_norm=cfg.get("use_spectral_norm", False),
        )

        for k in range(model.horizon):
            net_file = member_dir / f"network_k{k+1}.pt"
            state_dict = torch.load(str(net_file), map_location=device, weights_only=True)
            model.networks[k].load_state_dict(state_dict)
        model.eval()
        members.append(model)

    conformal_quantiles = None
    cal_path = ensemble_dir / "conformal_calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        conformal_quantiles = np.array(cal["conformal_quantiles"])

    ensemble = EnsembleDCNN(members, conformal_quantiles)
    ensemble.eval()

    first_cfg_path = Path(meta["members"][0]) / "config.json"
    with open(first_cfg_path) as f:
        first_cfg = json.load(f)
    ensemble.n_state_y = first_cfg.get("n_state_y", 15)
    ensemble.n_state_u = first_cfg.get("n_state_u", 15)

    return ensemble
