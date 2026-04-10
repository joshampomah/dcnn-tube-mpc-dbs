"""Spectral normalization utilities for contractive DC-NN."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

if TYPE_CHECKING:
    from .dcnn_models import ConvexNN, DCNNModel, MultiStepDCNN


class ScaledSpectralNorm(nn.Module):
    """Spectral normalization with scaling to target norm."""

    def __init__(
        self,
        module: nn.Linear,
        target: float = 0.5,
        n_power_iterations: int = 1,
    ):
        super().__init__()
        self.target = target
        self.module = spectral_norm(module, n_power_iterations=n_power_iterations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x) * self.target

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.module.bias


def apply_spectral_norm_to_linear(
    layer: nn.Linear,
    target: float = 0.5,
    n_power_iterations: int = 1,
) -> ScaledSpectralNorm:
    return ScaledSpectralNorm(layer, target=target, n_power_iterations=n_power_iterations)


def apply_spectral_norm_to_convex_nn(
    model: "ConvexNN",
    target: float = 0.5,
    n_power_iterations: int = 1,
    layers: str = "all",
) -> "ConvexNN":
    """Apply spectral normalization to layers in a ConvexNN."""
    apply_input = layers == "all"
    apply_hidden = layers in ("all", "hidden_only", "internal")
    apply_skip = layers in ("all", "internal")
    apply_output = layers == "all"

    if apply_input:
        model.input_layer = apply_spectral_norm_to_linear(
            model.input_layer, target=target, n_power_iterations=n_power_iterations
        )

    if apply_skip:
        for i in range(len(model.skip_layers)):
            model.skip_layers[i] = apply_spectral_norm_to_linear(
                model.skip_layers[i], target=target, n_power_iterations=n_power_iterations
            )

    if apply_hidden:
        for i in range(len(model.hidden_layers)):
            model.hidden_layers[i] = apply_spectral_norm_to_linear(
                model.hidden_layers[i], target=target, n_power_iterations=n_power_iterations
            )

    if apply_output:
        model.output_layer = apply_spectral_norm_to_linear(
            model.output_layer, target=target, n_power_iterations=n_power_iterations
        )

    model._has_spectral_norm = True
    model._spectral_norm_target = target
    model._spectral_norm_layers = layers

    return model


def remove_spectral_norm_from_convex_nn(model: "ConvexNN") -> "ConvexNN":
    """Remove spectral normalization from a ConvexNN for clean saving."""
    if not getattr(model, '_has_spectral_norm', False):
        return model

    def unwrap_layer(wrapped: ScaledSpectralNorm) -> nn.Linear:
        inner = wrapped.module
        if hasattr(inner, 'weight_orig'):
            new_layer = nn.Linear(inner.in_features, inner.out_features, bias=inner.bias is not None)
            with torch.no_grad():
                new_layer.weight.copy_(inner.weight * wrapped.target)
                if inner.bias is not None:
                    new_layer.bias.copy_(inner.bias * wrapped.target)
            return new_layer
        return inner

    if isinstance(model.input_layer, ScaledSpectralNorm):
        model.input_layer = unwrap_layer(model.input_layer)

    for i in range(len(model.skip_layers)):
        if isinstance(model.skip_layers[i], ScaledSpectralNorm):
            model.skip_layers[i] = unwrap_layer(model.skip_layers[i])

    for i in range(len(model.hidden_layers)):
        if isinstance(model.hidden_layers[i], ScaledSpectralNorm):
            model.hidden_layers[i] = unwrap_layer(model.hidden_layers[i])

    if isinstance(model.output_layer, ScaledSpectralNorm):
        model.output_layer = unwrap_layer(model.output_layer)

    model._has_spectral_norm = False

    return model


def estimate_lipschitz_constant(
    model: nn.Module,
    input_dim: int,
    n_samples: int = 1000,
    epsilon: float = 1e-3,
    device: Optional[torch.device] = None,
) -> float:
    """Estimate Lipschitz constant using finite differences."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    max_ratio = 0.0

    with torch.no_grad():
        for _ in range(n_samples):
            x = torch.randn(1, input_dim, device=device)
            direction = torch.randn(1, input_dim, device=device)
            direction = direction / direction.norm() * epsilon

            f_x = model(x)
            f_x_plus = model(x + direction)

            output_diff = (f_x_plus - f_x).norm().item()
            input_diff = direction.norm().item()

            ratio = output_diff / input_diff
            max_ratio = max(max_ratio, ratio)

    return max_ratio


def get_layer_spectral_norms(model: "ConvexNN") -> Dict[str, float]:
    """Get spectral norms of all layers in a ConvexNN."""
    results = {}

    def get_spectral_norm(layer, name: str):
        if isinstance(layer, ScaledSpectralNorm):
            results[name] = layer.target
        elif isinstance(layer, nn.Linear):
            with torch.no_grad():
                u = torch.randn(layer.weight.shape[0], device=layer.weight.device)
                for _ in range(10):
                    v = layer.weight.T @ u
                    v = v / v.norm()
                    u = layer.weight @ v
                    u = u / u.norm()
                sigma = (u @ layer.weight @ v).item()
                results[name] = abs(sigma)

    get_spectral_norm(model.input_layer, "input_layer")
    for i, layer in enumerate(model.hidden_layers):
        get_spectral_norm(layer, f"hidden_layer_{i}")
    for i, layer in enumerate(model.skip_layers):
        get_spectral_norm(layer, f"skip_layer_{i}")
    get_spectral_norm(model.output_layer, "output_layer")

    return results
