# Canonical owner: closed-loop-dbs-bench
"""PyTorch implementation of DC-NN (Difference of Convex Neural Networks).

This module provides PyTorch equivalents of the DC-NN models,
enabling GPU acceleration via CUDA/MPS for training and inference.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .spectral_norm import ScaledSpectralNorm


class ConvexNN(nn.Module):
    """Input-convex neural network with non-negative weight constraints.

    Architecture matches the TensorFlow implementation:
    - First layer: unconstrained Dense + ReLU
    - Hidden layers: non-negative weights + skip connections from input + ReLU
    - Output layer: non-negative weights

    The non-negative weight constraints ensure convexity in the input.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_layers: int,
        use_spectral_norm: bool = False,
        spectral_norm_target: float = 0.5,
        spectral_norm_n_power_iterations: int = 1,
        spectral_norm_layers: str = "all",
    ):
        """Initialize convex neural network.

        Args:
            n_input: Input dimension
            n_hidden: Hidden layer width
            n_layers: Number of hidden layers (after initial layer)
            use_spectral_norm: Enable spectral normalization for Lipschitz bound
            spectral_norm_target: Per-layer target spectral norm
            spectral_norm_n_power_iterations: Power iterations for SN estimation
            spectral_norm_layers: Which layers to apply SN ("all", "hidden_only", "internal")
        """
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.use_spectral_norm = use_spectral_norm
        self.spectral_norm_target = spectral_norm_target
        self.spectral_norm_n_power_iterations = spectral_norm_n_power_iterations
        self.spectral_norm_layers = spectral_norm_layers

        # First layer: unconstrained
        self.input_layer = nn.Linear(n_input, n_hidden)

        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        for _ in range(n_layers):
            # Main path: non-negative weights (enforced after optimizer step)
            self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            # Skip connection from input: unconstrained
            self.skip_layers.append(nn.Linear(n_input, n_hidden))

        # Output layer: non-negative weights
        self.output_layer = nn.Linear(n_hidden, 1)

        # Initialize weights
        self._init_weights()

        # Apply spectral normalization if enabled
        if use_spectral_norm:
            self._apply_spectral_norm()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_spectral_norm(self):
        """Apply spectral normalization to selected layers."""
        from .spectral_norm import apply_spectral_norm_to_convex_nn
        apply_spectral_norm_to_convex_nn(
            self,
            target=self.spectral_norm_target,
            n_power_iterations=self.spectral_norm_n_power_iterations,
            layers=self.spectral_norm_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_input)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Store input for skip connections
        x_input = x

        # First layer (unconstrained)
        h = torch.relu(self.input_layer(x))

        # Hidden layers with skip connections
        for hidden, skip in zip(self.hidden_layers, self.skip_layers):
            h = torch.relu(hidden(h) + skip(x_input))

        # Output layer
        return self.output_layer(h)

    def enforce_constraints(self):
        """Clamp weights to non-negative after optimizer step.

        Call this after each optimizer.step() to maintain convexity.
        Handles both regular layers and spectral-normalized layers.
        """
        with torch.no_grad():
            if not self.use_spectral_norm:
                # Fast path: direct weight access (no SN wrappers)
                for layer in self.hidden_layers:
                    layer.weight.data.clamp_(min=0)
                self.output_layer.weight.data.clamp_(min=0)
            else:
                # Slow path: handle spectral norm wrappers
                for layer in self.hidden_layers:
                    w = self._get_weight_tensor(layer)
                    w.clamp_(min=0)
                w = self._get_weight_tensor(self.output_layer)
                w.clamp_(min=0)

    def _get_weight_tensor(self, layer) -> torch.Tensor:
        """Get the weight tensor from a layer, handling spectral norm wrappers."""
        if isinstance(layer, ScaledSpectralNorm):
            inner = layer.module
            if hasattr(inner, 'parametrizations') and hasattr(inner.parametrizations, 'weight'):
                return inner.parametrizations.weight.original.data
            elif hasattr(inner, 'weight_orig'):
                return inner.weight_orig.data
            return inner.weight.data
        elif hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, 'weight'):
            return layer.parametrizations.weight.original.data
        elif hasattr(layer, 'weight_orig'):
            return layer.weight_orig.data
        else:
            return layer.weight.data


class DCNNModel(nn.Module):
    """Difference of Convex Neural Networks (DC-NN).

    Represents f(x) = f1(x) - f2(x) where f1 and f2 are convex NNs.
    This allows representing non-convex functions while maintaining
    useful properties for optimization (DC programming).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 64,
        n_layers: int = 1,
        use_spectral_norm: bool = False,
        spectral_norm_target: float = 0.5,
        spectral_norm_n_power_iterations: int = 1,
        spectral_norm_layers: str = "all",
    ):
        """Initialize DC-NN model."""
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.use_spectral_norm = use_spectral_norm
        self.spectral_norm_target = spectral_norm_target
        self.spectral_norm_n_power_iterations = spectral_norm_n_power_iterations
        self.spectral_norm_layers = spectral_norm_layers

        # Two convex networks
        self.f1 = ConvexNN(
            n_input, n_hidden, n_layers,
            use_spectral_norm=use_spectral_norm,
            spectral_norm_target=spectral_norm_target,
            spectral_norm_n_power_iterations=spectral_norm_n_power_iterations,
            spectral_norm_layers=spectral_norm_layers,
        )
        self.f2 = ConvexNN(
            n_input, n_hidden, n_layers,
            use_spectral_norm=use_spectral_norm,
            spectral_norm_target=spectral_norm_target,
            spectral_norm_n_power_iterations=spectral_norm_n_power_iterations,
            spectral_norm_layers=spectral_norm_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: f(x) = f1(x) - f2(x)."""
        return self.f1(x) - self.f2(x)

    def enforce_constraints(self):
        """Enforce non-negative weight constraints on both convex networks."""
        self.f1.enforce_constraints()
        self.f2.enforce_constraints()


class MultiStepDCNN(nn.Module):
    """Multi-step DC-NN predictor with N independent k-step networks.

    Each network k predicts k steps ahead directly (not recursively).
    Network k takes input [state, u[0:k]] and outputs y[k].
    """

    def __init__(
        self,
        n_state: int = 15,
        n_input: int = 1,
        n_hidden: int = 64,
        n_layers: int = 1,
        horizon: int = 5,
        use_spectral_norm: bool = False,
        spectral_norm_target: float = 0.5,
        spectral_norm_n_power_iterations: int = 1,
        spectral_norm_layers: str = "all",
    ):
        """Initialize multi-step predictor."""
        super().__init__()
        self.n_state = n_state
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.horizon = horizon
        self.use_spectral_norm = use_spectral_norm
        self.spectral_norm_target = spectral_norm_target
        self.spectral_norm_n_power_iterations = spectral_norm_n_power_iterations
        self.spectral_norm_layers = spectral_norm_layers

        # Create N independent networks
        self.networks = nn.ModuleList()
        for k in range(1, horizon + 1):
            # Network k has input dimension: n_state + k * n_input
            input_dim = n_state + k * n_input
            self.networks.append(DCNNModel(
                input_dim, n_hidden, n_layers,
                use_spectral_norm=use_spectral_norm,
                spectral_norm_target=spectral_norm_target,
                spectral_norm_n_power_iterations=spectral_norm_n_power_iterations,
                spectral_norm_layers=spectral_norm_layers,
            ))

    def forward_k(self, x: torch.Tensor, u: torch.Tensor, k: int) -> torch.Tensor:
        """Predict k steps ahead."""
        inputs = torch.cat([x, u[:, :k]], dim=1)
        return self.networks[k - 1](inputs)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Predict all steps 1 to horizon."""
        predictions = []
        for k in range(1, self.horizon + 1):
            predictions.append(self.forward_k(x, u, k))
        return tuple(predictions)

    def enforce_constraints(self):
        """Enforce non-negative weight constraints on all networks."""
        for network in self.networks:
            network.enforce_constraints()


def get_device(prefer_gpu: bool = True, force_device: str = None) -> torch.device:
    """Get the best available device for computation."""
    if force_device is not None:
        if force_device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            raise ValueError("CUDA requested but not available")
        elif force_device == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            raise ValueError("MPS requested but not available")
        elif force_device == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device type: {force_device}")

    if not prefer_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
            return torch.device("mps")
        except Exception:
            pass

    return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """Get information about a device for logging/reporting."""
    info = {"type": device.type, "device": str(device)}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["total_memory_gb"] = props.total_memory / 1e9
        info["compute_capability"] = f"{props.major}.{props.minor}"
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
        info["total_memory_gb"] = None
    else:
        info["name"] = "CPU"
        info["total_memory_gb"] = None

    return info


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
