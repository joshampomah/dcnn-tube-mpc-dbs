"""Jacobian computation and CVXPY-compatible evaluation for DC-NN models.

This module provides functions to compute gradients (Jacobians) of the DC-NN
predictor with respect to control inputs, as well as CVXPY-compatible forward
passes for embedding ICNNs directly in convex optimization problems.

For DC structure: f = f1 - f2, the Jacobian is: df/du = df1/du - df2/du

This implementation uses the analytical approach (layer-by-layer chain rule)
rather than automatic differentiation. Benefits:
- Provides separate A (df/dx) and B (df/du) matrices for state-space MPC
- Compatible with CVXPy for convex optimization
- More interpretable and avoids autograd overhead
- Enables caching of intermediate activations

CVXPY Integration (CDC25 DC-MPC):
- forward_from_weights_cvxpy(): Evaluates ICNN using CVXPY expressions
- Enables proper DC constraints where convex parts are kept exact
- Used for asymmetric linearization in tube MPC (equations 6-7 of CDC25)
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Activation functions for analytical computation
# =============================================================================


def _relu_numpy(x: np.ndarray) -> np.ndarray:
    """ReLU activation (numpy)."""
    return np.maximum(x, 0)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: Heaviside step function.

    Returns diagonal matrix for chain rule: diag(heaviside(z))

    Args:
        z: Pre-activation values, shape (n,) or (n, 1)

    Returns:
        Diagonal matrix of shape (n, n) where diagonal[i] = 1 if z[i] > 0 else 0
    """
    return np.diag(np.heaviside(z.flatten(), 0))


def _relu_derivative_vector(z: np.ndarray) -> np.ndarray:
    """Vectorized derivative of ReLU for efficient chain rule.

    Returns a 1D vector instead of diagonal matrix, enabling efficient
    element-wise multiplication (broadcasting) instead of matrix multiply.

    Args:
        z: Pre-activation values, shape (n,) or (n, 1)

    Returns:
        Vector of shape (n,) where result[i] = 1 if z[i] > 0 else 0
    """
    return np.heaviside(z.flatten(), 0)


# =============================================================================
# Weight extraction from PyTorch models
# =============================================================================


def extract_weights_from_convex_nn(model: nn.Module) -> List[np.ndarray]:
    """Extract weights from a ConvexNN model for analytical computation.

    The weight order matches the supervisor's convention:
    - weights[0], weights[1]: First layer W, b
    - weights[2+i*4 : 2+(i+1)*4]: Internal layer i (Wx, bx, W0, b0)
    - weights[-2], weights[-1]: Output layer W, b

    Args:
        model: ConvexNN PyTorch module

    Returns:
        List of weight arrays in format:
            [W0, b0, Wx1, bx1, W0_1, b0_1, ..., W_out, b_out]
    """
    weights = []

    # First layer: input_layer
    weights.append(model.input_layer.weight.detach().cpu().numpy())  # (n_hidden, n_input)
    weights.append(model.input_layer.bias.detach().cpu().numpy())    # (n_hidden,)

    # Internal layers: hidden_layers[i] and skip_layers[i]
    for hidden, skip in zip(model.hidden_layers, model.skip_layers):
        weights.append(hidden.weight.detach().cpu().numpy())  # Wx: (n_hidden, n_hidden)
        weights.append(hidden.bias.detach().cpu().numpy())    # bx: (n_hidden,)
        weights.append(skip.weight.detach().cpu().numpy())    # W0: (n_hidden, n_input)
        weights.append(skip.bias.detach().cpu().numpy())      # b0: (n_hidden,)

    # Output layer
    weights.append(model.output_layer.weight.detach().cpu().numpy())  # (1, n_hidden)
    weights.append(model.output_layer.bias.detach().cpu().numpy())    # (1,)

    return weights


# =============================================================================
# Analytical forward pass (CVXPy compatible)
# =============================================================================


def forward_from_weights(
    x: np.ndarray,
    weights: List[np.ndarray],
    sigma=None,
) -> np.ndarray:
    """Forward pass through ConvexNN using extracted weights.

    Compatible with CVXPy when sigma = lambda x: cp.maximum(x, 0).

    Args:
        x: Input vector (n_input,) or (n_input, 1)
        weights: List of weight arrays from extract_weights_from_convex_nn
        sigma: Activation function (default: numpy ReLU)

    Returns:
        Output value as 1D array

    Example:
        >>> weights = extract_weights_from_convex_nn(model)
        >>> y = forward_from_weights(x, weights)
        >>> # For CVXPy optimization:
        >>> import cvxpy as cp
        >>> sigma_cvx = lambda x: cp.maximum(x, 0)
        >>> y_cvx = forward_from_weights(x_var, weights, sigma=sigma_cvx)
    """
    if sigma is None:
        sigma = _relu_numpy

    # Ensure column vector
    if x.ndim == 1:
        x = x[:, None]

    # First layer
    x0 = x  # Store for skip connections
    W = weights[0]  # (n_hidden, n_input)
    b = weights[1]  # (n_hidden,)
    z = W @ x + b[:, None]
    h = sigma(z)

    # Internal layers
    n_internal = (len(weights) - 4) // 4
    for i in range(n_internal):
        Wx = weights[2 + i*4]      # (n_hidden, n_hidden)
        bx = weights[2 + i*4 + 1]  # (n_hidden,)
        W0 = weights[2 + i*4 + 2]  # (n_hidden, n_input)
        b0 = weights[2 + i*4 + 3]  # (n_hidden,)

        z = Wx @ h + bx[:, None] + W0 @ x0 + b0[:, None]
        h = sigma(z)

    # Output layer (no activation)
    W_out = weights[-2]  # (1, n_hidden)
    b_out = weights[-1]  # (1,)
    y = W_out @ h + b_out[:, None]

    return y.flatten()


# =============================================================================
# CVXPY-compatible ICNN evaluation (for DC-MPC constraints)
# =============================================================================


def forward_from_weights_cvxpy(
    z_k: "cp.Parameter",
    u: "cp.Variable",
    weights: List[np.ndarray],
    nonneg_params: List["cp.Parameter"] = None,
) -> "cp.Expression":
    """Forward pass through ConvexNN using CVXPY expressions.

    This enables embedding ICNN evaluations directly in CVXPY optimization
    problems, which is required for the CDC25 DC-MPC formulation where
    convex parts are evaluated exactly (not linearized).

    The network input is [z_k, u] where:
    - z_k is a CVXPY Parameter (fixed state, updated at each timestep)
    - u is a CVXPY Variable (control inputs being optimized)

    IMPORTANT: For CVXPY to verify DCP compliance, hidden/output layer weights
    must be declared as non-negative Parameters. Use build_icnn_cvxpy_params()
    to create properly structured parameters.

    Args:
        z_k: CVXPY Parameter of shape (n_state,) containing past observations.
        u: CVXPY Variable of shape (n_u,) for control inputs.
        weights: List of weight arrays from extract_weights_from_convex_nn.
        nonneg_params: Pre-built non-negative Parameters for hidden/output weights.
            If None, uses numpy arrays directly (won't pass DCP check).

    Returns:
        CVXPY Expression representing the network output (scalar).

    Example:
        >>> import cvxpy as cp
        >>> z_k = cp.Parameter(15, name="z_k")
        >>> u = cp.Variable(3, name="u")
        >>> weights = extract_weights_from_convex_nn(model.f1)
        >>> nonneg = build_icnn_cvxpy_params(weights, "f1")
        >>> f1_expr = forward_from_weights_cvxpy(z_k, u, weights, nonneg)
    """
    import cvxpy as cp

    # Concatenate state parameter and control variable
    x = cp.hstack([z_k, u])
    x0 = x  # Store for skip connections

    # First layer: z = W @ x + b, h = ReLU(z)
    W = weights[0]  # (n_hidden, n_input) - unconstrained
    b = weights[1]  # (n_hidden,)
    z = W @ x + b
    h = cp.maximum(z, 0)

    # Internal layers with skip connections
    n_internal = (len(weights) - 4) // 4

    if nonneg_params is not None:
        # Use pre-built non-negative Parameters for DCP compliance
        param_idx = 0
        for i in range(n_internal):
            Wx_param = nonneg_params[param_idx]  # Non-negative Parameter
            param_idx += 1
            bx = weights[2 + i*4 + 1]
            W0 = weights[2 + i*4 + 2]  # Skip weights - unconstrained
            b0 = weights[2 + i*4 + 3]

            z = Wx_param @ h + bx + W0 @ x0 + b0
            h = cp.maximum(z, 0)

        # Output layer
        W_out_param = nonneg_params[param_idx]  # Non-negative Parameter
        b_out = weights[-1]
        y = W_out_param @ h + b_out
    else:
        # Direct numpy arrays (won't pass DCP for general weights)
        for i in range(n_internal):
            Wx = np.maximum(weights[2 + i*4], 0)  # Clip for safety
            bx = weights[2 + i*4 + 1]
            W0 = weights[2 + i*4 + 2]
            b0 = weights[2 + i*4 + 3]

            z = Wx @ h + bx + W0 @ x0 + b0
            h = cp.maximum(z, 0)

        W_out = np.maximum(weights[-2], 0)
        b_out = weights[-1]
        y = W_out @ h + b_out

    return y[0] if y.shape else y


def forward_from_weights_cvxpy_epigraph(
    z_k: "cp.Parameter",
    u: "cp.Variable",
    weights: List[np.ndarray],
    prefix: str = "",
) -> Tuple["cp.Expression", List, List["cp.Variable"]]:
    """Forward pass through ConvexNN using epigraph formulation.

    Instead of using cp.maximum (which creates complex expression trees),
    this uses auxiliary variables with explicit constraints:
        h = max(z, 0)  -->  h >= z, h >= 0, h is Variable

    This may improve canonicalization speed since the constraint structure
    is more explicit and simpler for CVXPY to process.

    Args:
        z_k: CVXPY Parameter of shape (n_state,) containing past observations.
        u: CVXPY Variable of shape (n_u,) for control inputs.
        weights: List of weight arrays from extract_weights_from_convex_nn.
        prefix: Name prefix for auxiliary variables.

    Returns:
        Tuple of (output_expression, constraints_list, auxiliary_variables_list).
        The constraints must be added to the problem for correctness.
    """
    import cvxpy as cp

    constraints = []
    aux_vars = []

    # Concatenate state parameter and control variable
    x = cp.hstack([z_k, u])
    x0 = x  # Store for skip connections

    n_hidden = weights[0].shape[0]

    # First layer: z = W @ x + b, h = ReLU(z) via epigraph
    W = weights[0]  # (n_hidden, n_input)
    b = weights[1]  # (n_hidden,)
    z = W @ x + b

    # Epigraph: h >= z, h >= 0 (replaces h = max(z, 0))
    h = cp.Variable(n_hidden, name=f"{prefix}_h0")
    aux_vars.append(h)
    constraints.append(h >= z)
    constraints.append(h >= 0)

    # Internal layers with skip connections
    n_internal = (len(weights) - 4) // 4

    for i in range(n_internal):
        Wx = np.maximum(weights[2 + i*4], 0)  # Non-negative weights
        bx = weights[2 + i*4 + 1]
        W0 = weights[2 + i*4 + 2]  # Skip weights
        b0 = weights[2 + i*4 + 3]

        z = Wx @ h + bx + W0 @ x0 + b0

        # Epigraph for this layer
        h_new = cp.Variable(n_hidden, name=f"{prefix}_h{i+1}")
        aux_vars.append(h_new)
        constraints.append(h_new >= z)
        constraints.append(h_new >= 0)
        h = h_new

    # Output layer (no activation)
    W_out = np.maximum(weights[-2], 0)
    b_out = weights[-1]
    y = W_out @ h + b_out

    return y[0] if y.shape else y, constraints, aux_vars


def build_icnn_cvxpy_params(
    weights: List[np.ndarray],
    prefix: str = "",
) -> List["cp.Parameter"]:
    """Build CVXPY Parameters for ICNN hidden/output weights with nonneg=True.

    This creates Parameters that tell CVXPY the weights are non-negative,
    enabling DCP verification for the ICNN forward pass.

    Args:
        weights: List of weight arrays from extract_weights_from_convex_nn.
        prefix: Name prefix for parameters.

    Returns:
        List of non-negative CVXPY Parameters for hidden layer and output weights.
    """
    import cvxpy as cp

    params = []
    n_internal = (len(weights) - 4) // 4

    # Hidden layer weights (must be non-negative)
    for i in range(n_internal):
        Wx = weights[2 + i*4]
        Wx_clipped = np.maximum(Wx, 0)  # Ensure non-negative
        param = cp.Parameter(Wx.shape, nonneg=True, name=f"{prefix}_Wx_{i}")
        param.value = Wx_clipped
        params.append(param)

    # Output layer weights (must be non-negative)
    W_out = weights[-2]
    W_out_clipped = np.maximum(W_out, 0)
    param = cp.Parameter(W_out.shape, nonneg=True, name=f"{prefix}_Wout")
    param.value = W_out_clipped
    params.append(param)

    return params


def compute_activation_pattern(
    z_k: np.ndarray,
    u: np.ndarray,
    weights: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute which neurons are active at given input.

    Returns binary masks for each layer indicating which neurons have
    positive pre-activation values (and thus pass through the ReLU).

    Args:
        z_k: State vector of shape (n_state,).
        u: Control input of shape (n_u,).
        weights: List of weight arrays from extract_weights_from_convex_nn.

    Returns:
        List of binary masks, one per layer (including first and internal layers).
    """
    x = np.hstack([z_k, u])
    x0 = x

    activation_masks = []

    # First layer
    W = weights[0]
    b = weights[1]
    z = W @ x + b
    activation_masks.append((z > 0).astype(np.float32))
    h = np.maximum(z, 0)

    # Internal layers
    n_internal = (len(weights) - 4) // 4
    for i in range(n_internal):
        Wx = weights[2 + i*4]
        bx = weights[2 + i*4 + 1]
        W0 = weights[2 + i*4 + 2]
        b0 = weights[2 + i*4 + 3]

        z = Wx @ h + bx + W0 @ x0 + b0
        activation_masks.append((z > 0).astype(np.float32))
        h = np.maximum(z, 0)

    return activation_masks


def forward_from_weights_cvxpy_linearized(
    u: "cp.Variable",
    weights: List[np.ndarray],
    z_k_value: np.ndarray,
    activation_masks: List[np.ndarray],
    nonneg_params: List["cp.Parameter"] = None,
) -> "cp.Expression":
    """Forward pass using fixed activation pattern (DPP-compliant).

    Instead of using cp.maximum (which breaks DPP), this uses a pre-computed
    activation pattern to create a linear function of u. This enables CVXPY
    to cache the canonical form between solves.

    The key insight is that for a given activation pattern, the ICNN is a
    linear function of the input. We pre-compute which neurons are active
    based on the nominal operating point, then use that fixed pattern.

    Args:
        u: CVXPY Variable of shape (n_u,) for control inputs.
        weights: List of weight arrays from extract_weights_from_convex_nn.
        z_k_value: Current state as numpy array (not CVXPY parameter).
        activation_masks: Pre-computed activation patterns from compute_activation_pattern.
        nonneg_params: Pre-built non-negative Parameters (optional, for consistency).

    Returns:
        CVXPY Expression representing the network output (scalar).
    """
    import cvxpy as cp

    n_state = z_k_value.shape[0]
    n_u = u.shape[0]

    # Split weight matrices into state and control parts
    W = weights[0]  # (n_hidden, n_state + n_u)
    b = weights[1]
    W_z = W[:, :n_state]  # State part
    W_u = W[:, n_state:]  # Control part

    # First layer: apply activation mask
    mask = activation_masks[0]
    state_contribution = W_z @ z_k_value + b  # numpy constant
    z_affine = W_u @ u + state_contribution    # affine in u
    h = cp.multiply(mask, z_affine)            # element-wise, linear in u

    # Internal layers
    n_internal = (len(weights) - 4) // 4
    x0_z = z_k_value
    x0_u = u

    for i in range(n_internal):
        Wx = np.maximum(weights[2 + i*4], 0)  # Non-negative weights
        bx = weights[2 + i*4 + 1]
        W0 = weights[2 + i*4 + 2]
        b0 = weights[2 + i*4 + 3]
        W0_z = W0[:, :n_state]
        W0_u = W0[:, n_state:]

        mask = activation_masks[i + 1]
        state_contribution = W0_z @ x0_z + b0  # numpy constant
        z_affine = Wx @ h + bx + W0_u @ x0_u + state_contribution
        h = cp.multiply(mask, z_affine)

    # Output layer (no activation)
    W_out = np.maximum(weights[-2], 0)
    b_out = weights[-1]
    y = W_out @ h + b_out

    return y[0] if hasattr(y, '__len__') and len(y.shape) > 0 else y


# =============================================================================
# Analytical Jacobian computation
# =============================================================================


def compute_jacobian_analytical(
    x: np.ndarray,
    weights: List[np.ndarray],
    n_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian analytically using layer-by-layer chain rule.

    Computes both df/dx (A matrix) and df/du (B matrix) separately.

    For input [x_state, u], the Jacobian is partitioned as:
        df/d[x,u] = [A | B] where A = df/dx, B = df/du

    Args:
        x: Full input vector [state, control] of shape (n_state + n_control,)
        weights: List of weight arrays from extract_weights_from_convex_nn
        n_state: Number of state variables (to split Jacobian into A and B)

    Returns:
        Tuple (A, B) where:
            A: Jacobian w.r.t. state, shape (1, n_state)
            B: Jacobian w.r.t. control, shape (1, n_control)
    """
    sigma = _relu_numpy

    # Ensure column vector
    if x.ndim == 1:
        x = x[:, None]

    # First layer
    x0 = x  # Store for skip connections
    W = weights[0]  # (n_hidden, n_input)
    b = weights[1]  # (n_hidden,)
    z = W @ x + b[:, None]
    h = sigma(z)

    # Initialize Jacobians: df/d[state] and df/d[control]
    dsigma_z = _relu_derivative_vector(z[:, 0])  # (n_hidden,) vector
    A = dsigma_z[:, None] * W[:, :n_state]   # (n_hidden, n_state) via broadcasting
    B = dsigma_z[:, None] * W[:, n_state:]   # (n_hidden, n_control) via broadcasting

    # Internal layers
    n_internal = (len(weights) - 4) // 4
    for i in range(n_internal):
        Wx = weights[2 + i*4]      # (n_hidden, n_hidden) - from previous layer
        bx = weights[2 + i*4 + 1]  # (n_hidden,)
        W0 = weights[2 + i*4 + 2]  # (n_hidden, n_input) - skip from input
        b0 = weights[2 + i*4 + 3]  # (n_hidden,)

        z = Wx @ h + bx[:, None] + W0 @ x0 + b0[:, None]
        h = sigma(z)

        dsigma_z = _relu_derivative_vector(z[:, 0])  # (n_hidden,) vector
        A = dsigma_z[:, None] * (Wx @ A + W0[:, :n_state])
        B = dsigma_z[:, None] * (Wx @ B + W0[:, n_state:])

    # Output layer (no activation, so no dsigma)
    W_out = weights[-2]  # (1, n_hidden)
    A = W_out @ A  # (1, n_state)
    B = W_out @ B  # (1, n_control)

    return A, B


def compute_dcnn_jacobian_analytical(
    model_f1: nn.Module,
    model_f2: nn.Module,
    x: np.ndarray,
    u: np.ndarray,
    n_state: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian of DC-NN (f1 - f2) analytically.

    Args:
        model_f1: Convex component f1 (PyTorch ConvexNN)
        model_f2: Convex component f2 (PyTorch ConvexNN)
        x: State vector (n_state,)
        u: Control vector (n_control,)
        n_state: Number of states (default: inferred from x)

    Returns:
        Tuple (A, B) where:
            A: d(f1-f2)/dx, shape (1, n_state)
            B: d(f1-f2)/du, shape (1, n_control)
    """
    if n_state is None:
        n_state = x.shape[0]

    weights_f1 = extract_weights_from_convex_nn(model_f1)
    weights_f2 = extract_weights_from_convex_nn(model_f2)

    full_input = np.hstack([x, u])

    A1, B1 = compute_jacobian_analytical(full_input, weights_f1, n_state)
    A2, B2 = compute_jacobian_analytical(full_input, weights_f2, n_state)

    A = A1 - A2
    B = B1 - B2

    return A, B


# =============================================================================
# Main API functions
# =============================================================================


def compute_jacobian_wrt_u(
    model_f1: nn.Module,
    model_f2: nn.Module,
    x: np.ndarray,
    u: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Compute Jacobian of f = f1 - f2 with respect to control inputs u.

    For DC structure, the Jacobian decomposes as:
        df/du = df1/du - df2/du

    Args:
        model_f1: Convex component f1 (PyTorch module)
        model_f2: Convex component f2 (PyTorch module)
        x: State vector (n_state,)
        u: Control input vector (k,)
        device: Device parameter (kept for API compatibility, not used)

    Returns:
        Jacobian matrix (1, k) = d_beta_{k}/d_u_{0:k-1}
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    if u.ndim != 1:
        raise ValueError(f"u must be 1D, got shape {u.shape}")

    _, B = compute_dcnn_jacobian_analytical(model_f1, model_f2, x, u, n_state=x.shape[0])
    return B


def compute_jacobian_batch(
    model_f1: nn.Module,
    model_f2: nn.Module,
    x_batch: np.ndarray,
    u_batch: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Compute Jacobians for a batch of inputs.

    Args:
        model_f1: Convex component f1 (PyTorch module)
        model_f2: Convex component f2 (PyTorch module)
        x_batch: State vectors (B, n_state)
        u_batch: Control inputs (B, k)
        device: Device parameter (kept for API compatibility, not used)

    Returns:
        Jacobian matrices (B, k) - one Jacobian per input pair
    """
    if x_batch.ndim != 2:
        raise ValueError(f"x_batch must be 2D, got shape {x_batch.shape}")
    if u_batch.ndim != 2:
        raise ValueError(f"u_batch must be 2D, got shape {u_batch.shape}")
    if x_batch.shape[0] != u_batch.shape[0]:
        raise ValueError(
            f"Batch size mismatch: x_batch {x_batch.shape[0]} vs "
            f"u_batch {u_batch.shape[0]}"
        )

    batch_size = x_batch.shape[0]
    n_u = u_batch.shape[1]
    n_state = x_batch.shape[1]

    weights_f1 = extract_weights_from_convex_nn(model_f1)
    weights_f2 = extract_weights_from_convex_nn(model_f2)

    jacobian_batch = np.zeros((batch_size, n_u), dtype=np.float32)

    for i in range(batch_size):
        full_input = np.hstack([x_batch[i], u_batch[i]])
        _, B1 = compute_jacobian_analytical(full_input, weights_f1, n_state)
        _, B2 = compute_jacobian_analytical(full_input, weights_f2, n_state)
        jacobian_batch[i] = (B1 - B2).flatten()

    return jacobian_batch


def verify_jacobian_finite_diff(
    model_f1: nn.Module,
    model_f2: nn.Module,
    x: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-4,
    tol: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[bool, float]:
    """Verify analytical Jacobian against finite-difference approximation.

    Args:
        model_f1: Convex component f1
        model_f2: Convex component f2
        x: State vector (n_state,)
        u: Control input vector (k,)
        eps: Finite difference step size (default: 1e-4)
        tol: Tolerance for max absolute error (default: 1e-3)
        device: Device to run computation on

    Returns:
        Tuple of (is_correct, max_error):
            - is_correct: True if max error < tol
            - max_error: Maximum absolute difference between analytic and numeric
    """
    jac_analytic = compute_jacobian_wrt_u(model_f1, model_f2, x, u, device)

    weights_f1 = extract_weights_from_convex_nn(model_f1)
    weights_f2 = extract_weights_from_convex_nn(model_f2)

    def eval_f(u_val: np.ndarray) -> float:
        full_input = np.hstack([x, u_val])
        f1_val = forward_from_weights(full_input, weights_f1)[0]
        f2_val = forward_from_weights(full_input, weights_f2)[0]
        return f1_val - f2_val

    n_u = u.shape[0]
    jac_numeric = np.zeros((1, n_u), dtype=np.float32)

    for i in range(n_u):
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += eps
        u_minus[i] -= eps
        jac_numeric[0, i] = (eval_f(u_plus) - eval_f(u_minus)) / (2 * eps)

    error = np.abs(jac_analytic - jac_numeric)
    max_error = float(np.max(error))
    is_correct = max_error < tol

    return is_correct, max_error


# =============================================================================
# Additional utilities for SCP/MPC
# =============================================================================


def compute_component_jacobian_analytical(
    model: nn.Module,
    x: np.ndarray,
    u: np.ndarray,
    weights: List[np.ndarray] = None,
) -> np.ndarray:
    """Compute Jacobian of a single convex component w.r.t. control inputs.

    Args:
        model: Convex neural network (f1 or f2).
        x: State vector of shape (n_state,).
        u: Control inputs of shape (k,).
        weights: Pre-extracted weights from extract_weights_from_convex_nn().
            If None, extracts weights from model (slower).

    Returns:
        Jacobian array of shape (1, k).
    """
    if weights is None:
        weights = extract_weights_from_convex_nn(model)
    full_input = np.hstack([x, u])
    _, B = compute_jacobian_analytical(full_input, weights, n_state=x.shape[0])
    return B


def verify_jacobian_against_autograd(
    model_f1: nn.Module,
    model_f2: nn.Module,
    x: np.ndarray,
    u: np.ndarray,
    tol: float = 1e-5,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """Compare analytical Jacobian against PyTorch autograd.

    Args:
        model_f1: Convex component f1
        model_f2: Convex component f2
        x: State vector (n_state,)
        u: Control input vector (k,)
        tol: Tolerance for max absolute error
        device: Device for autograd computation

    Returns:
        Tuple of (is_correct, max_error, jac_analytical, jac_autograd)
    """
    jac_analytical = compute_jacobian_wrt_u(model_f1, model_f2, x, u)

    device = torch.device(device)
    model_f1 = model_f1.to(device).eval()
    model_f2 = model_f2.to(device).eval()

    n_u = u.shape[0]
    inputs = torch.tensor(
        np.hstack([x, u]).reshape(1, -1),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    f1_out = model_f1(inputs)
    f2_out = model_f2(inputs)
    f_out = f1_out - f2_out

    grad = torch.autograd.grad(f_out, inputs, create_graph=False)[0]
    jac_autograd = grad[:, -n_u:].detach().cpu().numpy()

    error = np.abs(jac_analytical - jac_autograd)
    max_error = float(np.max(error))
    is_correct = max_error < tol

    return is_correct, max_error, jac_analytical, jac_autograd
