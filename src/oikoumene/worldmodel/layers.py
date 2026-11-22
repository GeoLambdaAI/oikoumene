"""
JEPA layer primitives for world_model.py: forward/backward passes (GELU, Linear,
RMSNorm, AdaLN), the differentiable SIGReg regularizer and the Adam optimizer.
All backward passes are analytic and gradient-checked (test_world_model_gradcheck.py).
"""

import numpy as np

# ============================================================================
# Layer primitives (forward + backward)
# ============================================================================

# ============================================================================
# Activations
# ============================================================================

def gelu_forward(x):
    """GELU with tanh approximation (Hendrycks & Gimpel 2016)."""
    s = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(s)), (x, s)

def gelu_backward(dy, cache):
    x, s = cache
    th = np.tanh(s)
    ds_dx = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
    dy_dx = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th**2) * ds_dx
    return dy * dy_dx


# ============================================================================
# Linear: y = x @ W + b
# ============================================================================

def linear_forward(x, W, b):
    return x @ W + b, (x, W)

def linear_backward(dy, cache):
    x, W = cache
    # x: (batch, in), dy: (batch, out)
    # dW: (in, out), db: (out,), dx: (batch, in)
    # Reshape any leading dims to single batch axis for safety
    x2d = x.reshape(-1, x.shape[-1])
    dy2d = dy.reshape(-1, dy.shape[-1])
    dW = x2d.T @ dy2d
    db = dy2d.sum(axis=0)
    dx = (dy2d @ W.T).reshape(x.shape)
    return dx, dW, db


# ============================================================================
# RMSNorm: y = gamma * x / rms(x)
# ============================================================================

def rms_norm_forward(x, gamma, eps=1e-6):
    """x: (..., D), gamma: (D,)"""
    ms = np.mean(x**2, axis=-1, keepdims=True)
    r = np.sqrt(ms + eps)
    out = gamma * x / r
    return out, (x, gamma, r)

def rms_norm_backward(dy, cache):
    """
    Compute dL/dx and dL/dgamma.

    Math: y_i = gamma_i * x_i / r,  r = sqrt(mean(x^2) + eps)
    dy_i/dx_k = (gamma_i / r) * (delta_ik - x_i*x_k / (D * r^2))
    => dL/dx = (gamma * dy - x * mean(gamma * x * dy, axis=-1, keepdims=True) / r^2) / r
    """
    x, gamma, r = cache
    D = x.shape[-1]
    g_dy = gamma * dy  # (..., D)
    s = np.mean(g_dy * x, axis=-1, keepdims=True)  # (..., 1)
    dx = (g_dy - x * s / (r**2)) / r

    # dgamma: sum over all leading dims (batch)
    dgamma = np.sum(dy * x / r, axis=tuple(range(dy.ndim - 1)))
    return dx, dgamma


# ============================================================================
# AdaLN: y = rms_norm(h, gamma) * (1 + scale) + shift
# scale = action @ W_scale, shift = action @ W_shift
# ============================================================================

def adaln_forward(h, action, gamma, W_scale, W_shift, eps=1e-6):
    """h: (B, D), action: (B, A), gamma: (D,), W_scale/shift: (A, D)"""
    h_norm, rms_cache = rms_norm_forward(h, gamma, eps)
    scale = action @ W_scale
    shift = action @ W_shift
    out = h_norm * (1.0 + scale) + shift
    cache = (h_norm, action, scale, rms_cache)
    return out, cache

def adaln_backward(dy, cache, W_scale, W_shift):
    h_norm, action, scale, rms_cache = cache
    # y = h_norm * (1 + scale) + shift
    dh_norm = dy * (1.0 + scale)
    dscale = dy * h_norm
    dshift = dy

    # Backprop through linear projections from action
    dW_scale = action.T @ dscale
    dW_shift = action.T @ dshift
    daction = dscale @ W_scale.T + dshift @ W_shift.T

    # Backprop through rms_norm
    dh, dgamma = rms_norm_backward(dh_norm, rms_cache)
    return dh, daction, dgamma, dW_scale, dW_shift


# ============================================================================
# Differentiable SIGReg replacement
# ============================================================================
# The original SIGReg uses Cramer-Wold + Epps-Pulley which is non-trivial to
# differentiate. We replace it with a moments-matching formulation along
# random projections — same spirit (test for Gaussianity along random
# directions to prevent latent collapse), but smoothly differentiable.
#
# For unit-norm random directions u_k applied to centered/standardized
# projections Z @ u_k, a Gaussian has skewness 0 and excess kurtosis 0.
# This is the Jarque-Bera moment formulation.
# ============================================================================

def sigreg_forward(Z, n_projections=16, eps=1e-6, *, rng: np.random.RandomState):
    """
    Z: (batch, latent_dim)
    rng: seeded RandomState for the projection directions (required).
    Returns (loss, cache) where loss is differentiable w.r.t. Z.
    """
    B, D = Z.shape
    U = rng.randn(D, n_projections)
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10  # unit columns

    P = Z @ U  # (B, K)
    mu = P.mean(axis=0, keepdims=True)
    Pc = P - mu  # (B, K)
    var = (Pc**2).mean(axis=0, keepdims=True)  # (1, K)
    sigma = np.sqrt(var + eps)
    Pn = Pc / sigma  # standardized projections

    skew = (Pn**3).mean(axis=0)        # (K,)
    kurt = (Pn**4).mean(axis=0) - 3.0   # (K,) excess kurtosis

    # Also penalize variance shrinking to 0 (collapse) and exploding
    var_pen = ((sigma - 1.0)**2).mean()

    loss = (skew**2).mean() + (kurt**2).mean() + var_pen
    cache = (Z, U, mu, Pc, sigma, Pn, B, D, n_projections)
    return float(loss), cache

def sigreg_backward(cache):
    """
    Return dL/dZ. Loss is scalar so no upstream gradient needed.
    Implemented analytically term-by-term.
    """
    Z, U, mu, Pc, sigma, Pn, B, D, K = cache

    # We compute dL/dP first, then chain via dP/dZ = U.T (so dL/dZ = dL/dP @ U.T)
    # P = Z @ U  => dL/dZ = dL/dP @ U.T

    # mean and std are functions of P, so we need to be careful.
    # Standard approach: derive dL/dP_jk for each loss term.

    # Helper: for any per-column statistic f(P_:k), we have
    #   Pn_jk = (P_jk - mu_k) / sigma_k
    # with mu_k = mean over j, sigma_k = sqrt(mean((P-mu)^2) + eps)
    #
    # We want dL_skew/dP_jk where L_skew = (1/K) * sum_k skew_k^2,
    # skew_k = (1/B) * sum_j Pn_jk^3.
    #
    # By chain rule through the standardization, this is a known result:
    # for any function L = g(Pn) where Pn = (P - mu)/sigma per column,
    #   dL/dP_jk = (1/sigma_k) * [dL/dPn_jk - mean_j(dL/dPn) - Pn_jk * mean_j(dL/dPn * Pn)]
    # This is the standard "layer norm" backward pattern (along batch axis).

    # Skew loss
    coef_skew = 2.0 / K * (Pn**3).mean(axis=0)  # dL/dskew_k * skew_k weighting
    # Actually: L_s = (1/K) sum_k skew_k^2. dL_s/dskew_k = 2*skew_k/K
    # skew_k = mean_j Pn_jk^3 => dskew_k/dPn_jk = 3 * Pn_jk^2 / B
    skew_k = (Pn**3).mean(axis=0)  # (K,)
    dL_dskew = 2.0 * skew_k / K  # (K,)
    dL_dPn_skew = (3.0 * Pn**2 / B) * dL_dskew[None, :]  # (B, K)

    # Kurtosis loss: L_k = (1/K) sum_k kurt_k^2, kurt_k = mean Pn^4 - 3
    kurt_k = (Pn**4).mean(axis=0) - 3.0
    dL_dkurt = 2.0 * kurt_k / K
    dL_dPn_kurt = (4.0 * Pn**3 / B) * dL_dkurt[None, :]

    dL_dPn = dL_dPn_skew + dL_dPn_kurt  # (B, K)

    # Backprop through standardization (along batch axis):
    # dL/dP_jk = (1/sigma_k) * [dL/dPn_jk - mean_j(dL/dPn) - Pn_jk * mean_j(dL/dPn * Pn)]
    mean_dPn = dL_dPn.mean(axis=0, keepdims=True)
    mean_dPn_Pn = (dL_dPn * Pn).mean(axis=0, keepdims=True)
    dL_dP_main = (dL_dPn - mean_dPn - Pn * mean_dPn_Pn) / sigma  # (B, K)

    # Variance penalty: var_pen = mean_k (sigma_k - 1)^2
    # dvar_pen/dsigma_k = 2*(sigma_k - 1)/K
    # dsigma_k/dP_jk = (1/(B * sigma_k)) * (P_jk - mu_k) = Pn_jk / B
    dvar_dsigma = 2.0 * (sigma - 1.0) / K  # (1, K)
    dL_dP_var = dvar_dsigma * Pn / B  # (B, K)

    dL_dP = dL_dP_main + dL_dP_var

    # Chain through P = Z @ U
    dL_dZ = dL_dP @ U.T  # (B, D)
    return dL_dZ


# ============================================================================
# Adam optimizer (per-parameter state)
# ============================================================================

class Adam:
    """Adam optimizer over a flat dict of named parameters."""
    def __init__(self, params: dict, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
