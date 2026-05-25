"""
JEPA World Model — pure NumPy implementation with hand-written backprop.

Inspired by:
- LeCun (2022). A Path Towards Autonomous Machine Intelligence.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). LeWorldModel.
- Qu et al. (2026). Representation Learning for Spatiotemporal Physical Systems.

Key components:
- WorldEncoder: 3-layer MLP with RMSNorm + GELU, maps observations to latents
- WorldPredictor: 2-layer MLP with AdaLN action conditioning
- SIGReg: differentiable moments-based proxy for Cramer-Wold gaussianity test
- CEMPlanner: cross-entropy method for action selection in latent space
- JEPAWorldModel: orchestrates training with Adam optimizer

Notes on the v0.2 rewrite (replacing the random-perturbation gradients of v0.1):
- All backward passes are analytic and verified by finite-difference gradient
  check (see test_world_model_gradcheck.py). Relative error < 1e-8.
- AdaLN parameters (W_ada1_scale, W_ada1_shift, W_ada2_scale, W_ada2_shift)
  are now in the trainable parameter set — previously they were not.
- SIGReg gradients propagate into the loss — previously the regularizer was
  computed but did not influence training.
- AdaLN scale/shift weights are zero-initialized so that initial conditioning
  is identity (the standard DiT initialization trick).
"""

import numpy as np
from typing import Optional

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

def sigreg_forward(Z, n_projections=16, eps=1e-6, rng=None):
    """
    Z: (batch, latent_dim)
    Returns (loss, cache) where loss is differentiable w.r.t. Z.
    """
    rng = rng or np.random
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


# ============================================================================
# Encoder
# ============================================================================

class WorldEncoder:
    """3-layer MLP encoder with RMSNorm + GELU, maps obs -> latent."""

    def __init__(self, obs_dim: int, latent_dim: int = 64, hidden_dim: int = 128,
                 rng: Optional[np.random.RandomState] = None):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        rng = rng or np.random.RandomState(0)

        # He initialization
        s1 = np.sqrt(2.0 / obs_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        s3 = np.sqrt(2.0 / hidden_dim)

        self.params = {
            "W1": rng.randn(obs_dim, hidden_dim) * s1,
            "b1": np.zeros(hidden_dim),
            "g1": np.ones(hidden_dim),
            "W2": rng.randn(hidden_dim, hidden_dim) * s2,
            "b2": np.zeros(hidden_dim),
            "g2": np.ones(hidden_dim),
            "W3": rng.randn(hidden_dim, latent_dim) * s3,
            "b3": np.zeros(latent_dim),
            "g_out": np.ones(latent_dim),
        }

    def forward(self, obs: np.ndarray):
        """obs: (B, obs_dim) -> z: (B, latent_dim). Returns (z, cache)."""
        p = self.params
        h1, c_lin1 = linear_forward(obs, p["W1"], p["b1"])
        n1, c_n1 = rms_norm_forward(h1, p["g1"])
        a1, c_g1 = gelu_forward(n1)

        h2, c_lin2 = linear_forward(a1, p["W2"], p["b2"])
        n2, c_n2 = rms_norm_forward(h2, p["g2"])
        a2, c_g2 = gelu_forward(n2)

        h3, c_lin3 = linear_forward(a2, p["W3"], p["b3"])
        z, c_nout = rms_norm_forward(h3, p["g_out"])

        cache = (c_lin1, c_n1, c_g1, c_lin2, c_n2, c_g2, c_lin3, c_nout)
        return z, cache

    def backward(self, dz: np.ndarray, cache):
        c_lin1, c_n1, c_g1, c_lin2, c_n2, c_g2, c_lin3, c_nout = cache
        grads = {}

        dh3, grads["g_out"] = rms_norm_backward(dz, c_nout)
        da2, grads["W3"], grads["b3"] = linear_backward(dh3, c_lin3)

        dn2 = gelu_backward(da2, c_g2)
        dh2, grads["g2"] = rms_norm_backward(dn2, c_n2)
        da1, grads["W2"], grads["b2"] = linear_backward(dh2, c_lin2)

        dn1 = gelu_backward(da1, c_g1)
        dh1, grads["g1"] = rms_norm_backward(dn1, c_n1)
        dobs, grads["W1"], grads["b1"] = linear_backward(dh1, c_lin1)

        return dobs, grads


# ============================================================================
# Predictor with AdaLN action conditioning
# ============================================================================

class WorldPredictor:
    """
    2-layer MLP predictor with AdaLN action conditioning.

    Architecture per layer:
        h = h @ W
        h = AdaLN(h, action) = rms_norm(h, gamma) * (1 + scale(a)) + shift(a)
        h = GELU(h)

    Action-conditioning weights (W_ada*_scale/shift) are zero-initialized
    so that at init, scale=0, shift=0 -> AdaLN is identity layer norm.
    This is the DiT/Peebles-Xie 2022 initialization trick.

    Output is z + delta_z (residual connection — replaces the hardcoded
    0.8/0.2 mix from v0.1, which was non-standard).
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 8,
                 hidden_dim: int = 128, rng: Optional[np.random.RandomState] = None):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        rng = rng or np.random.RandomState(1)

        s_l = np.sqrt(2.0 / latent_dim)
        s_h = np.sqrt(2.0 / hidden_dim)

        self.params = {
            "W_lat":  rng.randn(latent_dim, hidden_dim) * s_l,
            "b_lat":  np.zeros(hidden_dim),
            "g1":     np.ones(hidden_dim),
            # AdaLN block 1 — zero init for identity at start
            "W_a1_s": np.zeros((action_dim, hidden_dim)),
            "W_a1_b": np.zeros((action_dim, hidden_dim)),

            "W_hid":  rng.randn(hidden_dim, hidden_dim) * s_h,
            "b_hid":  np.zeros(hidden_dim),
            "g2":     np.ones(hidden_dim),
            "W_a2_s": np.zeros((action_dim, hidden_dim)),
            "W_a2_b": np.zeros((action_dim, hidden_dim)),

            "W_out":  rng.randn(hidden_dim, latent_dim) * s_h,
            "b_out":  np.zeros(latent_dim),
            "g_out":  np.ones(latent_dim),
        }

    def forward(self, z: np.ndarray, action: np.ndarray):
        """z: (B, latent), action: (B, action) -> z_next: (B, latent)."""
        p = self.params

        # Layer 1: Linear -> AdaLN -> GELU
        h1_lin, c_lin1 = linear_forward(z, p["W_lat"], p["b_lat"])
        h1_ada, c_ada1 = adaln_forward(h1_lin, action, p["g1"], p["W_a1_s"], p["W_a1_b"])
        h1, c_g1 = gelu_forward(h1_ada)

        # Layer 2
        h2_lin, c_lin2 = linear_forward(h1, p["W_hid"], p["b_hid"])
        h2_ada, c_ada2 = adaln_forward(h2_lin, action, p["g2"], p["W_a2_s"], p["W_a2_b"])
        h2, c_g2 = gelu_forward(h2_ada)

        # Output
        h3, c_lin3 = linear_forward(h2, p["W_out"], p["b_out"])
        delta, c_nout = rms_norm_forward(h3, p["g_out"])

        # Residual: z_next = z + delta (standard, replaces 0.8/0.2 hardcoded mix)
        z_next = z + delta

        cache = (z, c_lin1, c_ada1, c_g1,
                 c_lin2, c_ada2, c_g2,
                 c_lin3, c_nout)
        return z_next, cache

    def backward(self, dz_next: np.ndarray, cache):
        p = self.params
        (z, c_lin1, c_ada1, c_g1,
         c_lin2, c_ada2, c_g2,
         c_lin3, c_nout) = cache

        grads = {}
        # Residual: z_next = z + delta
        dz_residual = dz_next.copy()  # part flowing back through z
        ddelta = dz_next

        # Output layer
        dh3, grads["g_out"] = rms_norm_backward(ddelta, c_nout)
        dh2, grads["W_out"], grads["b_out"] = linear_backward(dh3, c_lin3)

        # Layer 2 backward
        dh2_ada = gelu_backward(dh2, c_g2)
        dh2_lin, da2, grads["g2"], grads["W_a2_s"], grads["W_a2_b"] = adaln_backward(
            dh2_ada, c_ada2, p["W_a2_s"], p["W_a2_b"])
        dh1, grads["W_hid"], grads["b_hid"] = linear_backward(dh2_lin, c_lin2)

        # Layer 1 backward
        dh1_ada = gelu_backward(dh1, c_g1)
        dh1_lin, da1, grads["g1"], grads["W_a1_s"], grads["W_a1_b"] = adaln_backward(
            dh1_ada, c_ada1, p["W_a1_s"], p["W_a1_b"])
        dz_main, grads["W_lat"], grads["b_lat"] = linear_backward(dh1_lin, c_lin1)

        # Total gradients into z and action
        dz_total = dz_main + dz_residual
        daction = da1 + da2  # action used in both AdaLN blocks
        return dz_total, daction, grads


# ============================================================================
# CEM Planner (no gradients needed — uses predictor in forward mode only)
# ============================================================================

class CEMPlanner:
    """Cross-Entropy Method action sequence search in latent space."""

    def __init__(self, predictor: WorldPredictor, action_dim: int,
                 horizon: int = 5, n_samples: int = 64,
                 n_elites: int = 10, n_iterations: int = 8,
                 rng: Optional[np.random.RandomState] = None):
        self.predictor = predictor
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.rng = rng or np.random.RandomState(42)

    def plan(self, z_current: np.ndarray, z_goal: np.ndarray,
             action_bounds: tuple = (-1.0, 1.0)) -> np.ndarray:
        H, A = self.horizon, self.action_dim
        mu = np.zeros((H, A))
        sigma = np.ones((H, A)) * 0.5

        # Ensure z has batch dim for predictor
        z0 = z_current[None] if z_current.ndim == 1 else z_current
        zg = z_goal[None] if z_goal.ndim == 1 else z_goal

        for _ in range(self.n_iterations):
            noise = self.rng.randn(self.n_samples, H, A)
            actions = mu[None] + sigma[None] * noise
            actions = np.clip(actions, action_bounds[0], action_bounds[1])

            # Vectorized rollout: batch all samples at once
            z_batch = np.broadcast_to(z0, (self.n_samples, z0.shape[-1])).copy()
            for t in range(H):
                z_batch, _ = self.predictor.forward(z_batch, actions[:, t, :])
            costs = np.sum((z_batch - zg)**2, axis=-1)

            elite_idx = np.argsort(costs)[:self.n_elites]
            elites = actions[elite_idx]
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 0.01

        return mu


# ============================================================================
# Full JEPA World Model
# ============================================================================

class JEPAWorldModel:
    """
    Complete JEPA world model.

    Training loop:
        L = ||predictor(encoder(x_t), a_t) - encoder(x_{t+1})||^2
            + lambda_reg * SIGReg(encoder(x_t))

    All gradients flow through both pred_loss and SIGReg into both encoder
    and predictor weights, including AdaLN parameters.
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64,
                 hidden_dim: Optional[int] = None,
                 lr: float = 1e-3, lambda_reg: float = 0.01,
                 sigreg_projections: int = 16,
                 cem_horizon: int = 2, cem_samples: int = 12,
                 cem_elites: int = 4, cem_iterations: int = 2,
                 seed: int = 0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        if hidden_dim is None:
            hidden_dim = latent_dim * 2
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.sigreg_projections = sigreg_projections

        rng = np.random.RandomState(seed)
        self.encoder = WorldEncoder(obs_dim, latent_dim, hidden_dim, rng=rng)
        self.predictor = WorldPredictor(latent_dim, action_dim, hidden_dim, rng=rng)
        self.planner = CEMPlanner(self.predictor, action_dim,
                                   horizon=cem_horizon, n_samples=cem_samples,
                                   n_elites=cem_elites, n_iterations=cem_iterations,
                                   rng=rng)

        # One Adam per module so optimizer state stays grouped
        self.opt_enc = Adam(self.encoder.params, lr=lr)
        self.opt_pred = Adam(self.predictor.params, lr=lr)

        self.experience_buffer: list = []
        self.max_buffer_size = 5000
        self.train_steps = 0
        self._sigreg_rng = rng  # use deterministic RNG for SIGReg projections
        # Dedicated RNG for training-batch sampling — avoids coupling to
        # global np.random state, which made tests order-dependent.
        self._train_rng = np.random.RandomState(seed + 13)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """observation: (obs_dim,) or (B, obs_dim) -> latent."""
        single = (observation.ndim == 1)
        x = observation[None] if single else observation
        z, _ = self.encoder.forward(x)
        return z[0] if single else z

    def predict_next(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        single = (z.ndim == 1)
        z_in = z[None] if single else z
        a_in = action[None] if action.ndim == 1 else action
        z_next, _ = self.predictor.forward(z_in, a_in)
        return z_next[0] if single else z_next

    def plan_to_goal(self, current_obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
        z_cur = self.encode(current_obs)
        z_goal = self.encode(goal_obs)
        return self.planner.plan(z_cur, z_goal)

    def store_experience(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        self.experience_buffer.append((obs.copy(), action.copy(), next_obs.copy()))
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    # ------------------------------------------------------------------
    # Training step (full backprop, including SIGReg and AdaLN)
    # ------------------------------------------------------------------

    def train_step(self, batch_size: int = 32) -> dict:
        if len(self.experience_buffer) < batch_size:
            return {"pred_loss": 0.0, "reg_loss": 0.0, "total_loss": 0.0}

        idx = self._train_rng.choice(len(self.experience_buffer), batch_size, replace=False)
        obs = np.array([self.experience_buffer[i][0] for i in idx])
        act = np.array([self.experience_buffer[i][1] for i in idx])
        nxt = np.array([self.experience_buffer[i][2] for i in idx])

        # ---- Forward pass ----
        z, enc_cache = self.encoder.forward(obs)
        z_next_true, enc_cache_next = self.encoder.forward(nxt)
        z_next_pred, pred_cache = self.predictor.forward(z, act)

        # Prediction loss (MSE in latent space)
        diff = z_next_pred - z_next_true
        pred_loss = float(np.mean(diff**2))

        # SIGReg on z (anti-collapse)
        reg_loss, sig_cache = sigreg_forward(
            z, n_projections=self.sigreg_projections, rng=self._sigreg_rng)
        total_loss = pred_loss + self.lambda_reg * reg_loss

        # ---- Backward pass ----
        # MSE gradient: dL/dz_next_pred = 2/(B*D) * diff,  dL/dz_next_true = -2/(B*D) * diff
        B, D = diff.shape
        dz_next_pred = (2.0 / (B * D)) * diff
        dz_next_true = -(2.0 / (B * D)) * diff

        # Backprop through predictor: gives dz_input (into encoder for x_t)
        # and the predictor parameter grads
        dz_from_pred, _daction, pred_grads = self.predictor.backward(dz_next_pred, pred_cache)

        # SIGReg gradient on z (scaled by lambda_reg)
        dz_from_sigreg = self.lambda_reg * sigreg_backward(sig_cache)

        # Total gradient on z (current obs latent)
        dz_total = dz_from_pred + dz_from_sigreg

        # Backprop through encoder for current obs and next obs
        _, enc_grads_cur = self.encoder.backward(dz_total, enc_cache)
        _, enc_grads_next = self.encoder.backward(dz_next_true, enc_cache_next)

        # Sum encoder gradients (same encoder used twice in graph)
        enc_grads = {k: enc_grads_cur[k] + enc_grads_next[k] for k in enc_grads_cur}

        # Optional: gradient clipping for stability
        max_norm = 5.0
        for grads in (enc_grads, pred_grads):
            total_sq = sum(np.sum(g**2) for g in grads.values())
            n = np.sqrt(total_sq) + 1e-8
            if n > max_norm:
                scale = max_norm / n
                for k in grads:
                    grads[k] *= scale

        # ---- Adam update ----
        self.opt_enc.step(self.encoder.params, enc_grads)
        self.opt_pred.step(self.predictor.params, pred_grads)

        self.train_steps += 1
        return {
            "pred_loss": pred_loss,
            "reg_loss": float(reg_loss),
            "total_loss": float(total_loss),
        }

    # ------------------------------------------------------------------
    # Diagnostic metrics (LeWM + Qu et al. 2026)
    # ------------------------------------------------------------------

    def compute_temporal_straightness(self, recent_n: int = 20) -> float:
        """
        Temporal latent path straightness (Maes et al. 2026, Section 5.1).
        Score 1.0 = perfectly straight latent trajectory; ~0 = erratic.

        Note: only meaningful if recent_n experiences are temporally
        consecutive (as written, this is the case if store_experience
        is called once per env step from the same agent).
        """
        if len(self.experience_buffer) < recent_n:
            return 0.0
        recent = self.experience_buffer[-recent_n:]
        zs = np.array([self.encode(exp[0]) for exp in recent])
        steps = np.linalg.norm(np.diff(zs, axis=0), axis=1)
        total_path = float(steps.sum())
        direct = float(np.linalg.norm(zs[-1] - zs[0]))
        return float(np.clip(direct / (total_path + 1e-8), 0.0, 1.0))

    def probe_physical_understanding(self, obs_indices: tuple = (32, 34, 36),
                                     obs_scales: tuple = (5.0, 1.0, 1.0)) -> dict:
        """
        Linear probe for physical structure (Qu et al. 2026, arXiv:2603.13227).

        obs_indices/obs_scales let the caller specify which observation slots
        encode physical parameters and how to denormalize them. Defaults match
        the simulation's observation layout: [32]=temperature/5, [34]=tension,
        [36]=resource scarcity. Override for other layouts.
        """
        if len(self.experience_buffer) < 50:
            return {"physical_r2": 0.0, "n_samples": 0}

        recent = self.experience_buffer[-100:]
        Z, Y = [], []
        for obs, _, _ in recent:
            if max(obs_indices) >= len(obs):
                return {"physical_r2": 0.0, "n_samples": 0,
                        "error": "obs_indices out of range"}
            Z.append(self.encode(obs))
            Y.append([obs[i] * s for i, s in zip(obs_indices, obs_scales)])

        Z = np.array(Z)
        Y = np.array(Y)

        # Linear probe with bias term
        Z_aug = np.hstack([Z, np.ones((len(Z), 1))])
        try:
            W_probe, *_ = np.linalg.lstsq(Z_aug, Y, rcond=None)
            Y_pred = Z_aug @ W_probe
            ss_res = np.sum((Y - Y_pred)**2)
            ss_tot = np.sum((Y - Y.mean(axis=0))**2) + 1e-8
            r2 = float(1.0 - ss_res / ss_tot)
        except np.linalg.LinAlgError:
            r2 = 0.0

        return {"physical_r2": round(max(0.0, r2), 4), "n_samples": len(Z)}

    def get_world_understanding(self) -> dict:
        base = {
            "train_steps": self.train_steps,
            "buffer_size": len(self.experience_buffer),
            "latent_dim": self.latent_dim,
            "model_maturity": min(1.0, self.train_steps / 500.0),
        }
        if self.train_steps > 0 and self.train_steps % 50 == 0:
            base["temporal_straightness"] = self.compute_temporal_straightness()
            base["physical_understanding"] = self.probe_physical_understanding()
        return base
