"""
JEPA World Model — pure NumPy implementation with hand-written backprop.

Inspired by:
- LeCun (2022). A Path Towards Autonomous Machine Intelligence.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). LeWorldModel.
- Qu et al. (2026). Representation Learning for Spatiotemporal Physical Systems.

Key components:
- WorldEncoder: 3-layer MLP with RMSNorm + GELU, maps observations to latents
- WorldPredictor: 2-layer MLP with AdaLN action conditioning
- CEMPlanner: cross-entropy method for action selection in latent space
- JEPAWorldModel: orchestrates training with Adam optimizer

Layer primitives live in oikoumene.worldmodel.layers: forward/backward passes
for linear, GELU, RMSNorm, AdaLN and SIGReg (differentiable moments-based proxy
for the Cramer-Wold gaussianity test), and the Adam optimizer.

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

from oikoumene.worldmodel.layers import (
    Adam,
    adaln_backward, adaln_forward,
    gelu_backward, gelu_forward,
    linear_backward, linear_forward,
    rms_norm_backward, rms_norm_forward,
    sigreg_backward, sigreg_forward,
)


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


def heldout_probe_r2(Z: np.ndarray, Y: np.ndarray, test_fraction: float = 0.2) -> dict:
    """Linear-probe R^2 on held-out samples.

    Deterministic temporal split with no RNG: the last ``test_fraction`` of the
    (chronologically ordered) samples is the test set. A temporal split also
    avoids leakage between adjacent, autocorrelated experience samples. The
    returned R^2 is not clipped and can be negative (worse than predicting the
    test mean).
    """
    n_test = int(round(len(Z) * test_fraction))
    n_train = len(Z) - n_test
    if n_test < 2 or n_train < 2:
        return {"physical_r2_heldout": 0.0, "n_train": n_train, "n_test": n_test}
    Z_aug = np.hstack([Z, np.ones((len(Z), 1))])
    try:
        W_probe, *_ = np.linalg.lstsq(Z_aug[:n_train], Y[:n_train], rcond=None)
    except np.linalg.LinAlgError:
        return {"physical_r2_heldout": 0.0, "n_train": n_train, "n_test": n_test}
    Y_test = Y[n_train:]
    ss_res = np.sum((Y_test - Z_aug[n_train:] @ W_probe) ** 2)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2) + 1e-8
    return {"physical_r2_heldout": round(float(1.0 - ss_res / ss_tot), 4),
            "n_train": n_train, "n_test": n_test}


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

        # "physical_r2" is in-sample (fit and scored on the same samples).
        result = {"physical_r2": round(max(0.0, r2), 4), "n_samples": len(Z)}
        result.update(heldout_probe_r2(Z, Y))
        return result

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
