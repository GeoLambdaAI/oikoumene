"""
JEPA World Model — PyTorch backend (opt-in alternative to world_model.py).

This is a faithful PyTorch re-implementation of the pure-NumPy JEPA model in
`world_model.py`. At its DEFAULT settings it reproduces that model's
architecture and hyperparameters exactly, so weights can be copied across
backends (see `load_numpy_params` / `export_numpy_params`) and outputs match
to float precision. The value of the torch backend is autograd (no
hand-written backward), GPU support, and easy extensibility.

Fidelity target (default config = world_model.py ground truth):
- Encoder: 3-layer MLP, Linear -> RMSNorm(eps=1e-6) -> GELU(tanh approx) x2,
  then Linear -> RMSNorm (no final GELU). He-normal init, bias 0, gamma 1.
- Predictor: Linear -> AdaLN(action) -> GELU x2, Linear -> RMSNorm, residual
  z_next = z + delta. AdaLN scale/shift are bias-free linears, zero-init (DiT).
- SIGReg (default "moments"): random unit-norm projections + skew^2 + kurt^2 +
  (sigma-1)^2 variance penalty. M=15, lambda=0.01 as deployed.
- Loss: MSE(z_next_pred, z_next_true) + lambda * SIGReg(z_t). The target
  encoder is NOT detached — both branches backprop (end-to-end). This matches
  both world_model.py AND the LeWorldModel paper's "no stop-gradient" design.
- Optimizer: two separate Adam (encoder, predictor), lr=1e-3. Per-module
  global-norm gradient clipping at 5.0.

Opt-in PAPER-ALIGNED toggles (off by default; do NOT change default behavior):
- sigreg_mode="epps_pulley": characteristic-function distance to N(0,1) along
  random projections, trapezoid-integrated over nodes in [0.2, 4] (Maes et al.
  2026, arXiv:2603.19312, SIGReg). Pair with sigreg_projections (paper M=1024)
  and lambda_reg=0.1 for paper-like regularization.
- predictor_dropout > 0: dropout after each predictor activation (paper uses
  10% in its transformer predictor).

Reference: LeCun (2022); Maes, Le Lidec, Scieur, LeCun, Balestriero (2026),
arXiv:2603.19312. The paper's ViT/transformer backbone targets *pixel*
sequences; this simulation's observation is a low-dim vector, so the MLP
backbone of world_model.py is the appropriate adaptation and is preserved.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Layer primitives
# ============================================================================

class RMSNorm(nn.Module):
    """y = gamma * x / sqrt(mean(x^2) + eps). Matches world_model.rms_norm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.gamma * x / torch.sqrt(ms + self.eps)


class AdaLN(nn.Module):
    """
    y = rms_norm(h, gamma) * (1 + scale) + shift,
    with scale = action @ W_scale, shift = action @ W_shift (no bias).

    scale/shift weights are zero-initialized so AdaLN starts as identity
    layer norm (DiT / Peebles-Xie 2022 trick), matching world_model.py.
    """

    def __init__(self, dim: int, action_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps)
        self.scale = nn.Linear(action_dim, dim, bias=False)
        self.shift = nn.Linear(action_dim, dim, bias=False)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        return h_norm * (1.0 + self.scale(action)) + self.shift(action)


# ============================================================================
# Encoder
# ============================================================================

class TorchWorldEncoder(nn.Module):
    """3-layer MLP encoder with RMSNorm + GELU(tanh). obs -> latent."""

    def __init__(self, obs_dim: int, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.lin1 = nn.Linear(obs_dim, hidden_dim)
        self.norm1 = RMSNorm(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, latent_dim)
        self.norm_out = RMSNorm(latent_dim)
        self.act = nn.GELU(approximate="tanh")
        self._he_init()

    def _he_init(self) -> None:
        for lin in (self.lin1, self.lin2, self.lin3):
            nn.init.normal_(lin.weight, std=(2.0 / lin.in_features) ** 0.5)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.lin1(x)))
        h = self.act(self.norm2(self.lin2(h)))
        return self.norm_out(self.lin3(h))


# ============================================================================
# Predictor with AdaLN action conditioning
# ============================================================================

class TorchWorldPredictor(nn.Module):
    """2-layer MLP predictor with AdaLN action conditioning + residual."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 8,
                 hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.lin_lat = nn.Linear(latent_dim, hidden_dim)
        self.adaln1 = AdaLN(hidden_dim, action_dim)
        self.lin_hid = nn.Linear(hidden_dim, hidden_dim)
        self.adaln2 = AdaLN(hidden_dim, action_dim)
        self.lin_out = nn.Linear(hidden_dim, latent_dim)
        self.norm_out = RMSNorm(latent_dim)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.lin_lat.weight, std=(2.0 / latent_dim) ** 0.5)
        nn.init.zeros_(self.lin_lat.bias)
        nn.init.normal_(self.lin_hid.weight, std=(2.0 / hidden_dim) ** 0.5)
        nn.init.zeros_(self.lin_hid.bias)
        nn.init.normal_(self.lin_out.weight, std=(2.0 / hidden_dim) ** 0.5)
        nn.init.zeros_(self.lin_out.bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.act(self.adaln1(self.lin_lat(z), action))
        h = self.drop(h)
        h = self.act(self.adaln2(self.lin_hid(h), action))
        h = self.drop(h)
        delta = self.norm_out(self.lin_out(h))
        return z + delta


# ============================================================================
# SIGReg variants (differentiable; autograd handles the backward pass)
# ============================================================================

def sigreg_moments(Z: torch.Tensor, n_projections: int,
                   generator: torch.Generator, eps: float = 1e-6) -> torch.Tensor:
    """
    Moments-based Gaussianity proxy along random unit-norm projections.
    Mirrors world_model.sigreg_forward: skew^2 + kurt^2 + variance penalty.
    """
    B, D = Z.shape
    U = torch.randn(D, n_projections, generator=generator,
                    device=Z.device, dtype=Z.dtype)
    U = U / (U.norm(dim=0, keepdim=True) + 1e-10)

    P = Z @ U
    mu = P.mean(dim=0, keepdim=True)
    Pc = P - mu
    var = (Pc ** 2).mean(dim=0, keepdim=True)
    sigma = torch.sqrt(var + eps)
    Pn = Pc / sigma

    skew = (Pn ** 3).mean(dim=0)
    kurt = (Pn ** 4).mean(dim=0) - 3.0
    var_pen = ((sigma - 1.0) ** 2).mean()
    return (skew ** 2).mean() + (kurt ** 2).mean() + var_pen


def sigreg_epps_pulley(Z: torch.Tensor, n_projections: int,
                       generator: torch.Generator, n_nodes: int = 32,
                       eps: float = 1e-6) -> torch.Tensor:
    """
    Opt-in paper-aligned SIGReg (Maes et al. 2026, arXiv:2603.19312).

    Characteristic-function distance between the standardized projected
    embeddings and N(0,1), integrated by trapezoid over t-nodes in [0.2, 4],
    averaged over random unit-norm projections (Cramer-Wold). This is the
    Epps-Pulley test statistic; the published weighting constant is not fully
    specified, so a uniform trapezoid weight on [0.2, 4] is used.
    """
    B, D = Z.shape
    U = torch.randn(D, n_projections, generator=generator,
                    device=Z.device, dtype=Z.dtype)
    U = U / (U.norm(dim=0, keepdim=True) + 1e-10)

    P = Z @ U
    mu = P.mean(dim=0, keepdim=True)
    sigma = torch.sqrt((P - mu).pow(2).mean(dim=0, keepdim=True) + eps)
    Pn = (P - mu) / sigma  # (B, K) standardized

    t = torch.linspace(0.2, 4.0, n_nodes, device=Z.device, dtype=Z.dtype)  # (T,)
    arg = Pn.unsqueeze(-1) * t.view(1, 1, -1)  # (B, K, T)
    re = torch.cos(arg).mean(dim=0)            # (K, T) Re of empirical CF
    im = torch.sin(arg).mean(dim=0)            # (K, T) Im of empirical CF
    target = torch.exp(-0.5 * t ** 2).view(1, -1)  # (1, T) CF of N(0,1)
    integrand = (re - target) ** 2 + im ** 2       # (K, T)
    stat = torch.trapz(integrand, t, dim=-1)       # (K,)
    return stat.mean()


# ============================================================================
# CEM Planner (forward-only; no gradients)
# ============================================================================

class TorchCEMPlanner:
    """Cross-Entropy Method action search in latent space (mirrors CEMPlanner)."""

    def __init__(self, model: "TorchJEPAWorldModel", action_dim: int,
                 horizon: int = 5, n_samples: int = 64,
                 n_elites: int = 10, n_iterations: int = 8,
                 generator: Optional[torch.Generator] = None):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.generator = generator

    def plan(self, z_current: np.ndarray, z_goal: np.ndarray,
             action_bounds: tuple = (-1.0, 1.0)) -> np.ndarray:
        H, A = self.horizon, self.action_dim
        device = self.model.device

        z0 = torch.as_tensor(np.asarray(z_current), dtype=torch.float32, device=device)
        zg = torch.as_tensor(np.asarray(z_goal), dtype=torch.float32, device=device)
        if z0.ndim == 1:
            z0 = z0[None]
        if zg.ndim == 1:
            zg = zg[None]

        mu = torch.zeros(H, A, device=device)
        sigma = torch.ones(H, A, device=device) * 0.5

        self.model.predictor.eval()
        with torch.no_grad():
            for _ in range(self.n_iterations):
                noise = torch.randn(self.n_samples, H, A, generator=self.generator,
                                    device=device)
                actions = (mu[None] + sigma[None] * noise).clamp(
                    action_bounds[0], action_bounds[1])

                z_batch = z0.expand(self.n_samples, z0.shape[-1]).clone()
                for t in range(H):
                    z_batch = self.model.predictor(z_batch, actions[:, t, :])
                costs = ((z_batch - zg) ** 2).sum(dim=-1)

                elite_idx = torch.argsort(costs)[:self.n_elites]
                elites = actions[elite_idx]
                mu = elites.mean(dim=0)
                sigma = elites.std(dim=0) + 0.01

        return mu.cpu().numpy()


# ============================================================================
# Full JEPA World Model (torch) — mirrors JEPAWorldModel public API
# ============================================================================

class TorchJEPAWorldModel:
    """PyTorch JEPA world model. Drop-in for JEPAWorldModel (numpy backend)."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64,
                 hidden_dim: Optional[int] = None,
                 lr: float = 1e-3, lambda_reg: float = 0.01,
                 sigreg_projections: int = 16,
                 cem_horizon: int = 2, cem_samples: int = 12,
                 cem_elites: int = 4, cem_iterations: int = 2,
                 seed: int = 0,
                 device: str = "auto", num_threads: int = 1,
                 sigreg_mode: str = "moments", sigreg_nodes: int = 32,
                 predictor_dropout: float = 0.0,
                 target_stop_gradient: bool = False):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        if hidden_dim is None:
            hidden_dim = latent_dim * 2
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.sigreg_projections = sigreg_projections
        self.sigreg_mode = sigreg_mode
        self.sigreg_nodes = sigreg_nodes
        self.target_stop_gradient = target_stop_gradient

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if num_threads is not None and self.device.type == "cpu":
            torch.set_num_threads(int(num_threads))

        # Reproducible parameter initialization
        torch.manual_seed(seed)
        self.encoder = TorchWorldEncoder(obs_dim, latent_dim, hidden_dim).to(self.device)
        self.predictor = TorchWorldPredictor(
            latent_dim, action_dim, hidden_dim, dropout=predictor_dropout).to(self.device)

        # Generators for stochastic ops (kept off the global RNG)
        self._sigreg_gen = torch.Generator(device=self.device)
        self._sigreg_gen.manual_seed(seed)
        self._cem_gen = torch.Generator(device=self.device)
        self._cem_gen.manual_seed(seed + 42)
        self._train_rng = np.random.RandomState(seed + 13)

        self.planner = TorchCEMPlanner(
            self, action_dim, horizon=cem_horizon, n_samples=cem_samples,
            n_elites=cem_elites, n_iterations=cem_iterations,
            generator=self._cem_gen)

        # Two optimizers so encoder/predictor state stays grouped (matches numpy)
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.opt_pred = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        self.experience_buffer: list = []
        self.max_buffer_size = 5000
        self.train_steps = 0

    # ------------------------------------------------------------------
    # SIGReg dispatch
    # ------------------------------------------------------------------

    def _sigreg(self, Z: torch.Tensor) -> torch.Tensor:
        if self.sigreg_mode == "epps_pulley":
            return sigreg_epps_pulley(Z, self.sigreg_projections,
                                      self._sigreg_gen, n_nodes=self.sigreg_nodes)
        if self.sigreg_mode == "moments":
            return sigreg_moments(Z, self.sigreg_projections, self._sigreg_gen)
        raise ValueError(f"unknown sigreg_mode: {self.sigreg_mode!r}")

    # ------------------------------------------------------------------
    # Inference API (numpy in / numpy out)
    # ------------------------------------------------------------------

    def encode(self, observation: np.ndarray) -> np.ndarray:
        single = (observation.ndim == 1)
        x = observation[None] if single else observation
        self.encoder.eval()
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
            z = self.encoder(xt).cpu().numpy()
        return z[0] if single else z

    def predict_next(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        single = (z.ndim == 1)
        z_in = z[None] if single else z
        a_in = action[None] if action.ndim == 1 else action
        self.predictor.eval()
        with torch.no_grad():
            zt = torch.as_tensor(np.asarray(z_in), dtype=torch.float32, device=self.device)
            at = torch.as_tensor(np.asarray(a_in), dtype=torch.float32, device=self.device)
            z_next = self.predictor(zt, at).cpu().numpy()
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
    # Training (autograd; no hand-written backward)
    # ------------------------------------------------------------------

    def train_step(self, batch_size: int = 32) -> dict:
        if len(self.experience_buffer) < batch_size:
            return {"pred_loss": 0.0, "reg_loss": 0.0, "total_loss": 0.0}

        idx = self._train_rng.choice(len(self.experience_buffer), batch_size, replace=False)
        obs = np.array([self.experience_buffer[i][0] for i in idx])
        act = np.array([self.experience_buffer[i][1] for i in idx])
        nxt = np.array([self.experience_buffer[i][2] for i in idx])

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        nxt_t = torch.as_tensor(nxt, dtype=torch.float32, device=self.device)

        self.encoder.train()
        self.predictor.train()

        z = self.encoder(obs_t)
        z_next_true = self.encoder(nxt_t)
        if self.target_stop_gradient:  # default False: end-to-end (paper + repo)
            z_next_true = z_next_true.detach()
        z_next_pred = self.predictor(z, act_t)

        pred_loss = F.mse_loss(z_next_pred, z_next_true)
        reg_loss = self._sigreg(z)
        total_loss = pred_loss + self.lambda_reg * reg_loss

        self.opt_enc.zero_grad(set_to_none=True)
        self.opt_pred.zero_grad(set_to_none=True)
        total_loss.backward()

        # Per-module global-norm gradient clipping at 5.0 (matches numpy)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 5.0)

        self.opt_enc.step()
        self.opt_pred.step()

        self.train_steps += 1
        return {
            "pred_loss": float(pred_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    # ------------------------------------------------------------------
    # Weight bridge (cross-backend validation / warm-start)
    # ------------------------------------------------------------------

    def load_numpy_params(self, np_model) -> None:
        """
        Copy weights from a numpy JEPAWorldModel (world_model.py) into this
        torch model so both produce matching encode/predict outputs.
        numpy stores Linear weight as (in, out) and computes x @ W; torch
        nn.Linear stores (out, in) and computes x @ W.T, hence the transposes.
        """
        ep = np_model.encoder.params
        pp = np_model.predictor.params

        def t(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            enc = self.encoder
            enc.lin1.weight.copy_(t(ep["W1"]).T); enc.lin1.bias.copy_(t(ep["b1"]))
            enc.norm1.gamma.copy_(t(ep["g1"]))
            enc.lin2.weight.copy_(t(ep["W2"]).T); enc.lin2.bias.copy_(t(ep["b2"]))
            enc.norm2.gamma.copy_(t(ep["g2"]))
            enc.lin3.weight.copy_(t(ep["W3"]).T); enc.lin3.bias.copy_(t(ep["b3"]))
            enc.norm_out.gamma.copy_(t(ep["g_out"]))

            pred = self.predictor
            pred.lin_lat.weight.copy_(t(pp["W_lat"]).T); pred.lin_lat.bias.copy_(t(pp["b_lat"]))
            pred.adaln1.norm.gamma.copy_(t(pp["g1"]))
            pred.adaln1.scale.weight.copy_(t(pp["W_a1_s"]).T)
            pred.adaln1.shift.weight.copy_(t(pp["W_a1_b"]).T)
            pred.lin_hid.weight.copy_(t(pp["W_hid"]).T); pred.lin_hid.bias.copy_(t(pp["b_hid"]))
            pred.adaln2.norm.gamma.copy_(t(pp["g2"]))
            pred.adaln2.scale.weight.copy_(t(pp["W_a2_s"]).T)
            pred.adaln2.shift.weight.copy_(t(pp["W_a2_b"]).T)
            pred.lin_out.weight.copy_(t(pp["W_out"]).T); pred.lin_out.bias.copy_(t(pp["b_out"]))
            pred.norm_out.gamma.copy_(t(pp["g_out"]))

    def export_numpy_params(self) -> dict:
        """Inverse of load_numpy_params: torch state -> numpy params dicts."""
        def n(p):
            return p.detach().cpu().numpy().astype(np.float64)

        enc, pred = self.encoder, self.predictor
        encoder_params = {
            "W1": n(enc.lin1.weight).T, "b1": n(enc.lin1.bias), "g1": n(enc.norm1.gamma),
            "W2": n(enc.lin2.weight).T, "b2": n(enc.lin2.bias), "g2": n(enc.norm2.gamma),
            "W3": n(enc.lin3.weight).T, "b3": n(enc.lin3.bias), "g_out": n(enc.norm_out.gamma),
        }
        predictor_params = {
            "W_lat": n(pred.lin_lat.weight).T, "b_lat": n(pred.lin_lat.bias),
            "g1": n(pred.adaln1.norm.gamma),
            "W_a1_s": n(pred.adaln1.scale.weight).T, "W_a1_b": n(pred.adaln1.shift.weight).T,
            "W_hid": n(pred.lin_hid.weight).T, "b_hid": n(pred.lin_hid.bias),
            "g2": n(pred.adaln2.norm.gamma),
            "W_a2_s": n(pred.adaln2.scale.weight).T, "W_a2_b": n(pred.adaln2.shift.weight).T,
            "W_out": n(pred.lin_out.weight).T, "b_out": n(pred.lin_out.bias),
            "g_out": n(pred.norm_out.gamma),
        }
        return {"encoder": encoder_params, "predictor": predictor_params}

    # ------------------------------------------------------------------
    # Diagnostics (identical math to world_model.py; operate on numpy)
    # ------------------------------------------------------------------

    def compute_temporal_straightness(self, recent_n: int = 20) -> float:
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
        Z_aug = np.hstack([Z, np.ones((len(Z), 1))])
        try:
            W_probe, *_ = np.linalg.lstsq(Z_aug, Y, rcond=None)
            Y_pred = Z_aug @ W_probe
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2) + 1e-8
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
            "backend": "torch",
            "device": str(self.device),
        }
        if self.train_steps > 0 and self.train_steps % 50 == 0:
            base["temporal_straightness"] = self.compute_temporal_straightness()
            base["physical_understanding"] = self.probe_physical_understanding()
        return base
