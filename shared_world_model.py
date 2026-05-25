"""
Shared JEPA World Model — single model, vectorized batch operations.

ALL agents share one encoder + predictor. Individual differences come from
different observations, not different parameters. The PHYSICS is shared.

This module is a thin wrapper around JEPAWorldModel from world_model.py.
It adds vectorized batch primitives — most importantly plan_batch, which
runs CEM planning for N agents in parallel using a single matmul-rich path
through the shared predictor.

Compatibility notes (v0.2 rewrite):
- Training is delegated to JEPAWorldModel.train_step, which uses analytic
  backpropagation with Adam — the previous v0.1 finite-difference path is
  removed (it did not actually train the model).
- All AdaLN action-conditioning weights are now updated, and SIGReg
  gradients flow into the loss. See world_model.py for details.
- Encoder/predictor parameters now live in `self.encoder.params` (a dict)
  rather than as direct attributes (no `encoder.W1`). If you previously
  accessed weights directly, switch to `self.encoder.params['W1']`.

Architecture: LeWorldModel (Maes et al. 2026, arXiv:2603.19312).
"""

import numpy as np
from typing import Optional

from world_model import JEPAWorldModel


class SharedWorldModel:
    """
    Single JEPA model shared across all agents.

    Single-agent inference path is delegated; the value-add of this class
    is `plan_batch` and `encode_batch` — vectorized over N agents.
    """

    def __init__(self, obs_dim: int = 40, action_dim: int = 8,
                 latent_dim: int = 24,
                 lr: float = 1e-3, lambda_reg: float = 0.01,
                 sigreg_projections: int = 15,
                 cem_horizon: int = 2, cem_samples: int = 12,
                 cem_elites: int = 4, cem_iterations: int = 2,
                 max_buffer_size: int = 20000,
                 seed: int = 0,
                 backend: str = "numpy",
                 device: str = "auto", num_threads: int = 1,
                 sigreg_mode: str = "moments", sigreg_nodes: int = 32,
                 predictor_dropout: float = 0.0,
                 target_stop_gradient: bool = False):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.backend = backend

        # Backbone: full JEPA world model. Provides encoder, predictor,
        # CEM planner, training step, metrics. The torch backend is opt-in
        # and reproduces the numpy backend's architecture/hyperparameters at
        # its defaults (see world_model_torch.py).
        if backend == "numpy":
            self._jepa = JEPAWorldModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=latent_dim * 2,
                lr=lr,
                lambda_reg=lambda_reg,
                sigreg_projections=sigreg_projections,
                cem_horizon=cem_horizon,
                cem_samples=cem_samples,
                cem_elites=cem_elites,
                cem_iterations=cem_iterations,
                seed=seed,
            )
        elif backend == "torch":
            from world_model_torch import TorchJEPAWorldModel  # lazy: torch optional
            self._jepa = TorchJEPAWorldModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=latent_dim * 2,
                lr=lr,
                lambda_reg=lambda_reg,
                sigreg_projections=sigreg_projections,
                cem_horizon=cem_horizon,
                cem_samples=cem_samples,
                cem_elites=cem_elites,
                cem_iterations=cem_iterations,
                seed=seed,
                device=device,
                num_threads=num_threads,
                sigreg_mode=sigreg_mode,
                sigreg_nodes=sigreg_nodes,
                predictor_dropout=predictor_dropout,
                target_stop_gradient=target_stop_gradient,
            )
        else:
            raise ValueError(f"unknown backend: {backend!r} (expected 'numpy' or 'torch')")

        # Override buffer size (shared model needs more capacity)
        self._jepa.max_buffer_size = max_buffer_size

        # Dedicated RNG for batch CEM (reproducibility, doesn't leak global state)
        self._batch_rng = np.random.RandomState(seed + 7)

    # ------------------------------------------------------------------
    # Pass-through attribute access (compatibility with previous API)
    # ------------------------------------------------------------------

    @property
    def encoder(self):
        return self._jepa.encoder

    @property
    def predictor(self):
        return self._jepa.predictor

    @property
    def planner(self):
        """Per-agent CEM planner (compatibility shim)."""
        return self._jepa.planner

    @property
    def experience_buffer(self):
        return self._jepa.experience_buffer

    @property
    def train_steps(self):
        return self._jepa.train_steps

    @property
    def max_buffer_size(self):
        return self._jepa.max_buffer_size

    @property
    def lambda_reg(self):
        return self._jepa.lambda_reg

    # ------------------------------------------------------------------
    # Single-agent operations (delegated to JEPAWorldModel)
    # ------------------------------------------------------------------

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Single or batch encode. (D,) -> (latent,) or (N, D) -> (N, latent)."""
        return self._jepa.encode(observation)

    def predict_next(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._jepa.predict_next(z, action)

    def store_experience(self, obs, action, next_obs):
        self._jepa.store_experience(obs, action, next_obs)

    def train_step(self, batch_size: int = 64) -> dict:
        """Train using analytic backprop. Returns loss dict."""
        return self._jepa.train_step(batch_size)

    def get_world_understanding(self) -> dict:
        return self._jepa.get_world_understanding()

    def get_understanding(self) -> dict:
        """Alias for get_world_understanding (backward compat)."""
        return self._jepa.get_world_understanding()

    # ------------------------------------------------------------------
    # Vectorized batch operations (the value-add of this class)
    # ------------------------------------------------------------------

    def encode_batch(self, observations: np.ndarray) -> np.ndarray:
        """Batch encode: (N, obs_dim) -> (N, latent_dim)."""
        return self._jepa.encode(observations)

    def predict_batch(self, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Batch predict: (N, latent_dim), (N, action_dim) -> (N, latent_dim)."""
        return self._jepa.predict_next(z, actions)

    def plan_batch(self, z_current: np.ndarray, z_goal: np.ndarray,
                   n_samples: int = 12, n_elites: int = 4,
                   n_iterations: int = 2, horizon: int = 2,
                   action_bounds: tuple = (-1.0, 1.0)) -> np.ndarray:
        """
        Vectorized CEM planning for N agents simultaneously.

        Replaces N separate CEM loops with one batched rollout: at each CEM
        iteration we forward N*n_samples trajectories of length `horizon`
        through the shared predictor in single matmuls.

        Args:
            z_current: (N, latent_dim) — each agent's current latent state
            z_goal:    (N, latent_dim) — each agent's goal latent
            n_samples, n_elites, n_iterations, horizon: CEM hyperparameters
            action_bounds: per-dimension clipping range

        Returns:
            (N, action_dim) — first action of best plan for each agent.
        """
        N = z_current.shape[0]
        if N == 0:
            return np.zeros((0, self.action_dim), dtype=np.float32)

        A = self.action_dim
        S = n_samples
        H = horizon
        K = n_elites

        mu = np.zeros((N, H, A), dtype=np.float64)
        sigma = np.ones((N, H, A), dtype=np.float64) * 0.5

        z_current = z_current.astype(np.float64)
        z_goal = z_goal.astype(np.float64)

        for _ in range(n_iterations):
            # Sample (N, S, H, A) candidate action sequences
            noise = self._batch_rng.randn(N, S, H, A)
            actions = mu[:, None, :, :] + sigma[:, None, :, :] * noise
            np.clip(actions, action_bounds[0], action_bounds[1], out=actions)

            # Initial latents tiled S times: (N*S, latent_dim)
            z = np.broadcast_to(z_current[:, None, :],
                                (N, S, self.latent_dim)).reshape(N * S, -1).copy()

            # Roll out H steps through the predictor in one batched call per step
            for t in range(H):
                a_t = actions[:, :, t, :].reshape(N * S, A)
                z = self._jepa.predict_next(z, a_t)

            # Cost: squared distance to per-agent goal
            z_final = z.reshape(N, S, self.latent_dim)
            costs = np.sum((z_final - z_goal[:, None, :]) ** 2, axis=-1)  # (N, S)

            # Per-agent elite selection (K best per agent)
            elite_idx = np.argpartition(costs, K, axis=1)[:, :K]  # (N, K)
            batch_idx = np.arange(N)[:, None].repeat(K, axis=1)
            elite_actions = actions[batch_idx, elite_idx]  # (N, K, H, A)

            # Refit Gaussian per agent
            mu = elite_actions.mean(axis=1)
            sigma = elite_actions.std(axis=1) + 0.01

        # Return first action of mean elite plan per agent
        return mu[:, 0, :].astype(np.float32)

    def store_experience_batch(self, obs: np.ndarray, actions: np.ndarray,
                                next_obs: np.ndarray, cap: int = 50):
        """
        Store experiences from N agents in one call.

        Args:
            obs:      (N, obs_dim)
            actions:  (N, action_dim)
            next_obs: (N, obs_dim)
            cap: max number of agents to store per tick (perf safeguard)
        """
        buf = self._jepa.experience_buffer
        max_size = self._jepa.max_buffer_size
        n = min(len(obs), cap)
        for i in range(n):
            buf.append((obs[i].copy(), actions[i].copy(), next_obs[i].copy()))

        # Keep buffer below max_size — drop oldest in O(1) blocks
        if len(buf) > max_size:
            overflow = len(buf) - max_size
            del buf[:overflow]
