"""
Behavioural tests for the NumPy JEPA world model (world_model.JEPAWorldModel).

Covers the learning dynamics and inference API:
1. Prediction loss decreases when trained on predictable latent dynamics.
2. Action conditioning is *learned* (zero-init AdaLN ignores the action at
   init; after training the prediction depends on the action).
3. Anti-collapse: trained latents stay diverse (do not collapse to a point).
4. Linear probe for physical structure returns a well-formed R^2 in [0, 1].
5. CEM planner returns finite, in-bounds action sequences of the right shape.

Per-primitive backward correctness is covered separately in
test_world_model_gradcheck.py.
"""
import numpy as np

from world_model import JEPAWorldModel


def _train_on_linear_dynamics(model, obs_dim, action_dim, steps=800,
                              train_iters=400, seed=7):
    """Feed the model a predictable action-conditioned linear system."""
    rng = np.random.RandomState(seed)
    A = 0.95 * np.eye(obs_dim) + 0.02 * rng.randn(obs_dim, obs_dim)
    B = rng.randn(action_dim, obs_dim) * 0.3
    x = rng.randn(obs_dim) * 0.5
    for _ in range(steps):
        a = rng.uniform(-1, 1, action_dim)
        x_next = A @ x + B.T @ a + 0.02 * rng.randn(obs_dim)
        model.store_experience(x, a, x_next)
        x = x_next
    losses = [model.train_step(batch_size=64)["pred_loss"] for _ in range(train_iters)]
    return losses


def test_prediction_loss_decreases():
    print("\nTest: prediction loss decreases with training")
    m = JEPAWorldModel(obs_dim=20, action_dim=4, latent_dim=12,
                       lr=3e-3, lambda_reg=0.05, seed=11)
    losses = _train_on_linear_dynamics(m, 20, 4)
    early = float(np.mean(losses[:30]))
    late = float(np.mean(losses[-50:]))
    print(f"  early={early:.4f} late={late:.4f} ratio={early/late:.1f}x")
    assert late < 0.5 * early, f"loss did not decrease enough: {early} -> {late}"


def test_action_conditioning_is_learned():
    print("\nTest: action conditioning (zero-init identity -> learned)")
    m = JEPAWorldModel(obs_dim=20, action_dim=4, latent_dim=12,
                       lr=3e-3, lambda_reg=0.01, seed=5)
    obs = np.random.RandomState(0).randn(20)
    z = m.encode(obs)
    a1 = np.full(4, -0.9)
    a2 = np.full(4, 0.9)

    # At init the AdaLN scale/shift weights are zero, so the action has no
    # effect on the prediction (DiT-style identity init).
    pre = np.linalg.norm(m.predict_next(z, a1) - m.predict_next(z, a2))
    print(f"  pre-training action sensitivity:  {pre:.2e}")
    assert pre < 1e-12, "untrained AdaLN should ignore the action"

    _train_on_linear_dynamics(m, 20, 4)

    z = m.encode(obs)
    post = np.linalg.norm(m.predict_next(z, a1) - m.predict_next(z, a2))
    print(f"  post-training action sensitivity: {post:.4f}")
    assert post > 1e-3, "trained predictor should respond to the action"


def test_anti_collapse_latents_stay_diverse():
    print("\nTest: anti-collapse (latents stay diverse after training)")
    m = JEPAWorldModel(obs_dim=20, action_dim=4, latent_dim=12,
                       lr=3e-3, lambda_reg=0.1, seed=3)
    _train_on_linear_dynamics(m, 20, 4, steps=600, train_iters=300)

    rng = np.random.RandomState(123)
    Z = m.encode(rng.randn(50, 20))
    per_dim_std = float(np.std(Z, axis=0).mean())
    print(f"  mean per-dim latent std: {per_dim_std:.4f}")
    assert per_dim_std > 1e-2, "latents collapsed (no variation across inputs)"
    assert np.all(np.isfinite(Z))


def test_linear_probe_physical_understanding():
    print("\nTest: linear probe returns valid R^2")
    m = JEPAWorldModel(obs_dim=40, action_dim=8, latent_dim=24, seed=1)
    rng = np.random.RandomState(2)
    for _ in range(120):
        o = rng.randn(40); a = rng.uniform(-1, 1, 8); n = rng.randn(40)
        m.store_experience(o, a, n)

    probe = m.probe_physical_understanding()
    print(f"  probe: {probe}")
    assert "physical_r2" in probe and "n_samples" in probe
    assert probe["n_samples"] == 100
    assert 0.0 <= probe["physical_r2"] <= 1.0
    assert np.isfinite(probe["physical_r2"])


def test_cem_planner_output_valid():
    print("\nTest: CEM planner output validity")
    m = JEPAWorldModel(obs_dim=20, action_dim=4, latent_dim=12, seed=0)
    rng = np.random.RandomState(9)
    plan = m.plan_to_goal(rng.randn(20), rng.randn(20))
    print(f"  plan shape: {plan.shape}, range [{plan.min():.3f}, {plan.max():.3f}]")
    assert plan.shape == (m.planner.horizon, 4)
    assert np.all(np.isfinite(plan))
    assert plan.min() >= -1.0 and plan.max() <= 1.0


if __name__ == "__main__":
    test_prediction_loss_decreases()
    test_action_conditioning_is_learned()
    test_anti_collapse_latents_stay_diverse()
    test_linear_probe_physical_understanding()
    test_cem_planner_output_valid()
    print("\nALL JEPA MODEL TESTS PASSED")
