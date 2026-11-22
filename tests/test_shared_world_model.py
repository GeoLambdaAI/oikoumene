"""
Tests for SharedWorldModel.

Verifies:
1. Single-agent compatibility (encode, predict_next, store_experience, train_step)
2. Batch operations produce same outputs as sequential calls
3. plan_batch produces finite, in-range, agent-specific actions
4. Training works through the wrapper (loss decreases)
5. Experience buffer cap holds
"""
import numpy as np
from oikoumene.worldmodel.shared_world_model import SharedWorldModel


def test_single_agent_compat():
    print("Test 1: Single-agent API compatibility")
    m = SharedWorldModel(obs_dim=20, action_dim=4, latent_dim=12, seed=0)

    obs = np.random.RandomState(0).randn(20)
    z = m.encode(obs)
    assert z.shape == (12,), f"encode single shape: {z.shape}"

    a = np.random.RandomState(1).uniform(-1, 1, 4)
    z_next = m.predict_next(z, a)
    assert z_next.shape == (12,), f"predict_next single shape: {z_next.shape}"

    m.store_experience(obs, a, obs + 0.1)
    assert len(m.experience_buffer) == 1

    print("  PASS")


def test_batch_equals_sequential():
    print("\nTest 2: Batch outputs == sequential outputs")
    m = SharedWorldModel(obs_dim=20, action_dim=4, latent_dim=12, seed=0)
    rng = np.random.RandomState(42)
    N = 8
    obs_batch = rng.randn(N, 20)
    a_batch = rng.uniform(-1, 1, (N, 4))

    # Batch encode
    Z_batch = m.encode_batch(obs_batch)
    Z_seq = np.array([m.encode(obs_batch[i]) for i in range(N)])
    err_enc = np.max(np.abs(Z_batch - Z_seq))
    print(f"  encode  max diff: {err_enc:.2e}")
    assert err_enc < 1e-10

    # Batch predict
    Zn_batch = m.predict_batch(Z_batch, a_batch)
    Zn_seq = np.array([m.predict_next(Z_batch[i], a_batch[i]) for i in range(N)])
    err_pred = np.max(np.abs(Zn_batch - Zn_seq))
    print(f"  predict max diff: {err_pred:.2e}")
    assert err_pred < 1e-10

    print("  PASS")


def test_plan_batch():
    print("\nTest 3: plan_batch produces valid, agent-specific actions")
    m = SharedWorldModel(obs_dim=20, action_dim=4, latent_dim=12, seed=0)
    rng = np.random.RandomState(99)
    N = 5
    z_curr = rng.randn(N, 12)
    z_goal = rng.randn(N, 12)

    plan = m.plan_batch(z_curr, z_goal)
    print(f"  plan shape: {plan.shape}")
    assert plan.shape == (N, 4)
    assert np.all(np.isfinite(plan))
    assert plan.min() >= -1.0 and plan.max() <= 1.0
    print(f"  plan range: [{plan.min():.3f}, {plan.max():.3f}]")

    # Different agents should get different plans (since they have different goals)
    diffs = []
    for i in range(N):
        for j in range(i+1, N):
            diffs.append(np.linalg.norm(plan[i] - plan[j]))
    mean_diff = np.mean(diffs)
    print(f"  Mean inter-agent plan distance: {mean_diff:.4f}")
    assert mean_diff > 1e-3, "Plans should differ across agents with different goals"

    # Empty input edge case
    empty_plan = m.plan_batch(np.zeros((0, 12)), np.zeros((0, 12)))
    assert empty_plan.shape == (0, 4)
    print("  Empty-input edge case OK")

    print("  PASS")


def test_training_through_wrapper():
    print("\nTest 4: Training through wrapper actually reduces loss")
    m = SharedWorldModel(obs_dim=20, action_dim=4, latent_dim=12,
                         lr=3e-3, lambda_reg=0.05, seed=11)

    # Synthetic predictable dynamics
    rng = np.random.RandomState(7)
    A_mat = 0.95 * np.eye(20) + 0.02 * rng.randn(20, 20)
    B_mat = rng.randn(4, 20) * 0.3
    x = rng.randn(20) * 0.5
    for _ in range(800):
        a = rng.uniform(-1, 1, 4)
        x_next = A_mat @ x + B_mat.T @ a + 0.02 * rng.randn(20)
        m.store_experience(x, a, x_next)
        x = x_next

    losses = []
    for _ in range(400):
        info = m.train_step(batch_size=64)
        losses.append(info["pred_loss"])

    early = np.mean(losses[:30])
    late = np.mean(losses[-50:])
    print(f"  Mean early pred_loss: {early:.4f}")
    print(f"  Mean late  pred_loss: {late:.4f}")
    print(f"  Reduction:            {early/late:.1f}x")
    assert late < 0.5 * early, f"loss did not decrease enough: {early} -> {late}"
    print("  PASS")


def test_buffer_cap():
    print("\nTest 5: Experience buffer respects cap")
    m = SharedWorldModel(obs_dim=10, action_dim=2, latent_dim=8,
                         max_buffer_size=100, seed=0)
    obs = np.random.randn(50, 10)
    act = np.random.randn(50, 2)
    nxt = np.random.randn(50, 10)
    for _ in range(5):
        m.store_experience_batch(obs, act, nxt, cap=50)
    print(f"  buffer size after 5x50 stores (cap=100): {len(m.experience_buffer)}")
    assert len(m.experience_buffer) == 100
    print("  PASS")


def test_batch_plan_against_serial_plan():
    print("\nTest 6: plan_batch behaves like multiple per-agent CEM calls")
    # Note: due to independent RNG draws between batch and per-agent,
    # we can't expect bitwise equality. But the planner should pick
    # actions that drive z toward z_goal, so cost(plan) < cost(zero-action).
    m = SharedWorldModel(obs_dim=20, action_dim=4, latent_dim=12, seed=0)
    rng = np.random.RandomState(13)
    # Train briefly so predictor is non-trivial
    A_mat = 0.95 * np.eye(20) + 0.02 * rng.randn(20, 20)
    B_mat = rng.randn(4, 20) * 0.3
    x = rng.randn(20) * 0.5
    for _ in range(400):
        a = rng.uniform(-1, 1, 4)
        x_next = A_mat @ x + B_mat.T @ a + 0.02 * rng.randn(20)
        m.store_experience(x, a, x_next)
        x = x_next
    for _ in range(200):
        m.train_step(batch_size=64)

    N = 6
    z_curr = m.encode_batch(rng.randn(N, 20))
    z_goal = m.encode_batch(rng.randn(N, 20))

    # Cost with zero action
    zero_action = np.zeros((N, 4))
    z_zero = m.predict_batch(z_curr, zero_action)
    cost_zero = np.sum((z_zero - z_goal)**2, axis=-1)

    # Cost with planned action
    plan = m.plan_batch(z_curr, z_goal, n_samples=24, n_iterations=4)
    z_plan = m.predict_batch(z_curr, plan)
    cost_plan = np.sum((z_plan - z_goal)**2, axis=-1)

    improved = (cost_plan < cost_zero).sum()
    print(f"  Agents with cost_plan < cost_zero: {improved}/{N}")
    print(f"  Mean cost_zero: {cost_zero.mean():.4f}, mean cost_plan: {cost_plan.mean():.4f}")
    assert improved >= int(N * 0.6), \
        f"plan_batch should beat zero-action for majority, got {improved}/{N}"
    print("  PASS")


if __name__ == "__main__":
    test_single_agent_compat()
    test_batch_equals_sequential()
    test_plan_batch()
    test_training_through_wrapper()
    test_buffer_cap()
    test_batch_plan_against_serial_plan()
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
