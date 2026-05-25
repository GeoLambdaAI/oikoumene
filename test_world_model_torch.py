"""
Tests for the PyTorch JEPA backend (world_model_torch.py via SharedWorldModel).

Skips cleanly if torch is not installed. Verifies:
1. Single-agent API compatibility through backend="torch"
2. Batch outputs == sequential outputs (float32 tolerance)
3. plan_batch produces valid, agent-specific actions
4. Training reduces prediction loss on synthetic linear dynamics
5. Buffer cap holds
6. Weight cross-check: numpy weights copied into torch reproduce encode/predict
7. Paper-aligned Epps-Pulley SIGReg toggle trains without error
8. Device resolves to cpu/cuda and a CUDA round-trip works when available
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from shared_world_model import SharedWorldModel


def _torch_model(**kw):
    kw.setdefault("backend", "torch")
    kw.setdefault("device", "cpu")
    return SharedWorldModel(**kw)


def test_single_agent_compat_torch():
    m = _torch_model(obs_dim=20, action_dim=4, latent_dim=12, seed=0)

    obs = np.random.RandomState(0).randn(20)
    z = m.encode(obs)
    assert z.shape == (12,)

    a = np.random.RandomState(1).uniform(-1, 1, 4)
    z_next = m.predict_next(z, a)
    assert z_next.shape == (12,)

    m.store_experience(obs, a, obs + 0.1)
    assert len(m.experience_buffer) == 1


def test_batch_equals_sequential_torch():
    m = _torch_model(obs_dim=20, action_dim=4, latent_dim=12, seed=0)
    rng = np.random.RandomState(42)
    N = 8
    obs_batch = rng.randn(N, 20)
    a_batch = rng.uniform(-1, 1, (N, 4))

    Z_batch = m.encode_batch(obs_batch)
    Z_seq = np.array([m.encode(obs_batch[i]) for i in range(N)])
    assert np.max(np.abs(Z_batch - Z_seq)) < 1e-5

    Zn_batch = m.predict_batch(Z_batch, a_batch)
    Zn_seq = np.array([m.predict_next(Z_batch[i], a_batch[i]) for i in range(N)])
    assert np.max(np.abs(Zn_batch - Zn_seq)) < 1e-5


def test_plan_batch_torch():
    m = _torch_model(obs_dim=20, action_dim=4, latent_dim=12, seed=0)
    rng = np.random.RandomState(99)
    N = 5
    z_curr = rng.randn(N, 12)
    z_goal = rng.randn(N, 12)

    plan = m.plan_batch(z_curr, z_goal)
    assert plan.shape == (N, 4)
    assert np.all(np.isfinite(plan))
    assert plan.min() >= -1.0 and plan.max() <= 1.0

    diffs = [np.linalg.norm(plan[i] - plan[j])
             for i in range(N) for j in range(i + 1, N)]
    assert np.mean(diffs) > 1e-3

    empty_plan = m.plan_batch(np.zeros((0, 12)), np.zeros((0, 12)))
    assert empty_plan.shape == (0, 4)


def test_training_reduces_loss_torch():
    m = _torch_model(obs_dim=20, action_dim=4, latent_dim=12,
                     lr=3e-3, lambda_reg=0.05, seed=11)

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
    assert late < 0.6 * early, f"loss did not decrease enough: {early} -> {late}"


def test_buffer_cap_torch():
    m = _torch_model(obs_dim=10, action_dim=2, latent_dim=8,
                     max_buffer_size=100, seed=0)
    obs = np.random.randn(50, 10)
    act = np.random.randn(50, 2)
    nxt = np.random.randn(50, 10)
    for _ in range(5):
        m.store_experience_batch(obs, act, nxt, cap=50)
    assert len(m.experience_buffer) == 100


def test_weight_crosscheck_numpy_to_torch():
    """numpy weights copied into the torch model reproduce its outputs."""
    dims = dict(obs_dim=20, action_dim=4, latent_dim=12, seed=3)
    m_np = SharedWorldModel(backend="numpy", **dims)
    m_pt = _torch_model(**dims)

    m_pt._jepa.load_numpy_params(m_np._jepa)

    rng = np.random.RandomState(123)
    obs = rng.randn(16, 20)
    act = rng.uniform(-1, 1, (16, 4))

    z_np = m_np.encode_batch(obs)
    z_pt = m_pt.encode_batch(obs)
    assert np.max(np.abs(z_np - z_pt)) < 1e-4, "encoder mismatch after weight copy"

    zn_np = m_np.predict_batch(z_np, act)
    zn_pt = m_pt.predict_batch(z_np, act)
    assert np.max(np.abs(zn_np - zn_pt)) < 1e-4, "predictor mismatch after weight copy"


def test_export_roundtrip_numpy_params():
    """export_numpy_params is the inverse of load_numpy_params."""
    m_np = SharedWorldModel(backend="numpy", obs_dim=20, action_dim=4, latent_dim=12, seed=5)
    m_pt = _torch_model(obs_dim=20, action_dim=4, latent_dim=12, seed=5)
    m_pt._jepa.load_numpy_params(m_np._jepa)

    exported = m_pt._jepa.export_numpy_params()
    for k, v in m_np._jepa.encoder.params.items():
        assert np.max(np.abs(exported["encoder"][k] - v)) < 1e-5, f"encoder param {k}"
    for k, v in m_np._jepa.predictor.params.items():
        assert np.max(np.abs(exported["predictor"][k] - v)) < 1e-5, f"predictor param {k}"


def test_paper_toggle_epps_pulley_trains():
    """Opt-in Epps-Pulley SIGReg (paper-aligned) runs and reduces loss."""
    m = _torch_model(obs_dim=20, action_dim=4, latent_dim=12,
                     lr=3e-3, lambda_reg=0.1, seed=2,
                     sigreg_mode="epps_pulley", sigreg_projections=64)

    rng = np.random.RandomState(7)
    A_mat = 0.95 * np.eye(20) + 0.02 * rng.randn(20, 20)
    B_mat = rng.randn(4, 20) * 0.3
    x = rng.randn(20) * 0.5
    for _ in range(600):
        a = rng.uniform(-1, 1, 4)
        x_next = A_mat @ x + B_mat.T @ a + 0.02 * rng.randn(20)
        m.store_experience(x, a, x_next)
        x = x_next

    losses = []
    for _ in range(300):
        info = m.train_step(batch_size=64)
        assert np.isfinite(info["total_loss"])
        assert np.isfinite(info["reg_loss"])
        losses.append(info["pred_loss"])

    assert np.mean(losses[-50:]) < np.mean(losses[:30])


def test_device_resolution():
    m = _torch_model(obs_dim=10, action_dim=2, latent_dim=8, device="auto", seed=0)
    assert m._jepa.device.type in ("cpu", "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_roundtrip():
    m = _torch_model(obs_dim=10, action_dim=2, latent_dim=8, device="cuda", seed=0)
    assert m._jepa.device.type == "cuda"
    obs = np.random.RandomState(0).randn(4, 10)
    z = m.encode_batch(obs)
    assert z.shape == (4, 8) and np.all(np.isfinite(z))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
