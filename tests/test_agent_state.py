"""Tests and benchmarks for AgentStateArrays."""

import time
import numpy as np
from oikoumene.agents.agent_state import AgentStateArrays, N_TRAITS, N_SKILLS


def test_add_kill_reuse():
    s = AgentStateArrays(100)
    traits = np.random.rand(N_TRAITS).astype(np.float32)
    skills = np.random.rand(N_SKILLS).astype(np.float32)

    # Add 50 agents
    indices = []
    for i in range(50):
        idx = s.add_agent(i * 0.5, i * 1.0, traits, skills, energy=80)
        indices.append(idx)
    assert s.n_alive == 50
    assert s.count == 50

    # Kill 20
    for i in range(20):
        s.kill_agent(indices[i])
    s.invalidate_caches()
    assert s.n_alive == 30

    # Add 10 more — should reuse dead slots
    new_indices = []
    for i in range(10):
        idx = s.add_agent(0, 0, traits, skills)
        new_indices.append(idx)
    s.invalidate_caches()
    assert s.n_alive == 40
    # Reused slots should be < 50
    assert all(idx < 50 for idx in new_indices)
    print("  PASS: add/kill/reuse")


def test_kdtree():
    s = AgentStateArrays(500)
    traits = np.random.rand(N_TRAITS).astype(np.float32)
    skills = np.random.rand(N_SKILLS).astype(np.float32)

    # Add 200 agents scattered worldwide
    rng = np.random.RandomState(42)
    for _ in range(200):
        s.add_agent(rng.uniform(-50, 60), rng.uniform(-170, 170), traits, skills)

    s.rebuild_kdtree()

    # Query neighbors around (0, 0) within 10 degrees
    nearby = s.query_nearby(0, 0, 10.0)

    # Brute-force check
    idx = s.get_alive_indices()
    brute = []
    for i in idx:
        d = np.sqrt((s.lat[i])**2 + (s.lng[i])**2)
        if d <= 10.0:
            brute.append(i)
    assert set(nearby) == set(brute), f"KDTree: {len(nearby)} vs brute: {len(brute)}"
    print(f"  PASS: KDTree matches brute-force ({len(nearby)} neighbors)")


def test_batch_metabolism():
    """Compare batch metabolism with scalar loop — must match."""
    s = AgentStateArrays(200)
    traits = np.random.rand(N_TRAITS).astype(np.float32)
    skills = np.random.rand(N_SKILLS).astype(np.float32)

    for i in range(100):
        s.add_agent(i, i, traits, skills, energy=80 + i * 0.1)
    s.vlat[:100] = np.random.uniform(-0.05, 0.05, 100).astype(np.float32)
    s.vlng[:100] = np.random.uniform(-0.05, 0.05, 100).astype(np.float32)
    s.age[:100] = np.random.randint(0, 1000, 100).astype(np.int32)

    # Save state for scalar comparison
    energy_before = s.energy[:100].copy()
    age_before = s.age[:100].copy()

    # Scalar loop version
    scalar_energy = energy_before.copy()
    scalar_age = age_before.copy()
    for i in range(100):
        scalar_age[i] += 1
        base = 0.15 + scalar_age[i] / 8000.0
        speed = np.sqrt(s.vlat[i]**2 + s.vlng[i]**2)
        scalar_energy[i] -= base + speed * 0.5

    # Batch version
    s.batch_metabolism()

    diff = np.max(np.abs(s.energy[:100] - scalar_energy))
    assert diff < 1e-4, f"Metabolism mismatch: max diff = {diff}"
    assert np.all(s.age[:100] == scalar_age)
    print(f"  PASS: batch metabolism matches scalar (max diff: {diff:.2e})")


def test_batch_physics():
    """Test vectorized physics with landmask."""
    s = AgentStateArrays(100)
    traits = np.random.rand(N_TRAITS).astype(np.float32)
    skills = np.random.rand(N_SKILLS).astype(np.float32)

    # Add agents on land and near ocean
    for i in range(50):
        s.add_agent(40 + i * 0.5, 10 + i * 0.5, traits, skills, energy=80)
    s.vlat[:50] = 0.05
    s.vlng[:50] = 0.03

    # Simple landmask: everything above lat 0 is land
    def mock_landmask(lats, lngs):
        return lats > 0

    s.batch_apply_physics(mock_landmask, era_speed=1.0)

    # All agents should have moved (they started above 0)
    assert np.all(s.lat[:50] > 40), "Agents should have moved north"
    print("  PASS: batch physics moves agents correctly")


def benchmark(n_agents: int):
    """Benchmark core operations at given agent count."""
    s = AgentStateArrays(max(n_agents + 100, 4096))
    traits = np.random.rand(N_TRAITS).astype(np.float32)
    skills = np.random.rand(N_SKILLS).astype(np.float32)
    rng = np.random.RandomState(42)

    for _ in range(n_agents):
        s.add_agent(rng.uniform(-50, 65), rng.uniform(-170, 170), traits, skills,
                    energy=80 + rng.random() * 20)
    s.vlat[:n_agents] = rng.uniform(-0.05, 0.05, n_agents).astype(np.float32)
    s.vlng[:n_agents] = rng.uniform(-0.05, 0.05, n_agents).astype(np.float32)
    s.age[:n_agents] = rng.randint(0, 500, n_agents).astype(np.int32)

    def landmask(lats, lngs):
        return np.ones(len(lats), dtype=bool)  # All land for benchmark

    # Warmup
    s.invalidate_caches()
    s.batch_metabolism()
    s.rebuild_kdtree()

    # Benchmark
    N_ITERS = 50
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        s.invalidate_caches()
        s.batch_metabolism()
        s.batch_death_check()
        s.rebuild_kdtree()
        s.batch_wander()
        s.batch_apply_physics(landmask)
        neighbor_lists = s.query_all_neighbors(3.0)
        s.batch_separation(neighbor_lists)
    t1 = time.perf_counter()

    per_tick_ms = (t1 - t0) / N_ITERS * 1000
    return per_tick_ms


if __name__ == "__main__":
    print("=" * 60)
    print("AGENT STATE TESTS")
    print("=" * 60)

    test_add_kill_reuse()
    test_kdtree()
    test_batch_metabolism()
    test_batch_physics()

    print("\n" + "=" * 60)
    print("BENCHMARKS (50 iterations each)")
    print("=" * 60)
    for n in [500, 1000, 1500, 2000]:
        ms = benchmark(n)
        tps = 1000.0 / ms
        print(f"  {n:5d} agents: {ms:6.1f} ms/tick → {tps:5.1f} ticks/sec")

    print("\nDone!")
