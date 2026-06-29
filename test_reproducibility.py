"""
Determinism / reproducibility regression tests.

A fixed ``World(seed=...)`` must produce byte-identical trajectories across runs.
This guards the property that all stochastic agent behaviour (trait generation,
mutation, movement noise, partner selection, reproduction, research, names) is
driven by the world's seeded ``RandomState`` and never by the unseeded global
``np.random``.
"""
import numpy as np

from world import World


def _fingerprint(seed: int, ticks: int) -> list:
    """Run a seeded world for ``ticks`` steps and return a hashable state digest."""
    world = World(seed=seed)
    for _ in range(ticks):
        world.step()
    agents = sorted(world.agents, key=lambda a: a.id)
    return [
        (
            a.id,
            round(a.lat, 9),
            round(a.lng, 9),
            round(a.energy, 6),
            round(a.wealth, 6),
            a.name,
        )
        for a in agents
    ]


def test_same_seed_is_deterministic():
    """Two worlds with the same seed produce identical agent trajectories."""
    assert _fingerprint(42, 40) == _fingerprint(42, 40)


def test_different_seed_diverges():
    """Different seeds should (almost surely) produce different trajectories."""
    assert _fingerprint(42, 40) != _fingerprint(7, 40)


def test_no_global_numpy_random_leak():
    """Seeding the global np.random must not change a seeded world's outcome.

    If any agent code still used the global np.random, perturbing global state
    between constructing the world and stepping it would alter the result.
    """
    np.random.seed(123)
    world = World(seed=42)
    np.random.seed(999)  # perturb global RNG mid-run; must have no effect
    for _ in range(40):
        world.step()
    digest_a = [(a.id, round(a.lat, 9), round(a.wealth, 6)) for a in sorted(world.agents, key=lambda a: a.id)]

    assert digest_a == [
        (a.id, round(a.lat, 9), round(a.wealth, 6))
        for a in sorted(_rebuild_world(42, 40).agents, key=lambda a: a.id)
    ]


def _rebuild_world(seed: int, ticks: int) -> World:
    world = World(seed=seed)
    for _ in range(ticks):
        world.step()
    return world
