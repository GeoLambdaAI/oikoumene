"""
Tests for god_mode world interventions.

Focus: a drought must reduce regen while active and restore the *exact*
pre-drought regen rates on expiry — the old implementation re-applied the
reduction every tick and reverted with an approximate inverse, leaving
permanent multiplicative drift.
"""
import numpy as np

from world import World


def test_drought_reduces_then_restores_exactly():
    world = World(seed=1)
    gm = world.god_mode
    gm.config.enabled = True

    before = world.resources.food_regen.copy()
    gm.trigger_drought(world, lat=0.0, lng=20.0, radius_deg=10.0,
                       severity=0.8, duration_ticks=3)

    during = world.resources.food_regen.copy()
    assert (during < before - 1e-12).any(), "drought did not reduce any regen cells"

    # Tick past expiry; drought is applied once (not per tick) and reverted exactly.
    for _ in range(4):
        gm.update(world)

    after = world.resources.food_regen.copy()
    assert np.array_equal(after, before), (
        "regen not restored to the exact pre-drought baseline after expiry"
    )


def test_drought_not_recompounded_each_tick():
    """While active, the reduction must not compound tick over tick."""
    world = World(seed=2)
    gm = world.god_mode
    gm.config.enabled = True

    gm.trigger_drought(world, lat=0.0, lng=20.0, radius_deg=10.0,
                       severity=0.6, duration_ticks=5)
    snapshot = world.resources.food_regen.copy()

    # Advance a few ticks without expiring; reduced regen should be stable.
    for _ in range(3):
        gm.update(world)

    assert np.array_equal(world.resources.food_regen, snapshot), (
        "drought reduction compounded across ticks instead of staying constant"
    )
