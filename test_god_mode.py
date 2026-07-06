"""
Tests for god_mode world interventions.

Focus: a drought must reduce regen while active and restore the pre-drought
regen rates on expiry (to floating-point tolerance). Drought is now modeled as
a persistent multiplicative factor that the era's climate driver (paleo
ice-age / modern macro bridge) composes with, so it is no longer silently
overwritten mid-life; revert removes that factor (float division), which is
exact to ~1e-12 rather than bit-identical.
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

    # Tick past expiry; drought is applied once (not per tick) and reverted.
    for _ in range(4):
        gm.update(world)

    after = world.resources.food_regen.copy()
    assert np.allclose(after, before, rtol=1e-12, atol=1e-12), (
        "regen not restored to the pre-drought baseline after expiry"
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
