"""Great-circle neighbour search (finding B10, option a).

World.get_nearby_agents indexes unit-sphere xyz and queries the chord
equivalent of a great-circle radius in degree-equivalents (1 deg = 111 km),
the same metric as World._distance_deg.
"""
import numpy as np
import pytest

from oikoumene.agents.agents import Agent
from oikoumene.core.world import World


@pytest.fixture(scope="module")
def world() -> World:
    w = World(seed=1)
    w.logger.config.enabled = False
    return w


def _place(world: World, coords: list[tuple[float, float]]) -> list[Agent]:
    world.agents = [Agent(lat, lng, rng=world.rng) for lat, lng in coords]
    world._rebuild_spatial_grid()
    return world.agents


def _ids(agents: list[Agent]) -> set[int]:
    return {a.id for a in agents}


def test_60n_east_west_uses_true_distance(world: World) -> None:
    """At 60 N, 8 deg of longitude is ~444 km (~4 deg-equivalents), not 8."""
    a, b = _place(world, [(60.0, 10.0), (60.0, 18.0)])
    assert world._distance_deg(60.0, 10.0, 60.0, 18.0) == pytest.approx(4.0, abs=0.05)
    assert _ids(world.get_nearby_agents(60.0, 10.0, 5.0)) == {a.id, b.id}
    assert _ids(world.get_nearby_agents(60.0, 10.0, 3.0)) == {a.id}


def test_60n_north_south_unchanged(world: World) -> None:
    """Along a meridian degrees are true degree-equivalents at any latitude."""
    a, b = _place(world, [(60.0, 10.0), (64.0, 10.0)])
    assert _ids(world.get_nearby_agents(60.0, 10.0, 4.1)) == {a.id, b.id}
    assert _ids(world.get_nearby_agents(60.0, 10.0, 3.9)) == {a.id}


def test_across_dateline(world: World) -> None:
    """179 E and 179 W are ~2 deg-equivalents apart, not 358."""
    a, b = _place(world, [(0.0, 179.0), (0.0, -179.0)])
    assert _ids(world.get_nearby_agents(0.0, 179.0, 3.0)) == {a.id, b.id}
    assert _ids(world.get_nearby_agents(0.0, -179.5, 1.0)) == {b.id}
    a, b = _place(world, [(65.0, 178.0), (65.0, -178.0)])  # Bering Strait latitude
    assert _ids(world.get_nearby_agents(65.0, 178.0, 3.0)) == {a.id, b.id}


def test_matches_brute_force_great_circle(world: World) -> None:
    rng = np.random.RandomState(0)
    coords = list(zip(rng.uniform(-60, 75, 400), rng.uniform(-180, 180, 400)))
    agents = _place(world, coords)
    for lat, lng in coords[:50]:
        for r in (3.0, 5.0, 10.0):
            expected = {a.id for a in agents
                        if world._distance_deg(lat, lng, a.lat, a.lng) <= r - 1e-9}
            got = _ids(world.get_nearby_agents(lat, lng, r))
            near_boundary = {a.id for a in agents
                             if abs(world._distance_deg(lat, lng, a.lat, a.lng) - r) < 1e-6}
            assert expected <= got and got - expected <= near_boundary


def test_results_in_agent_list_order(world: World) -> None:
    agents = _place(world, [(0.0, 0.0), (0.5, 0.5), (-0.5, 0.2), (0.1, -0.3)])
    assert [a.id for a in world.get_nearby_agents(0.0, 0.0, 2.0)] == [a.id for a in agents]
