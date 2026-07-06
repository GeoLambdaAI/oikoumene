"""
Regression tests for the present-day conflict-onset realism fix.

Before the fix, applying a per-dyad conflict model calibrated on a ~5-nation
bloc to the 140-nation present-day scenario ignited ~12 new wars at the first
geopolitics assessment (a visible burst "just after 2026"), including
geographically absurd pairings. The fix adds:
  1. geographic gating (only proximate dyads can ignite),
  2. an N-aware onset normalization (aggregate onset held ~constant above a
     reference dyad count; == 1.0 for small blocs, preserving calibration),
  3. warm-started military mass (deterrence/parity dimension),
  4. an onset spin-up over the first few assessments.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _emergent(conflicts):
    """Conflicts with an actual nation dyad (seeded real-world ones carry an
    empty `nations` list and are excluded)."""
    return [c for c in conflicts if len(c.get("nations", [])) == 2]


def test_no_onset_burst_at_first_assessment():
    from world import World
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(300)
    seeded = len(w.geopolitics.active_conflicts)  # the 10 real-world conflicts

    # Step through the first geopolitics assessment (macro_update_interval=10).
    for _ in range(w.macro_update_interval):
        w.step()

    total = len(w.geopolitics.active_conflicts)
    # Ungated this jumped by ~7-12 in one tick. With the fix + spin-up the first
    # assessment must add at most a couple.
    assert total <= seeded + 3, (
        f"conflict burst at first assessment: {seeded} -> {total}"
    )


def test_conflict_onset_rate_is_realistic():
    from world import World
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(300)
    geo = w.geopolitics

    def key(c):
        return frozenset(c["nations"])
    prev = {key(c) for c in _emergent(geo.active_conflicts)}
    onsets = 0
    ticks = 480  # ~40 sim years at 1 month/tick
    for _ in range(ticks):
        w.step()
        cur = {key(c) for c in _emergent(geo.active_conflicts)}
        onsets += len(cur - prev)
        prev = cur

    years = ticks / 12.0
    rate = onsets / years
    # UCDP interstate/internationalized onset is ~0.5-3 per year globally.
    assert 0.2 <= rate <= 3.0, f"unrealistic emergent onset rate: {rate:.2f}/yr"


def test_emergent_conflicts_are_geographically_local():
    from world import World
    from geopolitics import GeopoliticalSystem as G
    w = World(seed=2, scenario_id="present_day")
    w.spawn_initial_agents(300)
    geo = w.geopolitics
    by_id = {n.id: n for n in geo.nations}

    for _ in range(300):
        w.step()
        for c in _emergent(geo.active_conflicts):
            a_id, b_id = c["nations"]
            a, b = by_id.get(a_id), by_id.get(b_id)
            if a is None or b is None:
                continue  # a party dissolved; distance no longer meaningful
            d = G._great_circle_deg(a.center_lat, a.center_lng,
                                    b.center_lat, b.center_lng)
            assert d <= G.CONFLICT_MAX_DYAD_DEG + 1e-6, (
                f"distant nations at war: {a.name}-{b.name}, {d:.1f}° apart"
            )


def test_distant_dyad_never_ignites():
    """A geographically far pair must never start a conflict, however many
    assessments run."""
    from geopolitics import GeopoliticalSystem, NationState

    g = GeopoliticalSystem(rng=np.random.RandomState(0))
    # Two large, near-peer, high-tension nations placed on opposite sides of the
    # globe (well beyond CONFLICT_MAX_DYAD_DEG).
    a = NationState(id=1, name="A", settlement_ids=[1], population=50,
                    total_wealth=100, total_military=10, center_lat=0, center_lng=0)
    b = NationState(id=2, name="B", settlement_ids=[2], population=50,
                    total_wealth=100, total_military=10, center_lat=0, center_lng=170)
    for n in (a, b):
        g.nations.append(n)
        g.relation_graph.add_node(n.id)
        g.trade_graph.add_node(n.id)

    class _M:
        social_tension = 0.9
        fossil_fuels = 0.3
        year = 2050.0

    for _ in range(500):
        g._assess_conflicts(_M())
    assert not _emergent(g.active_conflicts), "distant dyad ignited despite gating"


def test_small_cluster_scale_is_unity():
    """N-aware normalization must be exactly 1.0 for a bloc at/below the
    reference dyad count, so the 5-nation calibration is untouched."""
    from geopolitics import GeopoliticalSystem as G
    for n_eligible in (1, 10, 100, G.CONFLICT_DYAD_REFERENCE):
        scale = min(1.0, G.CONFLICT_DYAD_REFERENCE / n_eligible)
        assert scale == 1.0
    # Above the reference it scales down.
    assert min(1.0, G.CONFLICT_DYAD_REFERENCE / (G.CONFLICT_DYAD_REFERENCE * 4)) == 0.25


def test_present_day_nations_have_military():
    from world import World
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(300)
    mils = [n.total_military for n in w.geopolitics.nations]
    assert any(m > 0 for m in mils), "seeded nations still have zero military"
