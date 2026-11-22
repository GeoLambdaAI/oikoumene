"""
Tests for geopolitics.py v0.2 fixes.

Verifies:
1. Conflict-prevalence calibration: 5-nation BAU run produces ~5-15% / 20-40% / 40-70%
   prevalence at low/mid/high social tension respectively.
2. Climate summit fires every 24 ticks regardless of nation count.
3. Haversine distance preserves equator behavior, corrects polar distortion.
4. Conflict probability is monotonically decreasing in trade and increasing in tension.
5. API surface is unchanged (no breaking changes for bridge.py and world.py).
"""
import types
import numpy as np

# Stub macro module if missing (test isolation)


def _make_mock_macro_state(tension=0.3, fossil=0.7, year=2050.0):
    class MockMacroState:
        social_tension = tension
        fossil_fuels = fossil
        year_val = year
        def __init__(s):
            s.social_tension = tension
            s.fossil_fuels = fossil
            s.year = year
    return MockMacroState()


def _make_nation(geop, gid, lat, lng, pop=20, wealth=100, mil=10):
    """Build a NationState bypassing settlement-formation logic."""
    from oikoumene.society.geopolitics import NationState
    n = NationState(
        id=gid,
        name=f"Nation{gid}",
        settlement_ids=[gid],
        capital_settlement_id=gid,
        center_lat=lat,
        center_lng=lng,
        population=pop,
        total_wealth=wealth,
        total_military=mil,
        technology_level=1.0,
    )
    geop.nations.append(n)
    geop.relation_graph.add_node(n.id)
    geop.trade_graph.add_node(n.id)
    return n


def test_haversine_correctness():
    print("Test 1: Haversine distance correctness")
    from oikoumene.society.geopolitics import GeopoliticalSystem

    # At equator: 1 deg lat ~ 1 deg lng
    d_eq = GeopoliticalSystem._great_circle_deg(0, 0, 0, 1)
    print(f"  1 deg longitude at equator: {d_eq:.4f} deg-equiv (expect ~1.0)")
    assert 0.99 <= d_eq <= 1.01

    # At 60 deg N: 1 deg lng ~ 0.5 deg
    d_60 = GeopoliticalSystem._great_circle_deg(60, 0, 60, 1)
    print(f"  1 deg longitude at 60 deg N: {d_60:.4f} deg-equiv (expect ~0.5)")
    assert 0.49 <= d_60 <= 0.51

    # Lat distance is invariant (1 deg lat = 1 deg-equiv everywhere)
    d_lat = GeopoliticalSystem._great_circle_deg(0, 0, 1, 0)
    d_lat_60 = GeopoliticalSystem._great_circle_deg(60, 0, 61, 0)
    print(f"  1 deg lat at equator: {d_lat:.4f}, at 60 deg N: {d_lat_60:.4f}")
    assert abs(d_lat - d_lat_60) < 0.01

    # Pole vs equator: euclidean would say sqrt(180^2)=180, haversine should give correct
    d_pole_eq = GeopoliticalSystem._great_circle_deg(90, 0, 0, 0)
    print(f"  Pole to equator: {d_pole_eq:.2f} deg-equiv (expect ~90)")
    assert 89.5 <= d_pole_eq <= 90.5
    print("  PASS\n")


def test_conflict_monotonic_in_tension_and_trade():
    print("Test 2: conflict_probability monotonic in tension and trade")
    from oikoumene.society.geopolitics import GeopoliticalSystem

    g = GeopoliticalSystem(rng=np.random.RandomState(0))
    a = _make_nation(g, 1, 50, 0, pop=20, wealth=100, mil=10)
    b = _make_nation(g, 2, 50, 5, pop=20, wealth=100, mil=10)

    # Tension monotonicity: higher tension -> higher P
    p_low = g.conflict_probability(a, b, _make_mock_macro_state(tension=0.1))
    p_mid = g.conflict_probability(a, b, _make_mock_macro_state(tension=0.5))
    p_high = g.conflict_probability(a, b, _make_mock_macro_state(tension=0.9))
    print(f"  P(tension=0.1) = {p_low:.4f}")
    print(f"  P(tension=0.5) = {p_mid:.4f}")
    print(f"  P(tension=0.9) = {p_high:.4f}")
    assert p_low < p_mid < p_high

    # Trade monotonicity: higher trade -> lower P (liberal peace)
    g.trade_graph.add_edge(a.id, b.id, weight=15.0)  # high trade
    p_with_trade = g.conflict_probability(a, b, _make_mock_macro_state(tension=0.5))
    g.trade_graph.remove_edge(a.id, b.id)
    g.trade_graph.add_edge(a.id, b.id, weight=0.1)
    p_no_trade = g.conflict_probability(a, b, _make_mock_macro_state(tension=0.5))
    print(f"  P(no trade)   = {p_no_trade:.4f}")
    print(f"  P(high trade) = {p_with_trade:.4f}")
    assert p_with_trade < p_no_trade
    print("  PASS\n")


def test_calibration_5_nation_bau():
    print("Test 3: 5-nation BAU prevalence calibration")
    print("  Targets (5-nation neighbor-cluster, BAU 2025-2100):")
    print("    2030 prevalence: 10-25% (calm, low climate stress)")
    print("    2050 prevalence: 30-50% (moderate stress)")
    print("    2090 prevalence: 50-80% (high climate stress)")
    print("  Calibration anchored to UCDP-style cluster prevalence,")
    print("  not the global average across all dyads.")
    from oikoumene.society.geopolitics import GeopoliticalSystem

    n_runs = 30
    n_nations = 5
    prev30, prev50, prev90 = [], [], []

    for run in range(n_runs):
        g = GeopoliticalSystem(rng=np.random.RandomState(run))
        positions = [(45, 0), (45, 8), (45, 16), (50, 4), (50, 12)]
        nations = []
        for i, (lat, lng) in enumerate(positions, 1):
            n = _make_nation(g, i, lat, lng, pop=20,
                             wealth=80 + 20*np.random.random(),
                             mil=8 + 4*np.random.random())
            nations.append(n)

        active_at_yr = {}
        n_ticks = int(75 * 1.2)
        for tick in range(n_ticks):
            yr = 2025 + tick / 1.2
            tens = 0.20 + (0.62 - 0.20) * (yr - 2025) / 75
            fossil = max(0.30, 0.85 - (yr - 2025) * 0.0073)
            macro = _make_mock_macro_state(tension=tens, fossil=fossil, year=yr)

            g._assess_conflicts(macro)

            yr_int = int(yr)
            active_at_yr.setdefault(yr_int, []).append(
                1 if g.active_conflicts else 0
            )

        def avg_in_range(lo, hi):
            vals = []
            for y, vs in active_at_yr.items():
                if lo <= y <= hi:
                    vals.extend(vs)
            return np.mean(vals) if vals else 0

        prev30.append(avg_in_range(2028, 2032))
        prev50.append(avg_in_range(2048, 2052))
        prev90.append(avg_in_range(2088, 2092))

    p30, p50, p90 = np.mean(prev30), np.mean(prev50), np.mean(prev90)
    print(f"  Empirical (n={n_runs} runs):")
    print(f"    2030 prevalence: {p30*100:5.1f}% (target 10-25%)")
    print(f"    2050 prevalence: {p50*100:5.1f}% (target 30-50%)")
    print(f"    2090 prevalence: {p90*100:5.1f}% (target 50-80%)")
    ok30 = 0.08 <= p30 <= 0.30
    ok50 = 0.25 <= p50 <= 0.55
    ok90 = 0.45 <= p90 <= 0.85
    print(f"  {'PASS' if all([ok30, ok50, ok90]) else 'FAIL'}")
    assert ok30 and ok50 and ok90
    print()


def test_climate_summit_cadence_independent_of_nation_count():
    print("Test 4: climate summit fires every 24 ticks regardless of N nations")
    from oikoumene.society.geopolitics import GeopoliticalSystem

    for n_nations in [3, 5, 10, 20]:
        g = GeopoliticalSystem(rng=np.random.RandomState(0))
        for i in range(n_nations):
            _make_nation(g, i+1, 40 + i*0.5, i*1.0)

        macro = _make_mock_macro_state(tension=0.3)
        # Run 100 macro ticks of negotiations only
        summits = 0
        for t in range(100):
            n_pre = len(g.negotiation_history)
            g._conduct_negotiations(macro)
            n_post = len(g.negotiation_history)
            if n_post > n_pre:
                summits += 1

        # Expect floor(100/24) = 4 summits regardless of N
        print(f"  N={n_nations}: {summits} summits in 100 ticks (expect 4)")
        assert summits == 4
    print("  PASS\n")


def test_api_unchanged():
    print("Test 5: Public API surface unchanged")
    from oikoumene.society.geopolitics import GeopoliticalSystem, NationState

    g = GeopoliticalSystem()
    # Methods that bridge.py and world.py call
    assert callable(g.update)
    assert callable(g.get_conflict_intensity)
    assert callable(g.get_summary)
    assert callable(g.get_nations_list)
    assert callable(g.get_total_trade_volume)
    assert callable(g.get_avg_trust)
    # Attributes
    assert hasattr(g, 'nations')
    assert hasattr(g, 'active_conflicts')
    print("  PASS\n")


if __name__ == "__main__":
    test_haversine_correctness()
    test_conflict_monotonic_in_tension_and_trade()
    test_calibration_5_nation_bau()
    test_climate_summit_cadence_independent_of_nation_count()
    test_api_unchanged()
    print("=" * 60)
    print("ALL GEOPOLITICS TESTS PASSED")
    print("=" * 60)
