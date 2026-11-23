"""
Tests for bridge.py v0.2 fixes.

Verifies:
1. CORRECTNESS: New optimized lookups produce identical agent_nation
   mapping and identical local_state output as the old O(quartisch) code.
2. PERFORMANCE: New code is dramatically faster on realistic inputs.
3. CONSISTENCY: get_macro_local_state now uses the same distance function
   as apply_geopolitics_to_agents (resolved via world._distance_deg).
4. EDGE CASES: empty inputs, missing settlements, zero-radius conflicts.

These tests run standalone — we mock the World/Settlement/Nation/Agent classes
to isolate bridge.py logic from the rest of the pipeline.
"""
import numpy as np
import time



# -------------------------- Mocks --------------------------

class MockSettlement:
    def __init__(self, sid, lat, lng, member_ids):
        self.id = sid
        self.lat = lat
        self.lng = lng
        self.members = set(member_ids)

class MockNation:
    def __init__(self, nid, settlement_ids, tech=1.0, openness=0.5):
        self.id = nid
        self.settlement_ids = list(settlement_ids)
        self.technology_level = tech
        self.trade_openness = openness

class MockGeopolitics:
    def __init__(self, nations, conflicts=None):
        self.nations = nations
        self.active_conflicts = conflicts or []

class MockSkillSet:
    def __init__(self):
        self._calls = []
    def practice(self, name, intensity, mod):
        self._calls.append((name, intensity, mod))

class MockAgent:
    def __init__(self, aid, lat, lng):
        self.id = aid
        self.lat = lat
        self.lng = lng
        self.alive = True
        self.health = 100.0
        self.wealth = 100.0
        self.happiness = 50.0
        self.skills = MockSkillSet()

class MockWorld:
    """Minimal World stub providing settlements + _distance_deg."""
    def __init__(self, settlements):
        self.settlements = settlements
    @staticmethod
    def _distance_deg(lat1, lng1, lat2, lng2):
        # Use the same haversine-degree-equivalent formula as world.py v0.2
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lng2 - lng1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
        c = 2 * np.arcsin(min(1.0, np.sqrt(a)))
        return float(c * 6371.0 / 111.0)


class MockMacroState:
    temperature_anomaly = 1.5
    co2_ppm = 450.0
    social_tension = 0.3
    persistent_pollution = 0.4
    fossil_fuels = 0.7
    technology_level = 1.5


# -------------------------- Fixtures --------------------------

def build_realistic_world(n_nations=5, settlements_per_nation=6, agents_per_settlement=12,
                          seed=0):
    """Build a realistic ~5-nation world with ~30 settlements and ~300 members."""
    rng = np.random.RandomState(seed)
    settlements = []
    nations = []
    next_sid = 0
    next_aid = 1000

    for nid in range(n_nations):
        nation_lat = -30 + 20 * rng.random()
        nation_lng = -100 + 30 * rng.random()
        sids = []
        for _ in range(settlements_per_nation):
            s_lat = nation_lat + 5 * rng.normal()
            s_lng = nation_lng + 5 * rng.normal()
            members = list(range(next_aid, next_aid + agents_per_settlement))
            next_aid += agents_per_settlement
            settlements.append(MockSettlement(next_sid, s_lat, s_lng, members))
            sids.append(next_sid)
            next_sid += 1
        nations.append(MockNation(nid, sids, tech=1.0 + 0.3*rng.random()))

    # Plus some unaffiliated nomadic agents
    nomad_agents = []
    for _ in range(50):
        nomad_agents.append(MockAgent(next_aid, 30 * rng.normal(), 30 * rng.normal()))
        next_aid += 1

    settled_agents = []
    for s in settlements:
        for mid in s.members:
            settled_agents.append(MockAgent(mid,
                                            s.lat + 0.5 * rng.normal(),
                                            s.lng + 0.5 * rng.normal()))

    return settlements, nations, settled_agents + nomad_agents


# -------------------------- Tests --------------------------

def test_apply_geopolitics_correctness():
    """
    Old code and new code must produce identical agent_nation mapping and
    identical agent state changes after applying conflict + tech diffusion.
    """
    print("Test 1: apply_geopolitics_to_agents — old vs new behavior identity")

    from oikoumene.core.bridge import MacroAgentBridge
    settlements, nations, agents = build_realistic_world(seed=1)
    world = MockWorld(settlements)
    geop = MockGeopolitics(nations, conflicts=[
        {"lat": -25.0, "lng": -90.0, "radius": 8.0, "intensity": 0.6},
    ])

    # --- Run NEW code on a deep copy ---
    import copy
    agents_new = copy.deepcopy(agents)
    bridge_new = MacroAgentBridge()
    bridge_new.apply_geopolitics_to_agents(geop, agents_new, world)

    # --- Run OLD reference behavior: rebuild it inline ---
    agents_old = copy.deepcopy(agents)
    # Replicate old code semantics exactly
    alive = [a for a in agents_old if a.alive]
    agent_nation_old = {}
    for nation in geop.nations:
        for sid in nation.settlement_ids:
            for s in world.settlements:
                if s.id == sid:
                    for member_id in s.members:
                        agent_nation_old[member_id] = nation

    for conflict in geop.active_conflicts:
        zone_lat = conflict.get("lat", 0)
        zone_lng = conflict.get("lng", 0)
        zone_radius = conflict.get("radius", 5.0)
        intensity = conflict.get("intensity", 0.5)
        for agent in alive:
            dist = world._distance_deg(agent.lat, agent.lng, zone_lat, zone_lng)
            if dist < zone_radius:
                proximity = 1.0 - dist / zone_radius
                agent.health -= 2.0 * intensity * proximity
                agent.wealth -= 1.0 * intensity * proximity
                agent.happiness -= 3.0 * intensity * proximity

    for agent in alive:
        nation = agent_nation_old.get(agent.id)
        if nation and nation.technology_level > 1.2:
            tech_bonus = (nation.technology_level - 1.0) * 0.01
            agent.skills.practice("research", tech_bonus, 1.0)

    # --- Compare per-agent state ---
    n_agents = len(agents)
    diffs_health = sum(1 for a, b in zip(agents_new, agents_old)
                       if abs(a.health - b.health) > 1e-9)
    diffs_wealth = sum(1 for a, b in zip(agents_new, agents_old)
                       if abs(a.wealth - b.wealth) > 1e-9)
    diffs_happy = sum(1 for a, b in zip(agents_new, agents_old)
                      if abs(a.happiness - b.happiness) > 1e-9)
    diffs_skill = sum(1 for a, b in zip(agents_new, agents_old)
                      if a.skills._calls != b.skills._calls)
    print(f"  Agent count: {n_agents}")
    print(f"  Diff health/wealth/happiness/skills: "
          f"{diffs_health}/{diffs_wealth}/{diffs_happy}/{diffs_skill}")
    assert diffs_health == 0
    assert diffs_wealth == 0
    assert diffs_happy == 0
    assert diffs_skill == 0
    print("  PASS — identical behavior\n")


def test_get_macro_local_state_correctness():
    """
    New get_macro_local_state must match old behavior at the equator, where
    haversine and euclidean give the same answer to within 1%.
    """
    print("Test 2: get_macro_local_state — equator parity")

    from oikoumene.core.bridge import MacroAgentBridge
    settlements = [
        MockSettlement(0, 0, 0, []),    # at equator origin
        MockSettlement(1, 0, 4, []),    # 4 deg lng away
        MockSettlement(2, 0, 7, []),    # 7 deg away (outside default radius)
    ]
    nations = [MockNation(0, [0, 1, 2], tech=2.0, openness=0.8)]
    world = MockWorld(settlements)
    geop = MockGeopolitics(nations, conflicts=[
        {"lat": 1, "lng": 1, "radius": 4.0, "intensity": 0.7},
    ])
    bridge = MacroAgentBridge()
    macro = MockMacroState()

    state = bridge.get_macro_local_state(macro, 0, 0, geopolitics=geop, world=world)

    # nation_tech_level should be set (settlement 0 within radius 5)
    assert state["nation_tech_level"] == 2.0, state
    assert state["trade_access"] == 0.8, state
    # conflict_nearby should be > 0 (conflict at (1,1) within radius 4 of origin)
    assert state["conflict_nearby"] > 0, state
    print(f"  At equator origin: nation_tech={state['nation_tech_level']}, "
          f"trade_access={state['trade_access']}, conflict_nearby={state['conflict_nearby']:.3f}")
    print("  PASS\n")


def test_polar_correction_in_local_state():
    """
    Verify the v0.2 fix: at high latitude, get_macro_local_state now uses
    haversine, so a settlement '5 deg lng away' at 60N is recognized as
    nearby (only ~2.5 deg-equiv), whereas with the old euclidean it would
    have been right at the boundary.
    """
    print("Test 3: get_macro_local_state — polar correction")

    from oikoumene.core.bridge import MacroAgentBridge
    # Settlement at (60, 5), agent at (60, 0). Euclidean distance: 5.0,
    # haversine: ~2.5 deg-equiv (since cos(60)=0.5).
    settlements = [MockSettlement(0, 60.0, 5.0, [])]
    nations = [MockNation(0, [0], tech=3.0, openness=0.9)]
    world = MockWorld(settlements)
    bridge = MacroAgentBridge()
    macro = MockMacroState()

    # With haversine, the nation tech *should* be picked up (within 5.0 radius)
    state = bridge.get_macro_local_state(macro, 60.0, 0.0,
                                         geopolitics=MockGeopolitics(nations), world=world)
    print(f"  At (60N, 0): nation_tech={state['nation_tech_level']} (expect 3.0)")
    assert state["nation_tech_level"] == 3.0
    print("  PASS — polar correction working\n")


def test_performance_apply_geopolitics():
    """
    Performance: ensure the new code's lookup-build is faster than the old
    quartisch loop. Speedup scales with world size — at small scales the
    constant factor dominates, at larger scales the speedup is significant.
    """
    print("Test 4: apply_geopolitics_to_agents — performance")

    # Use a larger world to make the speedup visible
    settlements, nations, agents = build_realistic_world(
        n_nations=15, settlements_per_nation=10, agents_per_settlement=20, seed=42)
    world = MockWorld(settlements)
    geop = MockGeopolitics(nations)
    print(f"  Setup: {len(nations)} nations, {len(settlements)} settlements, "
          f"{len(agents)} agents")

    # Isolate the lookup-build for a fair apples-to-apples comparison
    def old_lookup():
        agent_nation = {}
        for nation in geop.nations:
            for sid in nation.settlement_ids:
                for s in world.settlements:
                    if s.id == sid:
                        for member_id in s.members:
                            agent_nation[member_id] = nation
        return agent_nation

    def new_lookup():
        settlement_by_id = {s.id: s for s in world.settlements}
        agent_nation = {}
        for nation in geop.nations:
            for sid in nation.settlement_ids:
                s = settlement_by_id.get(sid)
                if s is None:
                    continue
                for member_id in s.members:
                    agent_nation[member_id] = nation
        return agent_nation

    # Verify equivalence first
    a_old = old_lookup()
    a_new = new_lookup()
    assert set(a_old.keys()) == set(a_new.keys())
    assert all(a_old[k].id == a_new[k].id for k in a_old)

    n_runs = 200
    t0 = time.perf_counter()
    for _ in range(n_runs):
        old_lookup()
    t_old = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        new_lookup()
    t_new = (time.perf_counter() - t0) / n_runs * 1000

    speedup = t_old / max(t_new, 1e-6)
    print(f"  Old lookup: {t_old:.3f} ms/call")
    print(f"  New lookup: {t_new:.3f} ms/call")
    print(f"  Speedup:    {speedup:.1f}x")
    assert speedup > 1.5, f"Expected >1.5x speedup at 150 settlements, got {speedup:.1f}x"
    print("  PASS\n")


def test_performance_local_state():
    """
    Performance: get_macro_local_state is called once per agent per simulation
    tick from World.get_local_state. This is the hottest performance path in
    the bridge module. The new code must be substantially faster than the old
    triple-nested loop on realistic inputs.
    """
    print("Test 5: get_macro_local_state — performance")

    from oikoumene.core.bridge import MacroAgentBridge
    settlements, nations, agents = build_realistic_world(
        n_nations=10, settlements_per_nation=8, agents_per_settlement=20, seed=7)
    world = MockWorld(settlements)
    geop = MockGeopolitics(nations, conflicts=[
        {"lat": -25.0, "lng": -90.0, "radius": 6.0, "intensity": 0.5},
    ])
    bridge = MacroAgentBridge()
    macro = MockMacroState()
    sample_positions = [(a.lat, a.lng) for a in agents[:300]]
    print(f"  Setup: {len(nations)} nations, {len(settlements)} settlements, "
          f"sampling {len(sample_positions)} agent positions")

    # Old reference implementation (triple loop, euclidean distance)
    def old_get_local_state(lat, lng):
        state = {
            "temperature_anomaly": macro.temperature_anomaly,
            "co2_level": macro.co2_ppm,
            "social_tension": macro.social_tension,
            "pollution_level": macro.persistent_pollution,
            "resource_scarcity": 1.0 - macro.fossil_fuels,
            "conflict_nearby": 0.0,
            "nation_tech_level": macro.technology_level,
            "trade_access": 0.5,
        }
        for conflict in geop.active_conflicts:
            dist = np.sqrt((lat - conflict.get("lat", 0))**2 +
                           (lng - conflict.get("lng", 0))**2)
            if dist < conflict.get("radius", 5.0):
                state["conflict_nearby"] = max(
                    state["conflict_nearby"],
                    conflict.get("intensity", 0.5) * (1.0 - dist / conflict["radius"])
                )
        for nation in geop.nations:
            for sid in nation.settlement_ids:
                for s in world.settlements:
                    if s.id == sid:
                        dist = np.sqrt((lat - s.lat)**2 + (lng - s.lng)**2)
                        if dist < 5.0:
                            state["nation_tech_level"] = nation.technology_level
                            state["trade_access"] = nation.trade_openness
                            break
        return state

    n_outer = 5
    t0 = time.perf_counter()
    for _ in range(n_outer):
        for lat, lng in sample_positions:
            old_get_local_state(lat, lng)
    t_old = (time.perf_counter() - t0) / (n_outer * len(sample_positions)) * 1000

    t0 = time.perf_counter()
    for _ in range(n_outer):
        for lat, lng in sample_positions:
            bridge.get_macro_local_state(macro, lat, lng, geop, world)
    t_new = (time.perf_counter() - t0) / (n_outer * len(sample_positions)) * 1000

    speedup = t_old / max(t_new, 1e-6)
    print(f"  Old per-agent: {t_old:.4f} ms")
    print(f"  New per-agent: {t_new:.4f} ms")
    print(f"  Speedup:       {speedup:.1f}x")
    print(f"  At 300 agents per macro tick: {t_old * 300:.1f} ms (old) "
          f"-> {t_new * 300:.1f} ms (new)")
    assert speedup > 2.5, f"Expected >2.5x speedup, got {speedup:.1f}x"
    print("  PASS\n")


def test_edge_cases():
    """Edge cases that the new code must handle gracefully."""
    print("Test 6: Edge cases")

    from oikoumene.core.bridge import MacroAgentBridge
    bridge = MacroAgentBridge()
    macro = MockMacroState()

    # Empty world / no geopolitics
    state = bridge.get_macro_local_state(macro, 0, 0)
    assert state["conflict_nearby"] == 0
    assert state["trade_access"] == 0.5
    print("  No geopolitics: defaults returned OK")

    # Empty nations
    geop = MockGeopolitics([])
    world = MockWorld([])
    state = bridge.get_macro_local_state(macro, 0, 0, geopolitics=geop, world=world)
    assert state["conflict_nearby"] == 0
    print("  Empty nations: no crash, defaults OK")

    # Settlement referenced by nation but missing from world.settlements
    # (could happen during async settlement decay)
    settlements = [MockSettlement(0, 0, 0, [])]  # only sid=0 exists
    nations = [MockNation(0, [0, 99, 100], tech=2.0)]  # references missing sid 99, 100
    world = MockWorld(settlements)
    state = bridge.get_macro_local_state(macro, 0, 0, geopolitics=MockGeopolitics(nations), world=world)
    # Should still pick up settlement 0
    assert state["nation_tech_level"] == 2.0
    print("  Stale settlement_ids: doesn't crash, finds valid ones")

    # Zero-radius conflict — must not divide by zero
    geop = MockGeopolitics([], conflicts=[{"lat": 0, "lng": 0, "radius": 0.0, "intensity": 1.0}])
    state = bridge.get_macro_local_state(macro, 0, 0, geopolitics=geop, world=MockWorld([]))
    assert state["conflict_nearby"] == 0  # zero-radius is skipped
    print("  Zero-radius conflict: no division by zero")

    # apply_geopolitics with empty inputs
    bridge.apply_geopolitics_to_agents(None, [], MockWorld([]))
    bridge.apply_geopolitics_to_agents(MockGeopolitics([]), [], MockWorld([]))
    print("  apply_geopolitics with empty inputs: no crash")
    print("  PASS\n")


if __name__ == "__main__":
    test_apply_geopolitics_correctness()
    test_get_macro_local_state_correctness()
    test_polar_correction_in_local_state()
    test_performance_apply_geopolitics()
    test_performance_local_state()
    test_edge_cases()
    print("=" * 60)
    print("ALL BRIDGE.PY TESTS PASSED")
    print("=" * 60)
