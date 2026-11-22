"""
Tests for world.py v0.2 fixes.

We test the two fixes in isolation without instantiating the full World class,
which depends on earth/history/llm/macro/geopolitics/etc. modules. This keeps
the tests self-contained and fast.

Verifies:
1. _distance_deg correctness (haversine, degree-equivalents, polar correction)
2. Iteration-mutation safety: list() snapshot semantics
3. _distance_deg threshold semantics: settlement-proximity behavior is preserved
   at the equator and corrected at high latitudes
"""
import sys, os
import numpy as np


def test_distance_haversine():
    """Reproduce the _distance_deg method standalone for testing."""
    print("Test 1: _distance_deg correctness")

    # Reproduce the method (extracted from world.py to allow standalone test)
    def distance_deg(lat1, lng1, lat2, lng2):
        phi1 = np.radians(lat1); phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lng2 - lng1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
        c = 2 * np.arcsin(min(1.0, np.sqrt(a)))
        return float(c * 6371.0 / 111.0)

    # Equator: 1 deg longitude ~= 1 deg-equiv (within 1%)
    d_eq = distance_deg(0, 0, 0, 1)
    print(f"  1 deg longitude at equator: {d_eq:.4f} deg-equiv (expect ~1.0)")
    assert 0.99 <= d_eq <= 1.01

    # 60 deg N: 1 deg longitude ~= 0.5 deg-equiv (cos(60)=0.5)
    d_60 = distance_deg(60, 0, 60, 1)
    print(f"  1 deg longitude at 60 deg N: {d_60:.4f} deg-equiv (expect ~0.5)")
    assert 0.49 <= d_60 <= 0.51

    # 75 deg N (sim's highest latitude): 1 deg longitude ~= cos(75)=0.259
    d_75 = distance_deg(75, 0, 75, 1)
    print(f"  1 deg longitude at 75 deg N: {d_75:.4f} deg-equiv (expect ~0.26)")
    assert 0.255 <= d_75 <= 0.262

    # Latitude distance is invariant
    d_lat_eq = distance_deg(0, 0, 1, 0)
    d_lat_60 = distance_deg(60, 0, 61, 0)
    d_lat_75 = distance_deg(75, 0, 76, 0)
    print(f"  1 deg latitude at 0/60/75 N: {d_lat_eq:.4f} / {d_lat_60:.4f} / {d_lat_75:.4f}")
    assert abs(d_lat_eq - d_lat_60) < 0.01
    assert abs(d_lat_eq - d_lat_75) < 0.01

    # Symmetry
    assert distance_deg(40, 5, 50, 10) == distance_deg(50, 10, 40, 5)
    print("  Symmetry: OK")

    # Same point: distance is 0
    assert distance_deg(45.123, -12.456, 45.123, -12.456) == 0.0
    print("  Identity: distance(p,p) = 0")
    print("  PASS\n")


def test_threshold_semantics_preserved():
    """
    Verify that key threshold semantics from world.py are preserved.

    Specifically:
    - get_local_state radius=5.0 deg: at equator should still admit ~5 deg
      neighbors in lng; at 60N it should now correctly require ~10 deg lng span
      (since 1 deg lng = 0.5 deg-equiv at 60N). This is the desired behavior:
      "5-degree neighborhood" should mean ~555 km regardless of latitude.
    """
    print("Test 2: Threshold semantics — '5-deg neighborhood'")

    def distance_deg(lat1, lng1, lat2, lng2):
        phi1 = np.radians(lat1); phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lng2 - lng1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
        c = 2 * np.arcsin(min(1.0, np.sqrt(a)))
        return float(c * 6371.0 / 111.0)

    # At equator: an agent 5 deg lng away should be at the boundary
    d_eq_5 = distance_deg(0, 0, 0, 5)
    print(f"  At equator: agent 5 deg lng away = {d_eq_5:.3f} deg-equiv (expect ~5.0)")
    assert 4.95 <= d_eq_5 <= 5.05

    # At 60N: an agent 5 deg lng away is now correctly only ~2.5 deg-equiv
    # (because cos(60)=0.5). This means the radius-5 query at 60N now
    # correctly admits agents up to ~10 deg lng away, equivalent in km
    # to a 5-deg query at the equator.
    d_60_5 = distance_deg(60, 0, 60, 5)
    print(f"  At 60N: agent 5 deg lng away = {d_60_5:.3f} deg-equiv (expect ~2.5)")
    assert 2.45 <= d_60_5 <= 2.55

    # An agent 10 deg lng away at 60N: now ~5.0 deg-equiv (was 10 with euclidean)
    d_60_10 = distance_deg(60, 0, 60, 10)
    print(f"  At 60N: agent 10 deg lng away = {d_60_10:.3f} deg-equiv (expect ~4.97)")
    assert 4.9 <= d_60_10 <= 5.05

    # Verify the critical settlement-formation threshold (3.0 deg).
    # At equator, 3 deg lng away = 3.0 deg-equiv (boundary case)
    # At 60N, 3 deg lng away = 1.5 deg-equiv (well within boundary, OK)
    d_eq_3 = distance_deg(0, 0, 0, 3)
    d_60_3 = distance_deg(60, 0, 60, 3)
    print(f"  3-deg threshold: at equator {d_eq_3:.3f}, at 60N {d_60_3:.3f}")
    assert 2.95 <= d_eq_3 <= 3.05
    assert 1.45 <= d_60_3 <= 1.55
    print("  PASS\n")


def test_iteration_snapshot_semantics():
    """
    Verify that list() snapshot pattern correctly defers mutations.

    We simulate the world.step() pattern: iterating agents while
    new agents are appended.
    """
    print("Test 3: Iteration-mutation safety via list() snapshot")

    # Simulate the pattern
    class FakeAgent:
        def __init__(self, name, world, can_reproduce=False):
            self.name = name
            self.world = world
            self.can_reproduce = can_reproduce
            self.updates = 0
        def update(self):
            self.updates += 1
            if self.can_reproduce and self.updates == 1:
                # On first update, add a child
                self.world.append(FakeAgent(f"{self.name}_child", self.world))

    # Without snapshot: child gets visited in same loop
    agents_unsafe = []
    parent = FakeAgent("parent", agents_unsafe, can_reproduce=True)
    agents_unsafe.append(parent)
    for agent in agents_unsafe:  # buggy pattern
        agent.update()
    n_visited_unsafe = sum(a.updates for a in agents_unsafe)
    print(f"  Without snapshot: {len(agents_unsafe)} agents in list, {n_visited_unsafe} updates")
    assert len(agents_unsafe) == 2
    assert n_visited_unsafe == 2  # parent + child both updated

    # With snapshot: child deferred to next tick
    agents_safe = []
    parent = FakeAgent("parent", agents_safe, can_reproduce=True)
    agents_safe.append(parent)
    for agent in list(agents_safe):  # safe pattern
        agent.update()
    n_visited_safe = sum(a.updates for a in agents_safe)
    print(f"  With list() snapshot: {len(agents_safe)} agents in list, {n_visited_safe} updates")
    assert len(agents_safe) == 2  # child was added
    assert n_visited_safe == 1    # but only parent was updated

    # Next tick: child gets its first update
    for agent in list(agents_safe):
        agent.update()
    n_visited_after = sum(a.updates for a in agents_safe)
    print(f"  After 2nd tick: {n_visited_after} total updates (expect 3: parent×2 + child×1)")
    assert n_visited_after == 3
    print("  PASS\n")


def test_no_unused_state_drift():
    """
    Verify that the snapshot pattern doesn't accidentally hide any agents.
    All agents added to the world should eventually receive update() calls.
    """
    print("Test 4: Snapshot pattern doesn't lose agents")

    class TrackedAgent:
        def __init__(self, name, world, reproduces_at=None):
            self.name = name
            self.world = world
            self.updates = 0
            self.reproduces_at = reproduces_at or []
            self.alive = True
        def update(self):
            self.updates += 1
            if self.updates in self.reproduces_at:
                self.world.append(TrackedAgent(f"{self.name}-c{self.updates}", self.world))

    # Simulate 10 ticks with one parent that reproduces at ticks 1, 3, 5
    world = []
    world.append(TrackedAgent("root", world, reproduces_at=[1, 3, 5]))
    for tick in range(10):
        for a in list(world):
            if a.alive:
                a.update()

    # Expect: root has 10 updates, c1 has 9 (born at tick 1), c3 has 7, c5 has 5
    expected = {"root": 10, "root-c1": 9, "root-c3": 7, "root-c5": 5}
    actual = {a.name: a.updates for a in world}
    print(f"  Expected updates: {expected}")
    print(f"  Actual updates:   {actual}")
    assert actual == expected, f"got {actual}"
    print("  PASS\n")


if __name__ == "__main__":
    test_distance_haversine()
    test_threshold_semantics_preserved()
    test_iteration_snapshot_semantics()
    test_no_unused_state_drift()
    print("=" * 60)
    print("ALL WORLD.PY FIX TESTS PASSED")
    print("=" * 60)
