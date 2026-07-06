"""
Regression tests for the simulation-core bug fixes.

Covers:
  H2  present_day seeded nations survive the first geopolitics update
  H3  tech-discovery RNG is seeded (no global np.random leak) and reproducible
  M1  the scientific logger starts in present_day and end_run is idempotent
  M2  the paleo->macro handoff's negative anomaly survives the first ODE step
  H1  the sim-loop generation token invalidates a stale loop
  M3  god-mode drought composes with (survives) the macro bridge rewrite
  M4  resource baselines are captured pristine at init, not 50 ticks late
  M5  all entity id counters reset per World (reproducible ids)
  M6  settlement membership, negotiation history, alliance sets stay bounded
  L1  velocity observations stay ~[-1,1] regardless of era speed
  L2  premature health death is labeled illness, not old_age
  L3  migration probes produce in-range wrapped coordinates
  L4  spawn locations depend on the world seed
  L7  the macro handoff recomputes derived fields immediately
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# H2: seeded present-day nations must not be pruned on the first update
# ---------------------------------------------------------------------------

def test_present_day_nations_survive_first_geopolitics_update():
    from world import World

    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(300)

    n_before = len(w.geopolitics.nations)
    assert n_before > 0, "present_day should seed real-world nations"

    # Directly run the geopolitics cycle that used to delete them all.
    w.geopolitics.update(w.settlements, w.agents, w.macro.state)

    n_after = len(w.geopolitics.nations)
    assert n_after == n_before, (
        f"seeded nations were pruned: {n_before} -> {n_after}"
    )
    # Their seeded macro population must not have been zeroed.
    assert any(n.population > 0 for n in w.geopolitics.nations)
    # And they are flagged as seeded.
    assert all(n.seeded for n in w.geopolitics.nations)


def test_emergent_empty_nation_is_still_pruned():
    """The fix must not disable pruning for genuine (non-seeded) empty nations."""
    from geopolitics import GeopoliticalSystem, NationState

    g = GeopoliticalSystem(rng=np.random.RandomState(0))
    ghost = NationState(id=1, name="Ghost", settlement_ids=[999], seeded=False)
    g.nations.append(ghost)
    g.relation_graph.add_node(1)
    g.trade_graph.add_node(1)

    # No settlement with id 999 exists -> emergent nation should be removed.
    g._check_nation_formation(settlements=[], alive_agents=[])
    assert len(g.nations) == 0


# ---------------------------------------------------------------------------
# H3: reproducible, seeded tech discovery
# ---------------------------------------------------------------------------

def test_history_tech_discovery_uses_seeded_rng():
    """Perturbing the global np.random must not change tech-discovery outcomes."""
    from history import HistoricalSimulation

    def run(seed):
        sim = HistoricalSimulation(start_year_bp=12000,
                                   rng=np.random.RandomState(seed))
        techs = []
        for _ in range(2000):
            out = sim.advance_time(1)
            techs.extend(out["new_techs"])
            if sim.year_bp <= 0:
                break
        return techs

    np.random.seed(1)
    a = run(42)
    np.random.seed(999999)  # perturb global RNG
    b = run(42)
    assert a == b, "seeded history run must be independent of global np.random"

    # Different seed should (almost surely) diverge in discovery timing.
    c = run(7)
    assert a != c


def test_world_history_shares_seeded_rng():
    from world import World
    w = World(seed=5)
    assert w.history.rng is w.rng, "history must use the world's seeded RNG"


# ---------------------------------------------------------------------------
# M1: logger lifecycle
# ---------------------------------------------------------------------------

def test_logger_starts_in_present_day(tmp_path):
    from world import World

    w = World(seed=1, scenario_id="present_day")
    w.logger.config.enabled = True
    w.logger.config.log_dir = str(tmp_path)
    w.spawn_initial_agents(50)

    assert w.logger.is_running, "logger must start in present_day"

    w.step()
    w.logger.end_run()
    # A timeseries CSV with a header + at least one row must exist.
    csvs = list(tmp_path.rglob("timeseries.csv"))
    assert csvs, "no timeseries.csv written"
    assert csvs[0].read_text().count("\n") >= 2


def test_end_run_is_idempotent():
    from sim_logger import SimulationLogger, LoggerConfig
    lg = SimulationLogger(LoggerConfig(enabled=False))
    # Never started: end_run must not raise.
    lg.end_run()
    lg.end_run()
    assert lg.is_running is False


# ---------------------------------------------------------------------------
# M2: macro clamp preserves the negative handoff anomaly across the first step
# ---------------------------------------------------------------------------

def test_macro_handoff_continuity_survives_first_ode_step():
    from world import World

    w = World(seed=1, scenario_id="historical")
    paleo = w.history.paleoclimate.get_climate(201)
    assert paleo["temperature_anomaly"] < 0, "precondition: paleo anomaly is negative"

    w.history.year_bp = 199
    w._hand_off_macro_from_paleo()
    seeded_temp = w.macro.state.temperature_anomaly

    # Advance the macro ODE by one step; the anomaly must NOT snap up to 0.
    w.macro.step()
    after = w.macro.state.temperature_anomaly

    assert after < 0.1, f"anomaly snapped upward after first ODE step: {after}"
    assert abs(after - seeded_temp) < 0.2, (
        f"discontinuity at first ODE step: {seeded_temp} -> {after}"
    )


def test_macro_deep_ocean_temp_can_stay_negative():
    from macro import MacroModel
    m = MacroModel()
    m.state.deep_ocean_temp = -0.06
    y = m._state_to_vector()
    m._vector_to_state(y)  # round-trip must not clamp -0.06 to 0
    assert m.state.deep_ocean_temp < 0.0


# ---------------------------------------------------------------------------
# H1: simulation-loop generation token invalidates stale loops
# ---------------------------------------------------------------------------

def test_loop_generation_invalidates_stale_loop():
    import app as A

    A.loop_generation = 0
    A._start_loop()
    gen_a = A.loop_generation
    assert A.sim_running is True

    # A restart (as reset/jepa do) must bump the generation so the old loop,
    # which captured gen_a, no longer matches and will exit.
    A._stop_loop()
    A._start_loop()
    gen_b = A.loop_generation

    assert gen_b != gen_a, "restart must invalidate the previous loop generation"
    # Clean up so the spawned loop does not keep running against a torn state.
    A._stop_loop()


# ---------------------------------------------------------------------------
# M3: drought composes with the macro bridge instead of being overwritten
# ---------------------------------------------------------------------------

def test_drought_factor_survives_bridge_rewrite():
    from world import World
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(50)
    res = w.resources
    land = res.food_regen > 0

    # Bridge rewrite with no drought -> reference regen.
    w.bridge.apply_macro_to_world(w.macro.state, res, w.terrain, w.fertility, w.elevation)
    fr_no_drought = res.food_regen.copy()

    # Impose a 0.5 drought factor on land, then rewrite again. The bridge MUST
    # multiply by the factor (previously it overwrote it, erasing the drought).
    res.drought_food_factor[land] = 0.5
    w.bridge.apply_macro_to_world(w.macro.state, res, w.terrain, w.fertility, w.elevation)
    fr_drought = res.food_regen.copy()

    assert np.allclose(fr_drought[land], fr_no_drought[land] * 0.5, rtol=1e-9)


def test_ice_age_effects_skipped_in_modern_era():
    """present_day is macro-driven; the paleo ice-age rewrite must not run and
    clobber the bridge's regen back to the terrain baseline."""
    from world import World
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(50)
    assert w.macro_always_active is True
    # is_modern is True for present_day, so stepping past a %50 tick must keep
    # the drought factor intact (ice-age would have rebuilt regen from baseline).
    w.god_mode.config.enabled = True
    # Mark a drought factor and step across a multiple of 50.
    w.resources.drought_food_factor[:] = 0.7
    for _ in range(52):
        w.step()
    assert np.allclose(w.resources.drought_food_factor, 0.7), \
        "ice-age effects ran in modern era and disturbed resource state"


# ---------------------------------------------------------------------------
# M4: pristine baselines captured at init
# ---------------------------------------------------------------------------

def test_resource_baselines_captured_at_init():
    from world import World
    w = World(seed=1, scenario_id="historical")
    res = w.resources
    assert res._baseline_food_regen is not None
    assert res._was_iced is not None and not res._was_iced.any()
    # Baseline must equal the freshly-initialized regen (terrain x fertility),
    # i.e. captured before any harvest/macro mutation. In a brand-new world
    # nothing has mutated food_regen yet, so they match exactly.
    assert np.array_equal(res._baseline_food_regen, res.food_regen)


# ---------------------------------------------------------------------------
# M5: all entity id counters reset per World
# ---------------------------------------------------------------------------

def test_all_entity_id_counters_reset():
    from world import World, Business, Settlement, Agent
    from geopolitics import GeopoliticalSystem

    w1 = World(seed=1, scenario_id="present_day")
    w1.spawn_initial_agents(50)
    # Force a business id to advance.
    Business._next_id = 123

    World(seed=1, scenario_id="historical")  # constructing a new World resets counters
    assert Business._next_id == 0, "Business id counter not reset per World"
    assert Agent._next_id == 0
    assert Settlement._next_id == 0
    assert GeopoliticalSystem._next_nation_id == 0


def test_seeded_nation_ids_do_not_collide_with_emergent():
    from world import World
    from geopolitics import GeopoliticalSystem
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(50)
    seeded_ids = {n.id for n in w.geopolitics.nations}
    # The emergent counter must sit at or above the highest seeded id, so the
    # next organically-formed nation cannot reuse a seeded country's id.
    assert GeopoliticalSystem._next_nation_id >= max(seeded_ids)


# ---------------------------------------------------------------------------
# M6: bounded structures
# ---------------------------------------------------------------------------

def test_settlement_members_pruned_of_dead():
    from world import World, Settlement
    w = World(seed=1, scenario_id="historical")
    w.spawn_initial_agents(10)
    a0, a1, a2 = w.agents[0], w.agents[1], w.agents[2]

    s = Settlement(a0.lat, a0.lng, founder_id=a0.id, rng=w.rng)
    s.members = {a0.id, a1.id, a2.id}
    a1.alive = False  # a member dies

    s.update(w.agents)
    assert a1.id not in s.members, "dead member was not pruned"
    assert {a0.id, a2.id} <= s.members


def test_alliance_refs_purged_when_nation_removed():
    from geopolitics import GeopoliticalSystem, NationState
    g = GeopoliticalSystem(rng=np.random.RandomState(0))
    keep = NationState(id=1, name="Keep", settlement_ids=[10], seeded=True)
    keep.alliances.add(2)
    keep.rivals.add(3)
    gone = NationState(id=2, name="Gone", settlement_ids=[999], seeded=False)
    for n in (keep, gone):
        g.nations.append(n)
        g.relation_graph.add_node(n.id)
        g.trade_graph.add_node(n.id)

    # settlement 999 does not exist -> `gone` is pruned; keep must lose the ref.
    g._check_nation_formation(settlements=[], alive_agents=[])
    assert 2 not in keep.alliances, "dangling alliance id not purged"


def test_negotiation_history_is_bounded():
    from geopolitics import GeopoliticalSystem, NationState
    g = GeopoliticalSystem(rng=np.random.RandomState(0))
    for i in (1, 2):  # summits require >= 2 nations
        n = NationState(id=i, name=f"N{i}", settlement_ids=[i], carbon_policy=0.1)
        g.nations.append(n)
    g.negotiation_history = [{"i": i} for i in range(g._MAX_NEGOTIATION_HISTORY + 50)]

    class _M:
        year = 2050.0
    # Trigger one append at the 24-tick cadence, then it must be truncated.
    g._negotiation_counter = 23
    g._conduct_negotiations(_M())
    assert len(g.negotiation_history) <= g._MAX_NEGOTIATION_HISTORY


# ---------------------------------------------------------------------------
# L1: bounded velocity observation
# ---------------------------------------------------------------------------

def test_velocity_observation_bounded_by_era_speed():
    from agents import Agent
    a = Agent(0.0, 0.0, rng=np.random.RandomState(0))
    a._era_speed = 20.0
    a.vlat = Agent.MAX_SPEED * 20.0     # at the era-scaled max
    a.vlng = -Agent.MAX_SPEED * 20.0
    obs = a.observe({})
    assert abs(obs[18]) <= 1.01 and abs(obs[19]) <= 1.01


# ---------------------------------------------------------------------------
# L2: death-cause labeling
# ---------------------------------------------------------------------------

def test_premature_health_death_is_illness():
    from world import World
    w = World(seed=1, scenario_id="historical")
    w.spawn_initial_agents(10)
    a = w.agents[0]
    a.age = 1                 # young -> below the aging threshold
    a.energy = 60.0           # not starving
    a.health = -1.0           # health collapse (plague/conflict-like)
    result = a.update(w)
    assert result is not None and result["event"] == "death"
    assert result["cause"] == "illness"


# ---------------------------------------------------------------------------
# L3: migration coordinate validity
# ---------------------------------------------------------------------------

def test_migrate_produces_in_range_coordinates():
    from world import World
    w = World(seed=1, scenario_id="historical")
    w.spawn_initial_agents(10)
    a = w.agents[0]
    # Push the agent to an extreme so an unclamped probe would leave the grid.
    a.lat = a.LAT_MAX
    a.lng = 179.0
    for _ in range(20):
        a._action_migrate(w)
        assert a.LAT_MIN <= a.lat <= a.LAT_MAX
        assert -180.0 <= a.lng <= 180.0


# ---------------------------------------------------------------------------
# L4: spawn locations depend on the world seed
# ---------------------------------------------------------------------------

def test_spawn_locations_depend_on_seed():
    from history import get_spawn_locations
    a = get_spawn_locations(45000, 20, rng=np.random.RandomState(1))
    b = get_spawn_locations(45000, 20, rng=np.random.RandomState(2))
    assert a != b, "spawn locations ignore the world seed"
    # Determinism preserved for a fixed seed.
    c = get_spawn_locations(45000, 20, rng=np.random.RandomState(1))
    assert a == c


# ---------------------------------------------------------------------------
# L7: handoff recomputes derived macro fields immediately
# ---------------------------------------------------------------------------

def test_handoff_recomputes_derived_fields():
    from world import World
    w = World(seed=1, scenario_id="historical")
    # Present-day default radiative forcing (~425 ppm) is large.
    assert w.macro.state.radiative_forcing > 1.0

    w.history.year_bp = 199
    w._hand_off_macro_from_paleo()
    # After handoff the CO2 is pre-industrial (~283 ppm) -> forcing near zero.
    # If _compute_derived were not called, this would still read the ~2.2 default.
    assert w.macro.state.radiative_forcing < 0.3
