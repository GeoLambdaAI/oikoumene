"""
Oikoumene — simulation engine.

Manages:
- Real Earth geography with lat/lng coordinate system
- Biome-based terrain from coordinates (climate model)
- Resource spawning and depletion on Earth grid
- Business/economic system
- Society/governance structures
- Spatial indexing for agent interactions
- Complete simulation tick logic
"""

import logging

import numpy as np
from typing import Optional
from oikoumene.agents.agents import Agent
from oikoumene.core.earth import (classify_terrain, get_fertility, is_land,
                   find_land_spawn_points, generate_earth_grid)
from oikoumene.macro.macro import MacroModel
from oikoumene.society.geopolitics import GeopoliticalSystem
from oikoumene.core.bridge import MacroAgentBridge
from oikoumene.core.history import HistoricalSimulation, get_era, get_spawn_locations
from oikoumene.llm.llm_module import LLMModule, LLMConfig
from oikoumene.core.god_mode import GodMode, GodModeConfig
from oikoumene.core.scenarios import SCENARIOS, ScenarioLoader, ScenarioConfig
from oikoumene.worldmodel.shared_world_model import SharedWorldModel
from oikoumene.core.sim_logger import SimulationLogger, LoggerConfig
from oikoumene.core.resources import ResourceMap
from oikoumene.core.settlements import Business, Settlement, SettlementsMixin
from oikoumene.core.world_paleo import PaleoMixin
from oikoumene.core.world_spatial import SpatialMixin
from oikoumene.core.world_views import _JEPA_SEED_OFFSET, ViewsMixin

log = logging.getLogger(__name__)

WORLD_MODEL_MODES = ("jepa", "none", "random", "const")
# Human-readable labels for outputs. "none" removes the whole action vector,
# which also pins behaviour intensity at its 0.1 floor, so it is not "no steering".
WORLD_MODEL_MODE_LABELS = {
    "jepa": "JEPA planner",
    "none": "no action vector",
    "random": "random action (matched scale)",
    "const": "no steering, constant intensity",
}
DEFAULT_TRAIN_INTERVAL_TICKS = 20


# Offset for the ablation RNG stream; any constant works, it only has to keep
# the "random" steering draws off the main world RNG.
_STEERING_RNG_OFFSET = 104_729


# ============================================================================
# World Engine
# ============================================================================

class World(SpatialMixin, SettlementsMixin, PaleoMixin, ViewsMixin):
    """
    Earth simulation managing terrain, agents, resources, businesses,
    and settlements using real-world coordinates. Planet-agnostic by
    design — alternative scenarios (e.g. Mars) reuse this engine with
    different terrain and climate inputs.
    """

    def __init__(self, seed: int = 42, cell_size_deg: float = 2.0,
                 config: dict = None, scenario_id: str = "historical"):
        self.seed = seed
        self.config = config or {}
        self.rng = np.random.RandomState(seed)
        # Reset class-level entity ID counters so a fixed seed yields identical
        # IDs across runs (one World per process; IDs are otherwise monotonic
        # across instances and would break same-process reproducibility). ALL
        # entity counters must be reset — missing Business/nation counters left
        # business and nation IDs drifting across in-process resets (and let a
        # fresh emergent nation collide with seeded present_day nation id 1).
        Agent._next_id = 0
        Settlement._next_id = 0
        Business._next_id = 0
        GeopoliticalSystem._next_nation_id = 0
        self.tick = 0
        self.cell_size_deg = cell_size_deg

        # Scenario
        self.scenario = SCENARIOS.get(scenario_id, SCENARIOS["historical"])
        self.scenario_loader = ScenarioLoader()
        self.macro_always_active = self.scenario.macro_active_from_start
        # One-time flag: in a historical run the macro ODE is dormant during the
        # paleo era and must be seeded from the paleoclimate trajectory the first
        # time it activates (Industrial era), so climate hands off continuously
        # instead of snapping to the present-day defaults baked into MacroState.
        self._macro_handed_off = self.macro_always_active

        # Generate Earth terrain grid
        self.earth_grid = generate_earth_grid(
            lat_min=-60, lat_max=75,
            lng_min=-180, lng_max=180,
            cell_size_deg=cell_size_deg,
            seed=seed
        )

        self.terrain = self.earth_grid["terrain"]
        self.elevation = self.earth_grid["elevation"]
        self.fertility = self.earth_grid["fertility"]

        # Resources — initialized from Earth system data
        self.resources = ResourceMap(self.earth_grid)
        self.resources.initialize_from_terrain(
            self.terrain, self.fertility,
            minerals_grid=self.earth_grid.get("minerals"),
            freshwater_grid=self.earth_grid.get("freshwater"),
            fossil_grid=self.earth_grid.get("fossil_fuels"),
        )

        # Entities
        self.agents: list[Agent] = []
        self.businesses: list[Business] = []
        self.settlements: list[Settlement] = []

        # Statistics
        self.stats_history: list[dict] = []

        # Spatial index: cKDTree for fast neighbor lookups
        from scipy.spatial import cKDTree
        self._kdtree: Optional[cKDTree] = None
        self._kdtree_alive: Optional[list] = None  # Alive agents for index mapping

        # Historical simulation (70,000 years of human civilization)
        self.start_year_bp = self.config.get("start_year_bp", 70000)
        self.history = HistoricalSimulation(start_year_bp=self.start_year_bp, rng=self.rng)

        # Macro dynamics (Club of Rome / Earth4All) — activates in Industrial+ era.
        #
        # FIX: macro.step() is invoked every `macro_update_interval` world ticks
        # (see step()), and Modern era advances time at era.time_scale = 1/12 yr
        # per tick. The ODE step size dt_years must therefore equal the elapsed
        # sim-time per macro call: macro_update_interval * 1/12 = 10/12 yr.
        # The previous value (1/12) caused macro to under-integrate by a factor
        # of macro_update_interval, so the displayed CO2/temperature/sea-level
        # evolved at ~1/10 of the calibrated rate and the macro clock fell ~10x
        # behind the historical clock. The IPCC AR6 SSP2-4.5..SSP3-7.0 anchors
        # in test_macro.py BAU run pass at both step sizes (ODE solver adapts
        # internally; 0.3% drift in CO2_2100, well within calibrated envelope).
        self.macro_update_interval = 10  # ticks between macro updates
        self.macro = MacroModel(
            config={"dt_years": self.macro_update_interval / 12.0}
        )
        self.geopolitics = GeopoliticalSystem(rng=self.rng)
        self.bridge = MacroAgentBridge()

        # Shared JEPA world model — all agents use this single model.
        # Backend (numpy default / torch) and preset are swappable at runtime
        # via set_jepa_backend(); the dashboard exposes this as a knob.
        self.shared_world_model = SharedWorldModel(obs_dim=40, action_dim=8, latent_dim=24,
                                                   seed=seed + _JEPA_SEED_OFFSET)
        self.jepa_preset = "default"

        # World-model ablation (scripts/ablation_world_model.py):
        #   "jepa"   — encode + CEM planner steer agents (default behaviour)
        #   "none"   — "no action vector": all-zero action; no encode, planner,
        #              experience or training. Intensity = clip(|a[2]|, 0.1, 1) is then
        #              pinned at 0.1, so this removes more than steering.
        #   "random" — U(-1, 1)^8 * random_scale (the CEM action bounds), drawn from a
        #              dedicated RNG so the main world RNG sequence is unaffected
        #   "const"  — movement components a[0], a[1] zero and a[2] = const_intensity,
        #              so intensity is fixed and the CEM movement force is zero
        # In every mode except "jepa" the world model is not used or trained.
        self.world_model_mode = self.config.get("world_model_mode", "jepa")
        if self.world_model_mode not in WORLD_MODEL_MODES:
            raise ValueError(f"world_model_mode must be one of {WORLD_MODEL_MODES}, "
                             f"got {self.world_model_mode!r}")
        # A scalar, or one scale per action component (per-component matched control).
        scale = np.asarray(self.config.get("random_steering_scale", 1.0), dtype=float)
        if scale.shape not in ((), (8,)) or np.any(scale < 0):
            raise ValueError(f"random_steering_scale must be a scalar or 8 values >= 0, got {scale}")
        self.random_steering_scale = scale if scale.shape else float(scale)
        self.const_intensity = float(self.config.get("const_intensity", 0.31))
        if not 0.1 <= self.const_intensity <= 1.0:
            raise ValueError(f"const_intensity must be in [0.1, 1.0], got {self.const_intensity}")
        # Ticks between central world-model training steps.
        self.train_interval_ticks = int(self.config.get("train_interval_ticks",
                                                        DEFAULT_TRAIN_INTERVAL_TICKS))
        if self.train_interval_ticks < 1:
            raise ValueError(f"train_interval_ticks must be >= 1, got {self.train_interval_ticks}")
        self._steering_rng = np.random.RandomState(seed + _STEERING_RNG_OFFSET)
        import oikoumene.agents.agents as _agents_module
        _agents_module._shared_world_model = self.shared_world_model

        # LLM Social Cognition (disabled by default — toggle via UI)
        self.llm = LLMModule(LLMConfig(enabled=False))

        # God Mode interventions (disabled by default)
        self.god_mode = GodMode(GodModeConfig(enabled=False))

        # Recent dialogues ring buffer for UI
        self.recent_dialogues: list[dict] = []

        # Scientific logger
        self.logger = SimulationLogger(LoggerConfig(enabled=True))

    # ------------------------------------------------------------------
    # Agent Management
    # ------------------------------------------------------------------

    def spawn_initial_agents(self, count: int = 25):
        """Spawn initial population based on scenario."""
        if self.scenario.id == "present_day":
            self.scenario_loader.configure_world(self, self.scenario)
            self._rebuild_spatial_grid()
            # Start the scientific logger here too — the early return previously
            # skipped start_run(), so present_day runs never logged any ticks.
            self.logger.start_run(self)
            return

        year_bp = self.history.year_bp

        if year_bp > 5000:
            # Historical mode: spawn near migration waypoints for current era
            spawn_points = get_spawn_locations(year_bp, count, rng=self.rng)
            # Validate all points are on habitable land (not ice, not ocean)
            from oikoumene.core.earth import is_land
            validated = []
            for lat, lng in spawn_points:
                if is_land(lat, lng) and not self.history.paleoclimate.get_ice_mask(year_bp, lat, lng):
                    validated.append((lat, lng))
                else:
                    # Retry near origin point
                    for _ in range(10):
                        jlat = lat + self.rng.normal(0, 5)
                        jlng = lng + self.rng.normal(0, 5)
                        if is_land(jlat, jlng) and not self.history.paleoclimate.get_ice_mask(year_bp, jlat, jlng):
                            validated.append((jlat, jlng))
                            break

            # Top up if points were dropped (their origin + all jitter retries
            # landed on ocean/ice), so the run actually starts with `count`
            # agents instead of silently fewer. Re-draw fresh waypoints; bail out
            # with a warning only if the world is too glaciated to place them.
            top_up_attempts = 0
            while len(validated) < count and top_up_attempts < count * 20:
                top_up_attempts += 1
                lat, lng = get_spawn_locations(year_bp, 1, rng=self.rng)[0]
                jlat = lat + self.rng.normal(0, 5)
                jlng = lng + self.rng.normal(0, 5)
                if is_land(jlat, jlng) and not self.history.paleoclimate.get_ice_mask(year_bp, jlat, jlng):
                    validated.append((jlat, jlng))
            if len(validated) < count:
                log.warning("[spawn] placed %d/%d agents (year_bp=%.0f); habitable land is scarce.",
                            len(validated), count, year_bp)
            spawn_points = validated[:count]
        else:
            # Modern era: spread across habitable land
            spawn_points = find_land_spawn_points(count, self.seed)

        for lat, lng in spawn_points:
            agent = Agent(lat, lng, rng=self.rng)
            agent.energy = 80 + self.rng.random() * 20
            agent.wealth = 10 + self.rng.random() * 20
            self.agents.append(agent)

        self._rebuild_spatial_grid()

        # Start scientific logger
        self.logger.start_run(self)

    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        nearest = self._find_nearest_settlement(agent.lat, agent.lng)
        if nearest and self._distance_deg(agent.lat, agent.lng, nearest.lat, nearest.lng) < 3.0:
            nearest.members.add(agent.id)

    def ablation_action(self) -> np.ndarray:
        """Action vector replacing the planner output in the "none"/"random"/"const" modes."""
        if self.world_model_mode == "random":
            return self._steering_rng.uniform(-1.0, 1.0, 8) * self.random_steering_scale
        if self.world_model_mode == "const":
            action = np.zeros(8)
            action[2] = self.const_intensity  # intensity = clip(|a[2]|, 0.1, 1)
            return action
        return np.zeros(8)

    # ------------------------------------------------------------------
    # Simulation Tick
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """Execute one simulation tick."""
        self.tick += 1
        events = []

        # Start tick timer for logger
        self.logger.tick_start()

        # Reset LLM tick counter
        if self.llm.config.enabled:
            self.llm.reset_tick_counter()

        # Process God Mode active effects
        if self.god_mode.config.enabled:
            self.god_mode.update(self)

        # Rebuild spatial grid every tick for fast lookups
        self._rebuild_spatial_grid()

        # Update all agents.
        # FIX (v0.2): iterate over a snapshot of self.agents — agents born
        # this tick (via _action_reproduce -> world.add_agent) are appended
        # to self.agents during this loop. Iterating self.agents directly
        # would cause CPython to visit those newborns in the same tick:
        # they would immediately incur metabolism, age by one tick on
        # creation, and could potentially act before being properly placed
        # in the world. The snapshot defers them to the next tick — which
        # is the natural semantics for "newly born this tick".
        for agent in list(self.agents):
            result = agent.update(self)
            if result:
                events.append(result)
                # Track actions for macro feedback
                if result.get("event") == "action":
                    self.bridge.record_agent_action(result.get("action", ""))

        # Operate businesses
        for biz in self.businesses:
            biz.operate(self)

        # Update settlements, then prune dead ones (population 0 = no living
        # members). Settlements form throughout the 68,000-year paleo era and were
        # never removed, so the list grew without bound — making both this per-tick
        # loop and the geopolitics update (each O(settlements x agents)) progressively
        # slower, and causing an abrupt stall when geopolitics first activates in the
        # Industrial era and processes the whole accumulated backlog at once.
        for settlement in self.settlements:
            settlement.update(self.agents)
        if self.settlements:
            self.settlements = [s for s in self.settlements if s.population > 0]

        # Check for new settlements
        self._check_settlement_formation()

        # Regenerate resources
        self.resources.regenerate()

        # Remove dead businesses
        self.businesses = [b for b in self.businesses if b.active or b.age < 100]

        # Train shared world model centrally (every train_interval_ticks, default 20)
        if (self.world_model_mode == "jepa" and self.tick % self.train_interval_ticks == 0
                and len(self.shared_world_model.experience_buffer) > 32):
            self.shared_world_model.train_step(
                batch_size=min(64, len(self.shared_world_model.experience_buffer))
            )

        # Collect agent dialogues for UI
        for agent in self.agents:
            if agent.alive and agent.last_dialogue:
                self.recent_dialogues.append({
                    "tick": self.tick,
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "text": agent.last_dialogue[:200],
                    "action": agent.current_action,
                })
                agent.last_dialogue = None
        if len(self.recent_dialogues) > 50:
            self.recent_dialogues = self.recent_dialogues[-50:]

        # ---- Advance historical timeline ----
        history_result = self.history.advance_time(1)
        current_era = self.history.get_current_era()

        # Set era time scale so agents can scale their movement
        self._era_time_scale = current_era.time_scale

        # Migration wave spawning: periodically add agents at frontier locations
        # This represents the leading edge of human migration across continents
        if self.tick % 20 == 0 and self.history.year_bp > 5000:
            self._spawn_migration_frontier()

        # ---- Macro + Geopolitics integration (every N ticks) ----
        # Macro ODE system: active from start in present_day scenario, else Industrial+ era
        is_modern = self.macro_always_active or self.history.year_bp < 200

        # Apply paleoclimate ice effects to resources periodically — PALEO ERA
        # ONLY. Continental ice sheets are a Pleistocene phenomenon; once the
        # macro bridge takes over as the authoritative climate->resource driver
        # (Industrial+ / present_day), running the ice-age rewrite would just
        # clobber the bridge's food/water_regen back to the terrain baseline
        # every 50th tick (cold_factor == 1 with no ice), erasing macro climate
        # damage until the next macro tick.
        if self.tick % 50 == 0 and not is_modern:
            self._apply_ice_age_effects()

        # Continuous handoff: the moment a historical run first enters the
        # Industrial era, seed the macro state from the paleoclimate trajectory so
        # temperature/CO2/population carry over smoothly instead of snapping to the
        # present-day MacroState defaults. This MUST run on the first modern tick
        # regardless of the macro update cadence — otherwise, on modern ticks that
        # are not multiples of macro_update_interval, the era-aware summary would
        # read the un-seeded (present-day) macro state and briefly flash 2026.
        if is_modern and not self._macro_handed_off:
            self._hand_off_macro_from_paleo()
            self._macro_handed_off = True

        if self.tick % self.macro_update_interval == 0 and is_modern:
            # 1. Aggregate agent actions -> macro feedback
            feedback = self.bridge.aggregate_agent_feedback(
                self.agents, self.businesses, self.settlements
            )

            # Inject conflict intensity from geopolitics
            feedback["conflict_intensity"] = self.geopolitics.get_conflict_intensity()

            # Anthropogenic CO2 scales with the civilization's industrial
            # development: present-day scenarios start fully industrial; historical
            # runs ramp up as the civ discovers industrial technologies (so a
            # low/pre-industrial civ produces little fossil CO2).
            feedback["industrialization"] = (
                1.0 if self.macro_always_active
                else self.history.industrialization_level()
            )

            # 2. Advance macro model by one step
            self.macro.step(feedback)

            # 3. Apply macro effects to world resources
            self.bridge.apply_macro_to_world(
                self.macro.state, self.resources,
                self.terrain, self.fertility, self.elevation
            )

            # 4. Update geopolitics
            self.geopolitics.update(
                self.settlements, self.agents, self.macro.state
            )

            # 5. Apply geopolitical effects to agents
            self.bridge.apply_geopolitics_to_agents(
                self.geopolitics, self.agents, self
            )

            # 6. Reset accumulators
            self.bridge.reset_accumulators()

        # Collect statistics
        alive_agents = [a for a in self.agents if a.alive]

        # Build the (history, macro, geopolitics) summaries via the helper so
        # both the websocket "tick" emit (carrying `stats`) and the "full_state"
        # emit (carrying `world.get_full_state()`) deliver identical, era-aware
        # payloads to the frontend.
        history_summary, macro_summary, geopolitics_summary = (
            self._build_era_aware_summaries()
        )

        stats = {
            "tick": self.tick,
            "population": len(alive_agents),
            # total_born is the per-world monotonic Agent ID counter (reset in
            # World.__init__), so it stays correct even after dead agents are
            # pruned from self.agents below.
            "total_born": Agent._next_id,
            "avg_energy": float(np.mean([a.energy for a in alive_agents])) if alive_agents else 0,
            "avg_wealth": float(np.mean([a.wealth for a in alive_agents])) if alive_agents else 0,
            "avg_happiness": float(np.mean([a.happiness for a in alive_agents])) if alive_agents else 0,
            "avg_age": float(np.mean([a.age for a in alive_agents])) if alive_agents else 0,
            "max_generation": max((a.generation for a in alive_agents), default=0),
            "businesses": len([b for b in self.businesses if b.active]),
            "settlements": len(self.settlements),
            "events": events[:20],
            "history": history_summary,
            "macro": macro_summary,
            "geopolitics": geopolitics_summary,
            "llm": self.llm.get_status(),
            "god_mode": self.god_mode.get_status(),
        }
        self.stats_history.append(stats)
        if len(self.stats_history) > 2000:
            self.stats_history = self.stats_history[-1000:]

        # Scientific logging
        self.logger.log_tick(self, stats)

        # Prune dead agents so self.agents stays O(alive): unbounded growth over
        # long historical runs is both a memory leak and an O(total-born) cost in
        # every per-tick loop. Dead agents carry no behaviour; relationships and
        # settlement membership reference agent IDs, not list positions.
        if len(self.agents) != len(alive_agents):
            self.agents = [a for a in self.agents if a.alive]

        return stats
