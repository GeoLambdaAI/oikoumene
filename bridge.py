"""
Coupling Bridge — Connects macro dynamics <-> agent world <-> geopolitics.

Three bidirectional data flows every macro tick:
1. AGENTS -> MACRO: Agent actions aggregate into emissions, extraction, tech investment
2. MACRO -> WORLD: Climate/pollution degrades ResourceMap fertility & yields
3. GEOPOLITICS <-> BOTH: Nation policies constrain agents; conflict feeds macro stress

Design principle: the macro model changes the ENVIRONMENT (resources, fertility),
agents then autonomously adapt. No direct agent control from macro layer.

References:
- World3 resource-agriculture feedback (Meadows 2004)
- IPCC AR6 WG2 Chapter 5: food production under climate change
- Schewe et al. 2014: freshwater stress projections
"""

import numpy as np
from typing import TYPE_CHECKING, Optional

from earth import TerrainType

if TYPE_CHECKING:
    from macro import MacroState
    from world import ResourceMap, Settlement, Business
    from agents import Agent


# Per-terrain food-regeneration factors. MUST stay in sync with
# ResourceMap.initialize_from_terrain in world.py:78-113. We mirror these
# here (rather than copying the live food_regen array on first call) because
# world._apply_ice_age_effects mutates food_regen multiplicatively across
# paleo ticks (food_regen *= cold_factor), so a live snapshot taken on the
# first modern-era bridge call in a paleo-to-modern run would capture a
# paleo-decimated baseline. Re-deriving from terrain x fertility gives the
# clean modern-era baseline this layer is meant to scale.
_FOOD_REGEN_TERRAIN_FACTOR = {
    TerrainType.OCEAN:     0.0,
    TerrainType.PLAINS:    2.0,
    TerrainType.FOREST:    1.0,
    TerrainType.MOUNTAINS: 0.2,
    TerrainType.DESERT:    0.1,
    TerrainType.TUNDRA:    0.3,
}


class MacroAgentBridge:
    """
    Bidirectional coupling between macro ODE system and agent simulation.

    Called every macro_update_interval ticks from World.step().
    """

    def __init__(self):
        # Track cumulative extraction between macro ticks for feedback
        self._harvest_accumulator: float = 0.0
        self._business_ticks: int = 0
        self._research_actions: int = 0
        self._total_actions: int = 0
        # Snapshots of the resource map's regeneration arrays at first sight,
        # so apply_macro_to_world can rebuild them as base * macro_factor each
        # call instead of multiplying a one-way ratchet (FIX B1).
        # food_regen baseline is DERIVED from terrain x fertility rather than
        # copied live (see _FOOD_REGEN_TERRAIN_FACTOR comment), water and
        # minerals are copied live since no upstream code mutates them.
        self._base_water_regen: Optional[np.ndarray] = None
        self._base_minerals_regen: Optional[np.ndarray] = None
        self._base_food_regen: Optional[np.ndarray] = None
        self._base_resource_map_id: Optional[int] = None

    def reset_accumulators(self):
        """Reset per-interval accumulators. Called after each macro step."""
        self._harvest_accumulator = 0.0
        self._business_ticks = 0
        self._research_actions = 0
        self._total_actions = 0

    # ------------------------------------------------------------------
    # 1. AGENTS -> MACRO: Aggregate agent activity into macro feedback
    # ------------------------------------------------------------------

    def aggregate_agent_feedback(
        self,
        agents: list,
        businesses: list,
        settlements: list,
    ) -> dict:
        """
        Aggregate agent-level activity into macro-model feedback signals.

        Returns dict consumed by MacroModel.step():
            emission_multiplier: float    - industrial activity scaling
            extraction_multiplier: float  - resource extraction intensity
            renewable_investment: float   - green tech investment fraction
            conflict_intensity: float     - from geopolitics (set externally)
            population_factor: float      - agent reproduction rate signal
            research_boost: float         - R&D investment from agent research actions
        """
        alive = [a for a in agents if a.alive]
        n_alive = len(alive)

        if n_alive == 0:
            return self._default_feedback()

        # Population factor: agents reproducing faster -> higher pop growth
        avg_children = np.mean([a.children_count for a in alive])
        population_factor = 0.8 + 0.4 * min(1.0, avg_children / 3.0)

        # Emission multiplier: active businesses = more industrial emissions
        active_businesses = [b for b in businesses if b.active]
        n_businesses = len(active_businesses)
        # More businesses per capita = more emissions
        business_intensity = n_businesses / max(1, n_alive) * 10.0
        emission_multiplier = 0.5 + 0.5 * min(2.0, business_intensity)

        # Resource extraction: from business types and agent work actions
        extractive_types = {"farming", "mining", "crafting"}
        n_extractive = sum(1 for b in active_businesses
                           if b.business_type in extractive_types)
        extraction_multiplier = 0.7 + 0.6 * min(1.5, n_extractive / max(1, n_alive) * 10)

        # Renewable investment: fraction of businesses that are "green"
        # Research-heavy and trading businesses are proxies for service economy
        service_types = {"research", "trading", "diplomacy", "medicine"}
        n_service = sum(1 for b in active_businesses
                        if b.business_type in service_types)
        renewable_investment = n_service / max(1, n_businesses) if n_businesses > 0 else 0.0

        # Research boost: from agents doing research actions + research skill levels
        avg_research_skill = np.mean([a.skills.get_level("research") for a in alive])
        research_fraction = self._research_actions / max(1, self._total_actions)
        research_boost = (
            0.5 * avg_research_skill +
            0.5 * research_fraction
        ) * min(2.0, n_alive / 25.0)  # Scale with population

        return {
            "emission_multiplier": float(np.clip(emission_multiplier, 0.1, 3.0)),
            "extraction_multiplier": float(np.clip(extraction_multiplier, 0.1, 3.0)),
            "renewable_investment": float(np.clip(renewable_investment, 0.0, 1.0)),
            "conflict_intensity": 0.0,  # Set by geopolitics layer
            "population_factor": float(np.clip(population_factor, 0.5, 2.0)),
            "research_boost": float(np.clip(research_boost, 0.0, 2.0)),
        }

    def _default_feedback(self) -> dict:
        """Default feedback when no agents alive."""
        return {
            "emission_multiplier": 0.1,
            "extraction_multiplier": 0.1,
            "renewable_investment": 0.0,
            "conflict_intensity": 0.0,
            "population_factor": 0.5,
            "research_boost": 0.0,
        }

    def record_agent_action(self, action_type: str):
        """Called from World.step() to track agent actions between macro ticks."""
        self._total_actions += 1
        if action_type == "research":
            self._research_actions += 1

    # ------------------------------------------------------------------
    # 2. MACRO -> WORLD: Apply climate/pollution effects to resources
    # ------------------------------------------------------------------

    def apply_macro_to_world(
        self,
        macro_state: 'MacroState',
        resource_map: 'ResourceMap',
        terrain: np.ndarray,
        fertility: np.ndarray,
        elevation: np.ndarray,
    ):
        """
        Translate macro-level changes into resource map modifications.

        This is how agents FEEL climate change — through the resources they
        depend on becoming scarcer or shifting geographically.

        Effects applied:
        - Temperature -> fertility reduction (latitude-dependent)
        - Pollution -> food regeneration reduction
        - Sea level rise -> coastal cell flooding
        - Freshwater stress -> water regeneration reduction
        - Resource depletion -> global capacity scaling
        """
        T = macro_state.temperature_anomaly
        pollution = macro_state.persistent_pollution
        slr = macro_state.sea_level_rise_m
        freshwater = macro_state.freshwater_stress
        fossil_remaining = macro_state.fossil_fuels
        mineral_remaining = macro_state.minerals_global
        tech = macro_state.technology_level

        # Snapshot the per-cell regen baselines once (FIX B1 + food_regen).
        # Without this, the per-cell `*= water_factor` and `*= mineral_remaining`
        # below would compound across every macro tick, driving regen to zero
        # independently of the current macro state; and the previous food_regen
        # formula (`2.0 if plains else 1.0`) silently inflated mountain/desert/
        # tundra food_regen by 5x/10x/3.3x. Re-snap if a new resource_map is
        # passed (e.g. simulation restart with a fresh world).
        if (self._base_water_regen is None
                or self._base_resource_map_id != id(resource_map)
                or self._base_water_regen.shape != resource_map.water_regen.shape):
            self._base_water_regen = resource_map.water_regen.copy()
            self._base_minerals_regen = resource_map.minerals_regen.copy()
            # food_regen baseline derived from terrain x fertility (see module
            # comment on _FOOD_REGEN_TERRAIN_FACTOR for why this is derived,
            # not snapshotted). Vectorized lookup: build a length-6 factor
            # table indexed by terrain code in {0..5}, then broadcast.
            factor_lookup = np.zeros(6, dtype=np.float64)
            for code, factor in _FOOD_REGEN_TERRAIN_FACTOR.items():
                factor_lookup[code] = factor
            safe_terrain = np.clip(terrain.astype(np.int64), 0, 5)
            self._base_food_regen = factor_lookup[safe_terrain] * fertility
            self._base_resource_map_id = id(resource_map)

        rows, cols = terrain.shape

        # Vectorised macro -> resource translation. This was a per-cell Python
        # double loop (~8.5 ms over the 12k-cell grid) that ran every macro tick
        # once the Industrial era activated the macro layer, throttling the sim.
        # The numpy form below is numerically identical, cell-for-cell, but runs
        # in microseconds. Ocean cells (terrain code 0) are left untouched, exactly
        # as the original `continue` did.
        land = terrain != 0
        abs_lat = np.abs(
            resource_map.lat_max - (np.arange(rows) + 0.5) * resource_map.cell_size_deg
        )[:, None]  # (rows, 1), broadcasts across columns

        # --- Temperature impact on fertility (latitude-banded) ---
        # Source: Schlenker & Roberts 2009, IPCC AR6 WG2 Ch5. Each band value is a
        # scalar function of T, selected per row by absolute latitude.
        tf_tropical = max(0.3, 1.0 - 0.15 * max(0.0, T - 1.5) ** 1.3)
        tf_temperate = (1.0 + 0.05 * T) if T < 1.5 \
            else max(0.4, 1.075 - 0.08 * (T - 1.5) ** 1.5)
        tf_subarctic = min(1.3, 1.0 + 0.1 * T)
        tf_arctic = max(0.5, 1.0 - 0.05 * max(0.0, T - 2.0))
        temp_factor = np.broadcast_to(
            np.select(
                [abs_lat < 20, abs_lat < 45, abs_lat < 60],
                [tf_tropical, tf_temperate, tf_subarctic],
                default=tf_arctic,
            ).astype(np.float64),
            (rows, cols),
        ).copy()

        # --- Sea level rise: flood low-elevation cells (IPCC AR6 WG1 Ch9) ---
        if slr > 0.3:
            flood_prob = min(1.0, (slr - 0.3) / 0.5)
            if flood_prob > 0.5:
                temp_factor[elevation < 0.15] *= max(0.1, 1.0 - flood_prob)

        # --- Pollution impact on food regen (World3, Meadows 2004); scalar ---
        pollution_factor = max(0.4, 1.0 - 0.4 * pollution)

        # --- Freshwater stress -> water regen; arid terrain hit hardest ---
        # Source: Schewe et al. 2014. Desert (4) / Plains (1) / other terrain.
        water_factor = np.where(
            terrain == 4, max(0.2, 1.0 - 0.8 * freshwater),
            np.where(terrain == 1, max(0.5, 1.0 - 0.4 * freshwater),
                     max(0.6, 1.0 - 0.2 * freshwater)),
        )

        # --- Rebuild regen rates from cached baselines (no compounding) ---
        # food_regen baseline is the derived per-terrain * fertility grid; water
        # and minerals from their snapshots. Only land cells are written.
        # Compose with any active god-mode drought factor so this rewrite does
        # not erase a drought (which otherwise vanished at the next macro tick).
        combined = temp_factor * pollution_factor * water_factor
        resource_map.food_regen[land] = (
            self._base_food_regen[land] * combined[land]
            * resource_map.drought_food_factor[land]
        )
        resource_map.water_regen[land] = (
            self._base_water_regen[land] * water_factor[land]
            * resource_map.drought_water_factor[land]
        )
        resource_map.minerals_regen[land] = (
            self._base_minerals_regen[land] * mineral_remaining
        )

        # --- Global resource capacity scaling ---
        # As global stocks deplete, local max capacity also drops
        # This creates scarcity pressure that agents experience
        resource_map.food = np.minimum(
            resource_map.food,
            100.0 * macro_state.food_production_index
        )
        resource_map.minerals = np.minimum(
            resource_map.minerals,
            100.0 * mineral_remaining
        )

    # ------------------------------------------------------------------
    # 3. GEOPOLITICS -> AGENTS: Apply geopolitical effects
    # ------------------------------------------------------------------

    def apply_geopolitics_to_agents(
        self,
        geopolitics,
        agents: list,
        world,
    ):
        """
        Apply geopolitical effects to individual agents.

        Effects:
        - Conflict zones: agents lose health and wealth
        - Trade agreements: allied nation agents get trade bonuses
        - Sanctions: sanctioned nation agents get trade penalties
        - Technology diffusion: agents in advanced nations learn faster
        """
        if geopolitics is None or not geopolitics.nations:
            return

        alive = [a for a in agents if a.alive]

        # Build nation membership lookup: agent_id -> NationState.
        # FIX (v0.2): the previous quadruple-nested loop was O(N * M * S * K)
        # over (nations, settlements_per_nation, world_settlements, members),
        # which becomes very expensive once many settlements exist (~30k+
        # iterations per macro tick at modest scale). The new approach builds
        # an O(S) settlement_id -> Settlement dict once, then walks the
        # nation/settlement structure linearly.
        settlement_by_id = {s.id: s for s in world.settlements}
        agent_nation = {}
        for nation in geopolitics.nations:
            for sid in nation.settlement_ids:
                s = settlement_by_id.get(sid)
                if s is None:
                    continue
                for member_id in s.members:
                    agent_nation[member_id] = nation

        # Apply conflict effects
        for conflict in geopolitics.active_conflicts:
            zone_lat = conflict.get("lat", 0)
            zone_lng = conflict.get("lng", 0)
            zone_radius = conflict.get("radius", 5.0)
            intensity = conflict.get("intensity", 0.5)

            for agent in alive:
                dist = world._distance_deg(agent.lat, agent.lng, zone_lat, zone_lng)
                if dist < zone_radius:
                    # Proximity-scaled damage (clamped so conflict never drives
                    # vitals negative; this runs after the agent's own update()).
                    proximity = 1.0 - dist / zone_radius
                    agent.health = max(0.0, agent.health - 2.0 * intensity * proximity)
                    agent.wealth = max(0.0, agent.wealth - 1.0 * intensity * proximity)
                    agent.happiness = max(0.0, agent.happiness - 3.0 * intensity * proximity)

        # Technology diffusion: agents in nations with high tech benefit
        for agent in alive:
            nation = agent_nation.get(agent.id)
            if nation and nation.technology_level > 1.2:
                # Skill learning bonus from national tech infrastructure
                tech_bonus = (nation.technology_level - 1.0) * 0.01
                agent.skills.practice("research", tech_bonus, 1.0)

    # ------------------------------------------------------------------
    # Helper: Inject macro signals into agent observation
    # ------------------------------------------------------------------

    def get_macro_local_state(
        self,
        macro_state: 'MacroState',
        lat: float,
        lng: float,
        geopolitics=None,
        world=None,
    ) -> dict:
        """
        Get macro-derived fields for get_local_state().
        Extends the agent's observation with global context.
        """
        state = {
            "temperature_anomaly": macro_state.temperature_anomaly,
            "co2_level": macro_state.co2_ppm,
            "social_tension": macro_state.social_tension,
            "pollution_level": macro_state.persistent_pollution,
            "resource_scarcity": 1.0 - macro_state.fossil_fuels,
            "conflict_nearby": 0.0,
            "nation_tech_level": macro_state.technology_level,
            "trade_access": 0.5,
        }

        # Resolve a distance function: prefer World._distance_deg if available
        # (it now uses haversine in v0.2 and matches the rest of the codebase),
        # otherwise fall back to euclidean for tests/standalone use.
        if world is not None and hasattr(world, "_distance_deg"):
            distance = world._distance_deg
        else:
            def distance(la1, ln1, la2, ln2):
                return float(np.sqrt((la1 - la2) ** 2 + (ln1 - ln2) ** 2))

        # Check for nearby conflicts.
        # FIX (v0.2): previous code used euclidean distance directly via
        # np.sqrt(dlat**2 + dlng**2) — inconsistent with World._distance_deg
        # used elsewhere in this file. Using the world's distance function
        # gives consistent polar correction.
        if geopolitics and geopolitics.active_conflicts:
            for conflict in geopolitics.active_conflicts:
                radius = conflict.get("radius", 5.0)
                if radius <= 0:
                    continue
                dist = distance(lat, lng,
                                conflict.get("lat", 0), conflict.get("lng", 0))
                if dist < radius:
                    state["conflict_nearby"] = max(
                        state["conflict_nearby"],
                        conflict.get("intensity", 0.5) * (1.0 - dist / radius)
                    )

        # Nation-specific tech and trade access.
        # FIX (v0.2): the previous triple-nested loop (nations -> settlement_ids
        # -> world.settlements) was called from get_local_state, which runs
        # once per agent per simulation tick. At 300 agents x ~30 settlements
        # this produced ~4.5M iterations per tick. We now build the
        # settlement_id -> Settlement dict once and break out cleanly when a
        # nearby settlement is found. The original `break` only exited the
        # innermost loop, which was misleading — the outer loops kept running.
        if geopolitics and world:
            settlement_by_id = {s.id: s for s in world.settlements}
            found = False
            for nation in geopolitics.nations:
                if found:
                    break
                for sid in nation.settlement_ids:
                    s = settlement_by_id.get(sid)
                    if s is None:
                        continue
                    if distance(lat, lng, s.lat, s.lng) < 5.0:
                        state["nation_tech_level"] = nation.technology_level
                        state["trade_access"] = nation.trade_openness
                        found = True
                        break

        return state
