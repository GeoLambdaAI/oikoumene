"""
Autonomous AI Agent System with Physics-Based Movement.

Each agent is a fully autonomous entity with:
- JEPA world model for understanding/predicting the environment (LeCun 2025)
- Autoresearch-inspired learning loop (hypothesize -> act -> evaluate -> adapt)
- Physics-based movement with velocity, momentum, and terrain friction
- Personality traits, skills, memory, and goals
- Ability to reproduce (spawn new agents with inherited + mutated traits)
- Economic behavior (work, trade, build businesses)
- Social behavior (relationships, communication, governance)
- Death from starvation, health loss, or old age
"""

import numpy as np
from typing import Optional

from oikoumene.agents.actions import ActionsMixin
from oikoumene.agents.memory import AgentMemory, SkillSet
from oikoumene.agents.movement import MovementMixin
from oikoumene.agents.planning import PlanningMixin
from oikoumene.agents.traits import generate_traits
from oikoumene.worldmodel.world_model import JEPAWorldModel

# Shared world model — set by World.__init__(), all agents reference it
_shared_world_model = None


# ============================================================================
# Autonomous Agent
# ============================================================================

class Agent(MovementMixin, PlanningMixin, ActionsMixin):
    """
    Fully autonomous AI agent with physics-based movement.

    Core loop (autoresearch-inspired):
    1. Observe environment
    2. Encode observation via JEPA world model
    3. Plan actions using CEM in latent space
    4. Apply physics-based movement toward goal
    5. Execute action when in range
    6. Evaluate outcome
    7. Update world model and memory
    """

    _next_id = 0

    # Movement physics constants (in degrees) — BASE values for Modern era (1 month/tick)
    # These get scaled by era_speed_multiplier in update() for historical eras
    MAX_SPEED = 0.15         # Max degrees per tick (~15km) base
    FRICTION = 0.85          # Velocity damping per tick
    WANDER_STRENGTH = 0.02   # Random exploration force base
    GOAL_FORCE = 0.06        # Force toward current goal base
    SEPARATION_RADIUS = 0.3  # Min distance between agents (~30km)
    SEPARATION_FORCE = 0.04  # Push away when too close
    POPULATION_PRESSURE_RADIUS = 3.0  # Degrees: crowding pushes agents outward
    POPULATION_PRESSURE_FORCE = 0.05  # Outward push per nearby agent

    # Simulated habitable latitude band. Latitude is clamped to this range
    # (the poles are outside the agent world); longitude wraps at the dateline.
    LAT_MIN = -58.0
    LAT_MAX = 73.0

    # Life-cycle thresholds expressed in REAL-WORLD YEARS, not ticks.
    # These get converted to ticks at runtime via _yrs_to_ticks() using the
    # current era's time scale. This keeps behavior coherent across the
    # Paleolithic-to-Modern range of 200 years/tick down to 1 month/tick.
    REPRODUCE_MIN_AGE_YEARS = 15.0       # Earliest reproductive age
    REPRODUCE_COOLDOWN_YEARS = 6.0       # Years between own births
    AGING_THRESHOLD_YEARS = 60.0         # Onset of senescence health decline
    METABOLISM_DOUBLE_YEARS = 666.0      # Years at which age-related metab. doubles
                                         # (kept at the implicit Modern-era value
                                         #  8000 ticks * (1/12) yr/tick = 666 yr,
                                         #  so Modern behavior is unchanged)
    LIFESPAN_NORM_YEARS = 80.0           # Normalization for obs[4] age input
    SOCIALIZE_MIN_AGE_YEARS = 8.0        # Earliest age that motivates socializing
    # Earliest age that motivates governing. Previously read SOCIALIZE_MIN_AGE_YEARS
    # (likely copy-paste); split out with the same value, pending a decision.
    GOVERN_MIN_AGE_YEARS = 8.0

    def __init__(self, x: float, y: float,
                 parent_a: Optional['Agent'] = None,
                 parent_b: Optional['Agent'] = None,
                 *, rng: np.random.RandomState):
        # Seeded RNG for full reproducibility (required). World passes its own
        # RandomState so a fixed World(seed=...) yields identical trajectories.
        self.rng = rng

        Agent._next_id += 1
        self.id = Agent._next_id
        self.alive = True
        self.age = 0
        self.generation = 0

        # Position (lat/lng) and physics (degrees per tick)
        self.lat = x   # latitude
        self.lng = y   # longitude
        self.vlat = self.rng.uniform(-0.02, 0.02)
        self.vlng = self.rng.uniform(-0.02, 0.02)

        # Vital stats
        self.energy = 100.0
        self.health = 100.0
        self.wealth = 10.0
        self.happiness = 50.0

        # Genetics / Traits
        pa_traits = parent_a.traits if parent_a else None
        pb_traits = parent_b.traits if parent_b else None
        self.traits = generate_traits(pa_traits, pb_traits, rng=self.rng)

        if parent_a:
            self.generation = max(parent_a.generation,
                                  parent_b.generation if parent_b else 0) + 1

        # Name generation
        self.name = self._generate_name()

        # Skills
        self.skills = SkillSet(rng=self.rng)
        if parent_a:
            self.skills.inherit(parent_a.skills)
        if parent_b:
            self.skills.inherit(parent_b.skills, 0.2)

        # Memory (autoresearch-style experiment log)
        self.memory = AgentMemory()

        # World model (JEPA - LeCun 2025) — shared across all agents
        # The shared model is set by World.__init__() via agents._shared_world_model
        # Falls back to per-agent model if shared not available (backward compat)
        if _shared_world_model is not None:
            self.world_model = _shared_world_model
        else:
            self.world_model = JEPAWorldModel(40, 8, latent_dim=24)

        # Cached plan for tick-skipping (Step 2)
        self._cached_behavior: Optional[dict] = None
        self._plan_tick: int = 0  # Last tick when agent planned

        # Social
        self.relationships: dict[int, float] = {}  # agent_id -> affinity (-1 to 1)
        self.business_id: Optional[int] = None
        self.employer_id: Optional[int] = None
        self.home_lat: Optional[float] = None
        self.home_lng: Optional[float] = None

        # Current state
        self.current_action = "wander"
        self.current_goal = "survive"
        self.goal_lat: Optional[float] = None  # Target position
        self.goal_lng: Optional[float] = None

        # Reproduction
        self.reproduction_cooldown = 0
        self.children_count = 0

        # Era time scale cache — refreshed each update() from world.
        # Default = Modern (1 month/tick) so methods called before the first
        # update() (e.g. during agent construction in tests) still work.
        self._era_time_scale_cache: float = 1.0 / 12.0

        # LLM / God Mode attributes
        self.last_dialogue: Optional[str] = None
        self.dialogue_history: list[dict] = []
        self.divine_messages: list[dict] = []
        self.divine_trust: float = 0.5

        # Movement heading (for smooth direction changes)
        angle = self.rng.uniform(0, 2 * np.pi)
        self.heading_x = np.cos(angle)
        self.heading_y = np.sin(angle)

    def _generate_name(self) -> str:
        prefixes = ["Al", "Be", "Ca", "De", "El", "Fa", "Gi", "Ha", "Ix", "Jo",
                     "Ka", "Le", "Ma", "Ne", "Or", "Pa", "Qu", "Ra", "Sa", "Te",
                     "Ul", "Va", "Wi", "Xe", "Ya", "Ze"]
        suffixes = ["ra", "on", "ix", "us", "ia", "en", "or", "is", "um", "ax",
                     "el", "an", "os", "yl", "in", "ar", "et", "ov", "ut", "ab"]
        mid = ["ri", "lo", "na", "vi", "th", "mo", "da", "si", "ke", ""]
        return self.rng.choice(prefixes) + self.rng.choice(mid) + self.rng.choice(suffixes)

    # ------------------------------------------------------------------
    # Era-aware time helpers
    # ------------------------------------------------------------------

    def _yrs_to_ticks(self, years: float) -> int:
        """
        Convert real-world years to ticks at the current era's time scale.
        Floors to a minimum of 1 tick so that thresholds remain meaningful in
        Paleolithic eras (200 yr/tick) where most life-cycle events are sub-tick.
        """
        return max(1, int(round(years / self._era_time_scale_cache)))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update(self, world) -> Optional[dict]:
        """Main agent update cycle - called each simulation tick."""
        if not self.alive:
            return None

        self.age += 1
        self.reproduction_cooldown = max(0, self.reproduction_cooldown - 1)

        # Era speed multiplier: in early eras (200yr/tick), agents represent
        # entire populations migrating over centuries. Scale movement accordingly.
        # Human migration rate: ~1 km/year = ~0.01 deg/year
        # At 200 yr/tick -> need ~2.0 deg/tick effective speed
        era_time_scale = getattr(world, '_era_time_scale', 1.0/12.0)
        self._era_time_scale_cache = era_time_scale  # for _yrs_to_ticks() throughout this tick
        # Scale factor: how many "modern months" this tick represents
        self._era_speed = min(20.0, era_time_scale / (1.0/12.0))

        # Metabolism — base drain plus age-related component.
        # Age component: doubles after METABOLISM_DOUBLE_YEARS (era-scaled).
        # In Modern this matches the previous hardcoded 8000-tick denominator
        # (8000 ticks * 1/12 yr/tick = 666 yr); in Paleolithic the same number
        # of years compresses to fewer ticks.
        metab_double_ticks = self._yrs_to_ticks(self.METABOLISM_DOUBLE_YEARS)
        base_metabolism = 0.15 + (self.age / max(1, metab_double_ticks))
        speed = np.sqrt(self.vlat**2 + self.vlng**2)
        movement_cost = speed * 0.5  # Reduced: movement is less costly in migration eras
        self.energy -= base_metabolism + movement_cost
        self.happiness *= 0.999

        # Aging effects on health — kick in after AGING_THRESHOLD_YEARS (era-scaled)
        aging_threshold = self._yrs_to_ticks(self.AGING_THRESHOLD_YEARS)
        if self.age > aging_threshold:
            self.health -= 0.03 * (self.age / aging_threshold)
            self.health += 0.01 * self.traits["resilience"]

        # Death check. Distinguish the three real causes instead of labeling
        # every health-driven death "old_age": energy exhaustion is starvation;
        # a health collapse past the aging threshold is old age; a health
        # collapse before it is premature (disease/plague/conflict damage).
        if self.energy <= 0 or self.health <= 0:
            self.alive = False
            if self.energy <= 0:
                cause = "starvation"
            elif self.age > aging_threshold:
                cause = "old_age"
            else:
                cause = "illness"
            return {"event": "death", "agent_id": self.id, "agent_name": self.name,
                    "cause": cause, "age": self.age, "generation": self.generation}

        # Observe environment
        world_state = world.get_local_state(self.lat, self.lng)
        observation = self.observe(world_state)
        self._last_observation = observation.copy()

        # Store macro signals in memory
        self.memory.learn_fact("conflict_nearby",
                              world_state.get("conflict_nearby", 0))
        self.memory.learn_fact("resource_scarcity",
                              world_state.get("resource_scarcity", 0))

        # Decide and act
        behavior = self.decide_action(observation, world)
        outcome = self.execute_action(behavior, world)

        # Population pressure: when too many agents nearby, push outward
        nearby = world.get_nearby_agents(self.lat, self.lng,
                                          self.POPULATION_PRESSURE_RADIUS)
        n_nearby = len(nearby)
        if n_nearby > 3:
            # Crowding pressure increases with density
            pressure = self.POPULATION_PRESSURE_FORCE * (n_nearby - 3) * self._era_speed
            # Push away from crowd center
            clat = np.mean([a.lat for a in nearby])
            clng = np.mean([a.lng for a in nearby])
            dlat = self.lat - clat
            dlng = self.lng - clng
            dist = np.sqrt(dlat**2 + dlng**2) + 1e-8
            self.vlat += (dlat / dist) * pressure
            self.vlng += (dlng / dist) * pressure

        # Separation from immediate neighbors
        self._apply_separation(nearby)

        # Apply physics with era-scaled speed
        self._apply_physics(world)

        # World model training is handled centrally in World.step()
        # (shared model trains on aggregated experience from all agents)

        # Keep vital stats within their declared ranges so downstream consumers
        # (world-model observation in [0,1], geopolitics wealth aggregates) never
        # see negative or out-of-range values. Energy is already >0 here because
        # the death check above handled energy<=0.
        self.energy = float(np.clip(self.energy, 0.0, 100.0))
        self.health = float(np.clip(self.health, 0.0, 100.0))
        self.happiness = float(np.clip(self.happiness, 0.0, 100.0))
        self.wealth = max(0.0, self.wealth)

        return {"event": "action", "agent_id": self.id, "agent_name": self.name,
                "action": self.current_action, "outcome": outcome}

    def _update_relationship(self, other_id: int, delta: float):
        current = self.relationships.get(other_id, 0.0)
        self.relationships[other_id] = np.clip(current + delta, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Serialization for UI
    # ------------------------------------------------------------------

    def _store_dialogue(self, action_type: str, partner, result: dict):
        """Store dialogue for UI display and memory."""
        self.last_dialogue = (result.get("speech_text") or
                              result.get("proposer_text") or
                              result.get("agent_text") or
                              result.get("lesson_text"))
        entry = {
            "tick": self.age,
            "type": action_type,
            "partner": partner.name if partner else "audience",
            "text": (self.last_dialogue or "")[:200],
        }
        self.dialogue_history.append(entry)
        if len(self.dialogue_history) > 10:
            self.dialogue_history.pop(0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "alive": self.alive,
            "lat": round(self.lat, 4),
            "lng": round(self.lng, 4),
            "vlat": round(self.vlat, 4),
            "vlng": round(self.vlng, 4),
            "age": self.age,
            "generation": self.generation,
            "energy": round(self.energy, 1),
            "health": round(self.health, 1),
            "wealth": round(self.wealth, 1),
            "happiness": round(self.happiness, 1),
            "traits": {k: round(v, 3) for k, v in self.traits.items()},
            "top_skills": [(s, round(v, 3)) for s, v in self.skills.get_top_skills(3)],
            "current_action": self.current_action,
            "current_goal": self.current_goal,
            "children": self.children_count,
            "business_id": self.business_id,
            "relationships_count": len(self.relationships),
            "memory_summary": self.memory.get_summary(),
            "world_model": self.world_model.get_world_understanding(),
            "last_dialogue": self.last_dialogue[:200] if self.last_dialogue else None,
            "divine_trust": round(self.divine_trust, 3),
            "dialogue_count": len(self.dialogue_history),
        }
