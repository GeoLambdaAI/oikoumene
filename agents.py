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
from enum import Enum

from world_model import JEPAWorldModel

# Shared world model — set by World.__init__(), all agents reference it
_shared_world_model = None


# ============================================================================
# Agent Traits & Genetics
# ============================================================================

class TraitType(Enum):
    INTELLIGENCE = "intelligence"
    CREATIVITY = "creativity"
    SOCIABILITY = "sociability"
    AMBITION = "ambition"
    RISK_TOLERANCE = "risk_tolerance"
    COOPERATION = "cooperation"
    RESILIENCE = "resilience"
    CURIOSITY = "curiosity"


TRAIT_NAMES = [t.value for t in TraitType]


def _wrap_longitude(lng: float) -> float:
    """Wrap a longitude (degrees) to the half-open interval [-180, 180).

    Used so movement and coastline land-checks treat the dateline as continuous
    (a globe) instead of a wall, and so probes near +/-180 read the correct grid
    cell rather than the opposite edge.
    """
    return ((lng + 180.0) % 360.0) - 180.0


def generate_traits(parent_a: Optional[dict] = None,
                    parent_b: Optional[dict] = None,
                    mutation_rate: float = 0.15,
                    rng: Optional[np.random.RandomState] = None) -> dict:
    """Generate traits via crossover + mutation or random initialization.

    ``rng`` should be the world's seeded ``RandomState`` so trait generation is
    reproducible; it falls back to the global ``np.random`` when not supplied.
    """
    if rng is None:
        rng = np.random
    if parent_a is None:
        return {name: np.clip(rng.beta(2, 2), 0.05, 0.95) for name in TRAIT_NAMES}

    traits = {}
    for name in TRAIT_NAMES:
        # Crossover
        if parent_b and rng.random() < 0.5:
            val = parent_b.get(name, 0.5)
        else:
            val = parent_a.get(name, 0.5)
        # Mutation
        if rng.random() < mutation_rate:
            val += rng.normal(0, 0.1)
        traits[name] = float(np.clip(val, 0.05, 0.95))
    return traits


# ============================================================================
# Agent Memory (Autoresearch-inspired experiment log)
# ============================================================================

class AgentMemory:
    """
    Structured memory inspired by Karpathy's autoresearch results.tsv pattern.
    Agents maintain logs of actions and outcomes to avoid repeating failures
    and build on successful strategies.
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.episodic: list[dict] = []       # Specific events
        self.semantic: dict[str, float] = {}  # Learned facts (concept -> value)
        self.procedural: dict[str, dict] = {} # Learned strategies
        self.experiment_log: list[dict] = []  # Autoresearch-style log

    def store_episode(self, event: dict):
        self.episodic.append(event)
        if len(self.episodic) > self.capacity:
            self.episodic.pop(0)

    def store_experiment(self, hypothesis: str, action: str,
                         outcome: float, kept: bool, description: str):
        """Autoresearch-style structured logging of experiments."""
        self.experiment_log.append({
            "hypothesis": hypothesis,
            "action": action,
            "outcome": outcome,
            "kept": kept,
            "description": description
        })
        if len(self.experiment_log) > self.capacity:
            self.experiment_log.pop(0)

    def learn_fact(self, concept: str, value: float):
        """Update semantic knowledge with exponential moving average."""
        if concept in self.semantic:
            self.semantic[concept] = 0.7 * self.semantic[concept] + 0.3 * value
        else:
            self.semantic[concept] = value

    def learn_strategy(self, name: str, params: dict, effectiveness: float):
        """Store or update a behavioral strategy."""
        if name in self.procedural:
            old = self.procedural[name]
            old["uses"] = old.get("uses", 0) + 1
            old["effectiveness"] = 0.8 * old.get("effectiveness", 0.5) + 0.2 * effectiveness
            old["params"] = params
        else:
            self.procedural[name] = {
                "params": params,
                "effectiveness": effectiveness,
                "uses": 1
            }

    def get_best_strategy(self, context: str) -> Optional[dict]:
        """Find the most effective strategy for a context."""
        best = None
        best_score = -1
        for name, strategy in self.procedural.items():
            if context in name and strategy["effectiveness"] > best_score:
                best = strategy
                best_score = strategy["effectiveness"]
        return best

    def get_summary(self) -> dict:
        return {
            "episodes": len(self.episodic),
            "facts_known": len(self.semantic),
            "strategies": len(self.procedural),
            "experiments": len(self.experiment_log)
        }


# ============================================================================
# Agent Skills
# ============================================================================

class SkillSet:
    """Learnable skills that improve with practice."""

    SKILL_TYPES = [
        "farming", "mining", "crafting", "trading", "building",
        "research", "leadership", "diplomacy", "combat", "medicine"
    ]

    def __init__(self, initial_skills: Optional[dict] = None,
                 rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random
        if initial_skills:
            self.skills = {s: initial_skills.get(s, 0.01) for s in self.SKILL_TYPES}
        else:
            self.skills = {s: max(0.01, rng.exponential(0.1)) for s in self.SKILL_TYPES}

    def practice(self, skill_name: str, intensity: float = 1.0,
                 talent_modifier: float = 1.0):
        """Improve a skill through practice. Logarithmic growth."""
        if skill_name in self.skills:
            current = self.skills[skill_name]
            gain = 0.01 * intensity * talent_modifier / (1 + current)
            self.skills[skill_name] = min(1.0, current + gain)

    def get_level(self, skill_name: str) -> float:
        return self.skills.get(skill_name, 0.0)

    def get_top_skills(self, n: int = 3) -> list[tuple[str, float]]:
        sorted_skills = sorted(self.skills.items(), key=lambda x: x[1], reverse=True)
        return sorted_skills[:n]

    def top_skill(self) -> tuple[str, float]:
        """Return the single highest (name, level) skill, or a safe default.

        Guards the common ``get_top_skills(1)[0]`` pattern against an empty skill
        set (e.g. if skills ever become configurable / can be empty)."""
        top = self.get_top_skills(1)
        return top[0] if top else (self.SKILL_TYPES[0], 0.0)

    def inherit(self, parent_skills: 'SkillSet', inheritance_rate: float = 0.3) -> None:
        """Inherit partial skill aptitude from a parent."""
        for skill in self.SKILL_TYPES:
            inherited = parent_skills.get_level(skill) * inheritance_rate
            self.skills[skill] = max(self.skills[skill], inherited)


# ============================================================================
# Autonomous Agent
# ============================================================================

class Agent:
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

    def __init__(self, x: float, y: float,
                 parent_a: Optional['Agent'] = None,
                 parent_b: Optional['Agent'] = None,
                 rng: Optional[np.random.RandomState] = None):
        # Seeded RNG for full reproducibility. World passes its own RandomState so
        # that a fixed World(seed=...) yields identical agent trajectories; falls
        # back to the global np.random only when constructed without a world.
        self.rng = rng if rng is not None else np.random

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
    # Physics-Based Movement
    # ------------------------------------------------------------------

    def _apply_physics(self, world):
        """Apply velocity, friction, terrain effects, and boundary handling."""
        from earth import is_land

        # Terrain friction modifier
        r, c = world.resources.get_cell(self.lat, self.lng)
        terrain = world.terrain[min(r, world.terrain.shape[0]-1),
                                min(c, world.terrain.shape[1]-1)]
        terrain_friction = {
            0: 0.3,   # Ocean - very slow
            1: 0.90,  # Plains - easy movement
            2: 0.82,  # Forest - some resistance
            3: 0.70,  # Mountains - hard
            4: 0.85,  # Desert - sandy
            5: 0.75,  # Tundra - cold resistance
        }.get(terrain, 0.85)

        # Apply friction
        self.vlat *= terrain_friction
        self.vlng *= terrain_friction

        # Clamp speed — scaled by era (early eras = centuries of migration per tick)
        speed = np.sqrt(self.vlat**2 + self.vlng**2)
        era_mult = getattr(self, '_era_speed', 1.0)
        max_speed = self.MAX_SPEED * era_mult * (0.5 + 0.5 * self.energy / 100.0)
        if speed > max_speed:
            scale = max_speed / (speed + 1e-8)
            self.vlat *= scale
            self.vlng *= scale

        # Apply velocity. Latitude is clamped to the habitable band; longitude
        # wraps at the dateline so land checks near +/-180 use correctly wrapped
        # coordinates instead of reading the wrong grid edge.
        new_lat = float(np.clip(self.lat + self.vlat, self.LAT_MIN, self.LAT_MAX))
        new_lng = _wrap_longitude(self.lng + self.vlng)

        # Ocean avoidance - check if next position is ocean
        if not is_land(new_lat, new_lng):
            # Bounce back and steer away
            self.vlat *= -0.5
            self.vlng *= -0.5
            # Try to find land direction (check 8 directions). Probe coordinates
            # are wrapped/clamped, and the steering offset uses the shortest arc
            # in longitude so a probe across the dateline does not yield a huge
            # spurious vector.
            best_dist = float('inf')
            best_dlat, best_dlng = 0.0, 0.0
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                check_lat = float(np.clip(self.lat + np.sin(angle) * 2.0,
                                          self.LAT_MIN, self.LAT_MAX))
                check_lng = _wrap_longitude(self.lng + np.cos(angle) * 2.0)
                if is_land(check_lat, check_lng):
                    dlat = check_lat - self.lat
                    dlng = _wrap_longitude(check_lng - self.lng)
                    dist = dlat**2 + dlng**2
                    if dist < best_dist:
                        best_dist = dist
                        best_dlat = dlat
                        best_dlng = dlng
            if best_dist < float('inf'):
                d = np.sqrt(best_dlat**2 + best_dlng**2) + 1e-8
                self.vlat += best_dlat / d * 0.03
                self.vlng += best_dlng / d * 0.03
        else:
            self.lat = new_lat
            self.lng = new_lng

        # World boundary: clamp latitude to the habitable band, wrap longitude.
        self.lat = float(np.clip(self.lat, self.LAT_MIN, self.LAT_MAX))
        self.lng = _wrap_longitude(self.lng)

        # Update heading from velocity
        speed = np.sqrt(self.vlat**2 + self.vlng**2)
        if speed > 0.001:
            self.heading_x = self.vlng / speed
            self.heading_y = self.vlat / speed

    def _move_toward(self, tlat: float, tlng: float, strength: float = 1.0):
        """Apply force toward a target position (lat/lng), era-scaled."""
        dlat = tlat - self.lat
        dlng = tlng - self.lng
        dist = np.sqrt(dlat**2 + dlng**2) + 1e-8
        era_mult = getattr(self, '_era_speed', 1.0)
        force = min(strength * era_mult, dist * 0.1 * era_mult)
        self.vlat += (dlat / dist) * force
        self.vlng += (dlng / dist) * force

    def _wander(self):
        """Apply random wandering force for exploration."""
        angle_noise = self.rng.normal(0, 0.3)
        cos_a, sin_a = np.cos(angle_noise), np.sin(angle_noise)
        new_hx = self.heading_x * cos_a - self.heading_y * sin_a
        new_hy = self.heading_x * sin_a + self.heading_y * cos_a
        self.heading_x, self.heading_y = new_hx, new_hy

        era_mult = getattr(self, '_era_speed', 1.0)
        strength = self.WANDER_STRENGTH * (0.5 + 0.5 * self.traits["curiosity"]) * era_mult
        self.vlng += self.heading_x * strength
        self.vlat += self.heading_y * strength

    def _apply_separation(self, nearby_agents: list['Agent']):
        """Keep minimum distance from other agents."""
        for other in nearby_agents:
            if other.id == self.id or not other.alive:
                continue
            dlat = self.lat - other.lat
            dlng = self.lng - other.lng
            dist = np.sqrt(dlat**2 + dlng**2) + 1e-8
            if dist < self.SEPARATION_RADIUS:
                push = self.SEPARATION_FORCE * (1.0 - dist / self.SEPARATION_RADIUS)
                self.vlat += (dlat / dist) * push
                self.vlng += (dlng / dist) * push

    # ------------------------------------------------------------------
    # Observation & Encoding
    # ------------------------------------------------------------------

    def observe(self, world_state: dict) -> np.ndarray:
        """Build observation vector from local world state (40-dim)."""
        obs = np.zeros(40)

        # Self state [0-4]
        obs[0] = self.energy / 100.0
        obs[1] = self.health / 100.0
        obs[2] = self.wealth / 100.0
        obs[3] = self.happiness / 100.0
        obs[4] = self.age / max(1, self._yrs_to_ticks(self.LIFESPAN_NORM_YEARS))

        # Local resources [5-8]
        obs[5] = world_state.get("local_food", 0) / 100.0
        obs[6] = world_state.get("local_minerals", 0) / 100.0
        obs[7] = world_state.get("local_wood", 0) / 100.0
        obs[8] = world_state.get("local_water", 0) / 100.0

        # Population density [9]
        obs[9] = min(1.0, world_state.get("nearby_agents", 0) / 20.0)

        # Economic signals [10-12]
        obs[10] = world_state.get("local_demand", 0)
        obs[11] = world_state.get("local_supply", 0)
        obs[12] = world_state.get("avg_wealth", 0) / 100.0

        # Terrain [13-15]
        obs[13] = world_state.get("terrain_type", 0) / 5.0
        obs[14] = world_state.get("fertility", 0)
        obs[15] = world_state.get("elevation", 0)

        # Social signals [16-17]
        obs[16] = world_state.get("social_trust", 0.5)
        obs[17] = world_state.get("governance_stability", 0.5)

        # Velocity [18-19]
        obs[18] = self.vlat / self.MAX_SPEED
        obs[19] = self.vlng / self.MAX_SPEED

        # Traits encoded [20-27]
        for i, trait_name in enumerate(TRAIT_NAMES):
            if 20 + i < 28:
                obs[20 + i] = self.traits[trait_name]

        # Resource direction hints [28-29]
        best_food_dir = self.memory.semantic.get("best_food_direction", 0)
        obs[28] = np.sin(best_food_dir)
        obs[29] = np.cos(best_food_dir)

        # Nearest agent direction [30-31]
        obs[30] = world_state.get("nearest_agent_dx", 0)
        obs[31] = world_state.get("nearest_agent_dy", 0)

        # === Macro signals (from bridge) [32-39] ===
        obs[32] = world_state.get("temperature_anomaly", 0) / 5.0
        obs[33] = world_state.get("co2_level", 425) / 1000.0
        obs[34] = world_state.get("social_tension", 0.25)
        obs[35] = world_state.get("pollution_level", 0.3)
        obs[36] = world_state.get("resource_scarcity", 0)
        obs[37] = world_state.get("conflict_nearby", 0)
        obs[38] = world_state.get("nation_tech_level", 1.0) / 3.0
        obs[39] = world_state.get("trade_access", 0.5)

        return obs

    # ------------------------------------------------------------------
    # Decision Making (JEPA + CEM Planning)
    # ------------------------------------------------------------------

    # How often agents re-plan (in ticks). Between plans, they reuse cached behavior.
    PLAN_INTERVAL = 3

    def decide_action(self, observation: np.ndarray, world) -> dict:
        """
        Autonomous decision-making with tick-skipping.

        Full JEPA encode + CEM plan runs every PLAN_INTERVAL ticks.
        In between, the agent reuses its cached goal and applies
        movement forces without the expensive CEM step.

        This gives a ~3x throughput improvement with minimal
        behavior change (agents still move and act every tick).
        """
        # Check if we should re-plan or reuse cache
        should_plan = (
            self._cached_behavior is None or
            (self.age - self._plan_tick) >= self.PLAN_INTERVAL or
            self.energy < 20  # Emergency: always re-plan when low energy
        )

        if should_plan:
            z = self.world_model.encode(observation)

            # Evaluate needs (priority-based goal selection)
            needs = self._evaluate_needs()
            goal = max(needs, key=needs.get)
            self.current_goal = goal

            # Generate goal observation
            goal_obs = self._goal_to_observation(goal, observation)
            z_goal = self.world_model.encode(goal_obs)

            # Plan with CEM
            action_seq = self.world_model.planner.plan(z, z_goal)
            action = action_seq[0]

            # Cache the plan
            behavior = self._decode_action(action, goal)
            self._cached_behavior = behavior
            self._plan_tick = self.age
        else:
            # Reuse cached behavior but still apply movement
            behavior = self._cached_behavior
            goal = self.current_goal
            action = behavior.get("raw_action", np.zeros(8))

        # Apply goal-directed movement forces (every tick, even without re-plan)
        self._apply_goal_movement(goal, action, world)

        return behavior

    def _apply_goal_movement(self, goal: str, action_vector: np.ndarray, world):
        """Apply movement forces based on the current goal."""
        # Always wander a bit
        self._wander()

        # CEM-derived movement direction
        cem_dlat = float(np.tanh(action_vector[0]))
        cem_dlng = float(np.tanh(action_vector[1]))
        cem_strength = 0.015 + 0.015 * float(np.clip(abs(action_vector[2]), 0, 1))
        self.vlat += cem_dlat * cem_strength
        self.vlng += cem_dlng * cem_strength

        # Goal-specific movement
        if goal == "eat":
            best = self._find_resource_direction(world, "food")
            if best is not None:
                self._move_toward(best[0], best[1], self.GOAL_FORCE * 1.5)
                self.goal_lat, self.goal_lng = best

        elif goal in ("socialize", "reproduce", "trade"):
            nearby = world.get_nearby_agents(self.lat, self.lng, 10.0)
            others = [a for a in nearby if a.id != self.id and a.alive]
            if others:
                if goal == "reproduce":
                    min_age = self._yrs_to_ticks(self.REPRODUCE_MIN_AGE_YEARS)
                    candidates = [a for a in others if a.energy > 30 and a.age > min_age
                                  and a.reproduction_cooldown <= 0]
                    if candidates:
                        target = max(candidates,
                                     key=lambda a: self.relationships.get(a.id, 0))
                    else:
                        target = min(others,
                                     key=lambda a: (a.lat-self.lat)**2 + (a.lng-self.lng)**2)
                else:
                    target = min(others,
                                 key=lambda a: (a.lat-self.lat)**2 + (a.lng-self.lng)**2)
                self._move_toward(target.lat, target.lng, self.GOAL_FORCE)
                self.goal_lat, self.goal_lng = target.lat, target.lng

        elif goal == "explore":
            if self.rng.random() < 0.05:
                angle = self.rng.uniform(0, 2 * np.pi)
                self.heading_x = np.cos(angle)
                self.heading_y = np.sin(angle)
            self.vlng += self.heading_x * self.GOAL_FORCE * self.traits["curiosity"]
            self.vlat += self.heading_y * self.GOAL_FORCE * self.traits["curiosity"]

        elif goal in ("work", "build_business"):
            if self.home_lat is not None:
                self._move_toward(self.home_lat, self.home_lng, self.GOAL_FORCE * 0.5)
            elif self.business_id is not None:
                for biz in world.businesses:
                    if biz.id == self.business_id:
                        self._move_toward(biz.lat, biz.lng, self.GOAL_FORCE)
                        break

        elif goal == "heal":
            self.vlat *= 0.5
            self.vlng *= 0.5

        elif goal == "govern":
            settlement = world._find_nearest_settlement(self.lat, self.lng)
            if settlement:
                self._move_toward(settlement.lat, settlement.lng, self.GOAL_FORCE)

        elif goal == "migrate":
            # Strong movement in current heading (fleeing)
            self.vlng += self.heading_x * self.GOAL_FORCE * 2.0
            self.vlat += self.heading_y * self.GOAL_FORCE * 2.0

    def _find_resource_direction(self, world, resource_type: str):
        """Scan nearby cells for best resource location."""
        best_val = 0
        best_pos = None
        cr, cc = world.resources.get_cell(self.lat, self.lng)
        csd = world.resources.cell_size_deg

        # Scan 5x5 area around agent
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r = cr + dr
                c = cc + dc
                if 0 <= r < world.resources.rows and 0 <= c < world.resources.cols:
                    layer = getattr(world.resources, resource_type)
                    val = layer[r, c]
                    if val > best_val:
                        best_val = val
                        # Convert grid cell back to lat/lng
                        cell_lat = world.resources.lat_max - (r + 0.5) * csd
                        cell_lng = world.resources.lng_min + (c + 0.5) * csd
                        best_pos = (cell_lat, cell_lng)

        if best_val > 1.0:
            if best_pos:
                angle = np.arctan2(best_pos[0] - self.lat, best_pos[1] - self.lng)
                self.memory.learn_fact("best_food_direction", angle)
            return best_pos
        return None

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

    def _evaluate_needs(self) -> dict:
        """Maslow-inspired needs hierarchy."""
        needs = {}

        # Survival (highest priority when threatened)
        needs["eat"] = max(0, (70 - self.energy) / 70) * 2.0
        needs["heal"] = max(0, (60 - self.health) / 60) * 1.8

        # Economic
        needs["work"] = max(0, (50 - self.wealth) / 50) * 1.2
        needs["trade"] = 0.3 * self.traits["ambition"] if self.wealth > 20 else 0
        needs["build_business"] = (0.5 * self.traits["ambition"] *
                                    self.traits["creativity"] if self.wealth > 80 else 0)

        # Social
        needs["socialize"] = (0.4 * self.traits["sociability"] *
                              max(0, (60 - self.happiness) / 60))
        repro_age_ticks = self._yrs_to_ticks(self.REPRODUCE_MIN_AGE_YEARS)
        needs["reproduce"] = (0.6 if self.energy > 60 and self.age > repro_age_ticks
                              and self.reproduction_cooldown <= 0
                              and self.wealth > 30 else 0)

        # Exploration / Growth
        needs["explore"] = 0.3 * self.traits["curiosity"]
        needs["research"] = 0.2 * self.traits["curiosity"] * self.traits["intelligence"]

        # Leadership
        govern_age_ticks = self._yrs_to_ticks(self.SOCIALIZE_MIN_AGE_YEARS)
        needs["govern"] = (0.3 * self.traits["ambition"] *
                           self.traits["sociability"] if self.age > govern_age_ticks else 0)

        # Macro-driven needs (respond to global conditions)
        # Migrate: flee when local conditions are bad (conflict, resource scarcity)
        conflict = self.memory.semantic.get("conflict_nearby", 0)
        scarcity = self.memory.semantic.get("resource_scarcity", 0)
        needs["migrate"] = max(0, (conflict * 0.5 + scarcity * 0.3 - 0.3)) * (
            1.0 - self.traits["resilience"]
        )

        return needs

    def _goal_to_observation(self, goal: str, current_obs: np.ndarray) -> np.ndarray:
        """Create a target observation representing the desired goal state."""
        target = current_obs.copy()

        goal_deltas = {
            "eat":            {0: 0.3, 3: 0.1},
            "heal":           {1: 0.3, 3: 0.1},
            "work":           {2: 0.2, 0: -0.05},
            "trade":          {2: 0.15, 10: 0.1, 11: 0.1},
            "build_business": {2: 0.3, 10: 0.2},
            "socialize":      {3: 0.3, 16: 0.1},
            "reproduce":      {0: -0.1, 3: 0.2},
            "explore":        {13: 0.2, 14: 0.1},
            "research":       {3: 0.1},
            "govern":         {17: 0.2, 16: 0.1},
        }

        deltas = goal_deltas.get(goal, {})
        for idx, delta in deltas.items():
            target[idx] = np.clip(target[idx] + delta, 0, 1)

        return target

    def _decode_action(self, action_vector: np.ndarray, goal: str) -> dict:
        """Decode continuous action vector into discrete behavior."""
        intensity = float(np.clip(np.abs(action_vector[2]), 0.1, 1.0))

        return {
            "type": goal,
            "intensity": intensity,
            "target_id": None,
            "raw_action": action_vector,
        }

    # ------------------------------------------------------------------
    # Action Execution
    # ------------------------------------------------------------------

    def execute_action(self, behavior: dict, world) -> dict:
        """Execute the decided behavior and return the outcome."""
        action_type = behavior["type"]
        self.current_action = action_type
        outcome = {"success": False, "reward": 0.0, "description": ""}

        if action_type == "eat":
            outcome = self._action_eat(world, behavior["intensity"])
        elif action_type == "work":
            outcome = self._action_work(world, behavior["intensity"])
        elif action_type == "trade":
            outcome = self._action_trade(world, behavior)
        elif action_type == "build_business":
            outcome = self._action_build_business(world)
        elif action_type == "socialize":
            outcome = self._action_socialize(world)
        elif action_type == "reproduce":
            outcome = self._action_reproduce(world)
        elif action_type == "explore":
            outcome = self._action_explore()
        elif action_type == "research":
            outcome = self._action_research(world)
        elif action_type == "heal":
            outcome = self._action_heal(world)
        elif action_type == "govern":
            outcome = self._action_govern(world)
        elif action_type == "migrate":
            outcome = self._action_migrate(world)

        # Store experience for world model training
        if hasattr(self, '_last_observation'):
            obs = self._last_observation
            next_obs = obs.copy()
            next_obs[0] = self.energy / 100.0
            next_obs[1] = self.health / 100.0
            next_obs[2] = self.wealth / 100.0
            next_obs[3] = self.happiness / 100.0
            action = behavior.get("raw_action", np.zeros(8))
            self.world_model.store_experience(obs, action, next_obs)

        # Log experiment (autoresearch style)
        self.memory.store_experiment(
            hypothesis=f"Action '{action_type}' will improve my situation",
            action=action_type,
            outcome=outcome["reward"],
            kept=outcome["success"],
            description=outcome["description"]
        )

        return outcome

    def _action_eat(self, world, intensity: float) -> dict:
        food = world.harvest_resource(self.lat, self.lng, "food", intensity * 10)
        if food > 0:
            self.energy = min(100, self.energy + food * 2)
            self.skills.practice("farming", intensity, self.traits["intelligence"])
            return {"success": True, "reward": food * 0.1,
                    "description": f"Ate {food:.1f} food"}
        return {"success": False, "reward": -0.1, "description": "No food found"}

    def _action_work(self, world, intensity: float) -> dict:
        top_skill = self.skills.top_skill()
        skill_name, skill_level = top_skill
        earnings = intensity * skill_level * 5 * (1 + self.traits["ambition"])
        self.wealth += earnings
        self.energy -= intensity * 3
        self.skills.practice(skill_name, intensity, self.traits["intelligence"])
        self.memory.learn_fact(f"work_{skill_name}_profit", earnings)
        # Set home near work location
        if self.home_lat is None:
            self.home_lat = self.lat
            self.home_lng = self.lng
        return {"success": True, "reward": earnings * 0.05,
                "description": f"Worked as {skill_name}, earned {earnings:.1f}"}

    def _action_trade(self, world, behavior: dict) -> dict:
        nearby = world.get_nearby_agents(self.lat, self.lng, radius=3.0)
        nearby = [a for a in nearby if a.id != self.id]
        if not nearby:
            return {"success": False, "reward": -0.05, "description": "No trading partners"}

        partner = nearby[self.rng.randint(len(nearby))]
        skill_diff = (self.skills.get_level("trading") -
                      partner.skills.get_level("trading"))
        base_trade_value = max(0.5, 2 + skill_diff * 3) * behavior["intensity"]

        # LLM negotiation modifies trade outcome
        llm = getattr(world, 'llm', None)
        modifier = 1.0
        if llm and llm.can_call():
            result = llm.generate_trade_negotiation(
                agent=self, partner=partner,
                current_trade_value=base_trade_value,
                context={"relationship": self.relationships.get(partner.id, 0),
                          "agent_wealth": self.wealth, "partner_wealth": partner.wealth}
            )
            modifier = result.get("modifier", 1.0)
            self._store_dialogue("trade", partner, result)

        trade_value = base_trade_value * float(np.clip(modifier, 0.3, 2.5))

        # Conservative exchange: the wealth the initiator earns is transferred from
        # the partner (bounded by what the partner can pay), so trade never creates
        # or destroys wealth and no balance goes negative. Both parties still gain
        # relationship affinity from a successful exchange.
        transfer = min(trade_value, max(0.0, partner.wealth))
        self.wealth += transfer
        partner.wealth -= transfer
        self.skills.practice("trading", 1.0, self.traits["intelligence"])

        self._update_relationship(partner.id, 0.1)
        partner._update_relationship(self.id, 0.05)

        return {"success": True, "reward": transfer * 0.1,
                "description": f"Traded with {partner.name} for {transfer:.1f}"}

    def _action_build_business(self, world) -> dict:
        if self.wealth < 50:
            return {"success": False, "reward": -0.1, "description": "Not enough capital"}

        top_skill = self.skills.top_skill()
        business_type = top_skill[0]
        investment = min(self.wealth * 0.4, 100)
        self.wealth -= investment

        biz = world.create_business(
            owner_id=self.id,
            lat=self.lat, lng=self.lng,
            business_type=business_type,
            capital=investment
        )
        self.business_id = biz["id"]
        self.skills.practice("leadership", 1.0, self.traits["ambition"])
        self.memory.learn_strategy("business", {"type": business_type, "investment": investment}, 0.5)
        return {"success": True, "reward": 0.5,
                "description": f"Founded {business_type} business (capital: {investment:.0f})"}

    def _action_socialize(self, world) -> dict:
        nearby = world.get_nearby_agents(self.lat, self.lng, radius=5.0)
        nearby = [a for a in nearby if a.id != self.id]
        if not nearby:
            return {"success": False, "reward": -0.05, "description": "Nobody nearby"}

        partner = nearby[self.rng.randint(len(nearby))]
        compatibility = 1.0 - np.mean([
            abs(self.traits[t] - partner.traits[t]) for t in TRAIT_NAMES
        ])

        # LLM social dialogue modifies relationship and happiness gains
        llm = getattr(world, 'llm', None)
        rel_mod = 1.0
        happy_mod = 1.0
        if llm and llm.can_call():
            result = llm.generate_social_dialogue(
                agent=self, partner=partner, compatibility=compatibility,
                context={"relationship": self.relationships.get(partner.id, 0)}
            )
            rel_mod = result.get("relationship_modifier", 1.0)
            happy_mod = result.get("happiness_modifier", 1.0)
            self._store_dialogue("socialize", partner, result)

        self.happiness = min(100, self.happiness + compatibility * 5 * happy_mod)
        self._update_relationship(partner.id, compatibility * 0.2 * rel_mod)
        partner._update_relationship(self.id, compatibility * 0.15 * rel_mod)
        self.skills.practice("diplomacy", 0.5, self.traits["sociability"])
        return {"success": True, "reward": compatibility * 0.2,
                "description": f"Socialized with {partner.name} (compat: {compatibility:.2f})"}

    def _action_reproduce(self, world) -> dict:
        if self.energy < 40 or self.reproduction_cooldown > 0 or self.wealth < 20:
            return {"success": False, "reward": 0, "description": "Cannot reproduce now"}

        nearby = world.get_nearby_agents(self.lat, self.lng, radius=3.0)
        min_age = self._yrs_to_ticks(self.REPRODUCE_MIN_AGE_YEARS)
        candidates = [a for a in nearby if a.id != self.id and a.alive
                       and a.energy > 30 and a.age > min_age
                       and a.reproduction_cooldown <= 0 and a.wealth >= 10]

        if not candidates:
            return {"success": False, "reward": -0.05, "description": "No suitable partner"}

        partner = max(candidates, key=lambda a: self.relationships.get(a.id, 0))

        offspring_lat = (self.lat + partner.lat) / 2 + self.rng.normal(0, 0.2)
        offspring_lng = (self.lng + partner.lng) / 2 + self.rng.normal(0, 0.2)

        child = Agent(offspring_lat, offspring_lng, parent_a=self, parent_b=partner, rng=self.rng)
        world.add_agent(child)

        # Both parents pay an energy + wealth cost, kept non-negative; the partner is
        # debited symmetrically so reproduction never creates wealth from nothing.
        self.energy = max(0.0, self.energy - 25)
        self.wealth = max(0.0, self.wealth - 10)
        partner.energy = max(0.0, partner.energy - 15)
        partner.wealth = max(0.0, partner.wealth - 10)
        cooldown_ticks = self._yrs_to_ticks(self.REPRODUCE_COOLDOWN_YEARS)
        self.reproduction_cooldown = cooldown_ticks
        partner.reproduction_cooldown = cooldown_ticks
        self.children_count += 1
        partner.children_count += 1

        self.happiness = min(100, self.happiness + 15)
        partner.happiness = min(100, partner.happiness + 10)

        return {"success": True, "reward": 1.0,
                "description": f"Reproduced with {partner.name} -> {child.name} (gen {child.generation})"}

    def _action_explore(self) -> dict:
        self.energy -= 1.5
        self.skills.practice("research", 0.3, self.traits["curiosity"])
        discovery_chance = self.traits["curiosity"] * self.traits["intelligence"]
        reward = 0.1
        desc = "Explored new territory"
        if self.rng.random() < discovery_chance * 0.3:
            reward = 0.5
            desc = "Made a discovery while exploring!"
            self.happiness = min(100, self.happiness + 5)
        return {"success": True, "reward": reward, "description": desc}

    def _action_research(self, world) -> dict:
        """Autoresearch-inspired: formulate hypothesis, test, learn."""
        topics = ["farming_efficiency", "trading_strategy", "construction",
                  "medicine_knowledge", "social_dynamics"]
        topic = topics[self.rng.randint(len(topics))]

        research_power = (self.traits["intelligence"] * self.traits["curiosity"] *
                          self.skills.get_level("research"))
        success = self.rng.random() < 0.2 + research_power * 0.5

        self.energy -= 3
        self.skills.practice("research", 1.0, self.traits["intelligence"])

        if success:
            skill_mapping = {
                "farming_efficiency": "farming",
                "trading_strategy": "trading",
                "construction": "building",
                "medicine_knowledge": "medicine",
                "social_dynamics": "diplomacy"
            }
            related_skill = skill_mapping.get(topic, "research")
            self.skills.practice(related_skill, 2.0, self.traits["intelligence"])
            self.memory.learn_fact(topic, min(1.0, self.memory.semantic.get(topic, 0) + 0.1))
            self.happiness = min(100, self.happiness + 3)
            return {"success": True, "reward": 0.4,
                    "description": f"Research breakthrough in {topic}!"}

        return {"success": False, "reward": 0.05,
                "description": f"Researched {topic}, no breakthrough yet"}

    def _action_heal(self, world) -> dict:
        heal_amount = 5 + self.skills.get_level("medicine") * 15
        self.health = min(100, self.health + heal_amount)
        self.energy -= 2
        self.skills.practice("medicine", 0.5, self.traits["intelligence"])
        return {"success": True, "reward": heal_amount * 0.02,
                "description": f"Healed for {heal_amount:.1f} HP"}

    def _action_govern(self, world) -> dict:
        nearby = world.get_nearby_agents(self.lat, self.lng, radius=5.0)
        if len(nearby) < 3:
            return {"success": False, "reward": 0, "description": "Not enough people to govern"}

        base_influence = (self.traits["ambition"] * self.traits["sociability"] *
                          self.skills.get_level("leadership"))

        # LLM governance speech modifies influence
        llm = getattr(world, 'llm', None)
        influence_mod = 1.0
        if llm and llm.can_call():
            result = llm.generate_governance_speech(
                agent=self, audience_agents=[a for a in nearby if a.id != self.id],
                context={"social_tension": world.macro.state.social_tension}
            )
            influence_mod = result.get("influence_modifier", 1.0)
            self._store_dialogue("govern", None, result)

        influence = base_influence * float(np.clip(influence_mod, 0.3, 2.0))
        self.skills.practice("leadership", 1.0, self.traits["ambition"])

        for agent in nearby:
            if agent.id != self.id:
                agent._update_relationship(self.id, influence * 0.1)

        self.happiness = min(100, self.happiness + influence * 5)
        self.energy -= 2
        return {"success": True, "reward": influence * 0.3,
                "description": f"Governed {len(nearby)-1} nearby agents (influence: {influence:.2f})"}

    def _action_migrate(self, world) -> dict:
        """Flee current location toward safer territory (climate/conflict migration)."""
        from earth import is_land
        # Pick a random direction biased away from current problems
        # Move a large distance
        angle = self.rng.uniform(0, 2 * np.pi)
        move_dist = 2.0 + 3.0 * self.traits["curiosity"]  # 2-5 degrees
        new_lat = self.lat + np.sin(angle) * move_dist
        new_lng = self.lng + np.cos(angle) * move_dist

        # Ensure we land on land
        attempts = 0
        while not is_land(new_lat, new_lng) and attempts < 8:
            angle += np.pi / 4
            new_lat = self.lat + np.sin(angle) * move_dist
            new_lng = self.lng + np.cos(angle) * move_dist
            attempts += 1

        if is_land(new_lat, new_lng):
            self.lat = new_lat
            self.lng = new_lng
            self.vlat = 0
            self.vlng = 0
            self.home_lat = None  # Reset home
            self.home_lng = None
            self.energy -= 10
            self.happiness -= 5
            return {"success": True, "reward": 0.2,
                    "description": f"Migrated to ({new_lat:.1f}, {new_lng:.1f})"}

        self.energy -= 5
        return {"success": False, "reward": -0.1,
                "description": "Migration failed - no suitable destination"}

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

        # Death check
        if self.energy <= 0 or self.health <= 0:
            self.alive = False
            cause = "starvation" if self.energy <= 0 else "old_age"
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
