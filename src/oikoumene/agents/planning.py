"""
Planning mixin for Agent (agents.py): observation, JEPA + CEM decision making, needs
evaluation and action decoding. Defines PLAN_INTERVAL.

Expects on the host class: the attributes _cached_behavior, _plan_tick, age,
current_goal, energy, happiness, health, memory, reproduction_cooldown, traits, vlat,
vlng, wealth and world_model; the class constants GOVERN_MIN_AGE_YEARS,
LIFESPAN_NORM_YEARS, MAX_SPEED and REPRODUCE_MIN_AGE_YEARS; and the methods
_yrs_to_ticks() and _apply_goal_movement() (MovementMixin).
"""

import numpy as np

from oikoumene.agents.traits import TRAIT_NAMES


class PlanningMixin:
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

        # Velocity [18-19] — normalize by the ERA-SCALED max speed, not the base
        # MAX_SPEED. Movement is clamped to MAX_SPEED * era_mult (era_mult up to
        # 20 in fast paleo eras); dividing by the base alone let these two
        # features reach +/-20 while every other feature is ~[0,1], dominating
        # the shared JEPA encoder input.
        era_mult = getattr(self, "_era_speed", 1.0)
        speed_norm = self.MAX_SPEED * max(era_mult, 1e-6)
        obs[18] = self.vlat / speed_norm
        obs[19] = self.vlng / speed_norm

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
            if getattr(world, "world_model_mode", "jepa") == "jepa":
                goal, action = self._plan_with_world_model(observation)
            else:
                # Ablation: same goal selection, action from World.ablation_action()
                goal = self._select_goal()
                action = world.ablation_action()

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

    def _select_goal(self) -> str:
        """Evaluate needs (priority-based goal selection) and set current_goal."""
        needs = self._evaluate_needs()
        goal = max(needs, key=needs.get)
        self.current_goal = goal
        return goal

    def _plan_with_world_model(self, observation: np.ndarray) -> tuple[str, np.ndarray]:
        """JEPA encode + CEM plan; returns (goal, first action of the plan)."""
        z = self.world_model.encode(observation)
        goal = self._select_goal()
        # Generate goal observation
        goal_obs = self._goal_to_observation(goal, observation)
        z_goal = self.world_model.encode(goal_obs)
        # Plan with CEM
        action_seq = self.world_model.planner.plan(z, z_goal)
        return goal, action_seq[0]

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
        govern_age_ticks = self._yrs_to_ticks(self.GOVERN_MIN_AGE_YEARS)
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
