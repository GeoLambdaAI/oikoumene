"""
Actions mixin for Agent (agents.py): execute_action and the _action_* handlers.

Expects on the host class: the attributes _last_observation, business_id,
children_count, current_action, energy, happiness, health, home_lat, home_lng, id, lat,
lng, memory, relationships, reproduction_cooldown, rng, skills, traits, vlat, vlng,
wealth and world_model; the class constants LAT_MAX, LAT_MIN, REPRODUCE_COOLDOWN_YEARS
and REPRODUCE_MIN_AGE_YEARS; and the methods _store_dialogue(), _update_relationship()
and _yrs_to_ticks(). _action_reproduce creates Agent instances (imported in the method).
"""

import numpy as np

from oikoumene.agents.traits import TRAIT_NAMES, _wrap_longitude


class ActionsMixin:
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

        # Store experience for world model training (not in ablation modes)
        if hasattr(self, '_last_observation') and getattr(world, "world_model_mode", "jepa") == "jepa":
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

        from oikoumene.agents.agents import Agent
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
        from oikoumene.core.earth import is_land
        # Pick a random direction biased away from current problems
        # Move a large distance
        angle = self.rng.uniform(0, 2 * np.pi)
        move_dist = 2.0 + 3.0 * self.traits["curiosity"]  # 2-5 degrees
        # Clamp latitude to the habitable band and WRAP longitude at the
        # antimeridian, so the probe (and the resulting position) is always a
        # valid coordinate. Without this, new_lng could exceed +/-180 and
        # is_land would silently read a clamped, wrong grid cell near the edge.
        new_lat = float(np.clip(self.lat + np.sin(angle) * move_dist,
                                self.LAT_MIN, self.LAT_MAX))
        new_lng = _wrap_longitude(self.lng + np.cos(angle) * move_dist)

        # Ensure we land on land
        attempts = 0
        while not is_land(new_lat, new_lng) and attempts < 8:
            angle += np.pi / 4
            new_lat = float(np.clip(self.lat + np.sin(angle) * move_dist,
                                    self.LAT_MIN, self.LAT_MAX))
            new_lng = _wrap_longitude(self.lng + np.cos(angle) * move_dist)
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
