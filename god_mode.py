"""
God Mode — Interventional experiment system.

Allows the user to inject messages, trigger events, and modify world
parameters through the UI. Agents react based on personality — skeptical
agents may ignore divine commands while cooperative ones follow.

All interventions are logged with timestamps for scientific reproducibility.
The system is fully optional and togglable.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class GodModeConfig:
    enabled: bool = False
    log_interventions: bool = True


@dataclass
class ActiveEffect:
    """A time-limited world effect (drought, plague, etc.)."""
    effect_type: str
    lat: float
    lng: float
    radius_deg: float
    severity: float        # 0.0-1.0
    remaining_ticks: int
    total_ticks: int
    params: dict = field(default_factory=dict)


class GodMode:
    """
    Divine intervention system for the simulation.

    Intervention types:
    - Message: whisper (1 agent), vision (1 nation), commandment (area broadcast)
    - World: drought, plague, resource_discovery, grant_technology, modify_climate

    All interventions are logged. Active effects tick down and auto-expire.
    """

    def __init__(self, config: Optional[GodModeConfig] = None):
        self.config = config or GodModeConfig()
        self.active_effects: list[ActiveEffect] = []
        self.intervention_log: list[dict] = []

    # ------------------------------------------------------------------
    # Message Interventions
    # ------------------------------------------------------------------

    def whisper_to_agent(self, world, agent_id: int, message: str) -> dict:
        """Send a private divine message to one agent."""
        target = None
        for a in world.agents:
            if a.id == agent_id and a.alive:
                target = a
                break

        if target is None:
            return {"success": False, "error": "Agent not found or dead"}

        # Inject into agent's divine message queue
        if not hasattr(target, 'divine_messages'):
            target.divine_messages = []
        target.divine_messages.append({
            "type": "whisper",
            "text": message,
            "tick": world.tick,
        })

        self._log("whisper", {"agent_id": agent_id, "message": message[:200]},
                  world.tick)
        return {"success": True, "agent_name": target.name}

    def send_vision_to_nation(self, world, nation_id: int, message: str) -> dict:
        """Send a vision to all agents in a nation."""
        nation = None
        for n in world.geopolitics.nations:
            if n.id == nation_id:
                nation = n
                break

        if nation is None:
            return {"success": False, "error": "Nation not found"}

        # Find all agents in this nation's settlements
        count = 0
        for settlement in world.settlements:
            if settlement.id in nation.settlement_ids:
                for agent in world.agents:
                    if agent.id in settlement.members and agent.alive:
                        if not hasattr(agent, 'divine_messages'):
                            agent.divine_messages = []
                        agent.divine_messages.append({
                            "type": "vision",
                            "text": message,
                            "tick": world.tick,
                        })
                        count += 1

        self._log("vision", {"nation_id": nation_id, "message": message[:200],
                              "reached": count}, world.tick)
        return {"success": True, "nation_name": nation.name, "agents_reached": count}

    def issue_commandment(self, world, message: str,
                          lat: float, lng: float, radius: float) -> dict:
        """Broadcast a divine commandment to all agents in an area."""
        count = 0
        for agent in world.agents:
            if not agent.alive:
                continue
            dist = np.sqrt((agent.lat - lat)**2 + (agent.lng - lng)**2)
            if dist <= radius:
                if not hasattr(agent, 'divine_messages'):
                    agent.divine_messages = []
                agent.divine_messages.append({
                    "type": "commandment",
                    "text": message,
                    "tick": world.tick,
                })
                count += 1

        self._log("commandment", {"message": message[:200], "lat": lat,
                                    "lng": lng, "radius": radius, "reached": count},
                  world.tick)
        return {"success": True, "agents_reached": count}

    # ------------------------------------------------------------------
    # World Interventions
    # ------------------------------------------------------------------

    def trigger_drought(self, world, lat: float, lng: float,
                        radius_deg: float, severity: float,
                        duration_ticks: int) -> dict:
        """
        Reduce food and water regeneration in an area.

        severity: 0.0-1.0 (1.0 = total crop failure)
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        effect = ActiveEffect(
            effect_type="drought", lat=lat, lng=lng,
            radius_deg=radius_deg, severity=severity,
            remaining_ticks=duration_ticks, total_ticks=duration_ticks,
        )
        self.active_effects.append(effect)

        # Apply immediate effect to resource map
        self._apply_drought(world, effect)

        self._log("drought", {"lat": lat, "lng": lng, "radius": radius_deg,
                                "severity": severity, "duration": duration_ticks},
                  world.tick)
        return {"success": True, "effect_id": len(self.active_effects) - 1}

    def trigger_plague(self, world, lat: float, lng: float,
                       radius_deg: float, severity: float,
                       duration_ticks: int) -> dict:
        """
        Cause health damage to agents in an area over time.

        severity: 0.0-1.0 (damage = severity * 5 per tick)
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        effect = ActiveEffect(
            effect_type="plague", lat=lat, lng=lng,
            radius_deg=radius_deg, severity=severity,
            remaining_ticks=duration_ticks, total_ticks=duration_ticks,
        )
        self.active_effects.append(effect)

        self._log("plague", {"lat": lat, "lng": lng, "radius": radius_deg,
                               "severity": severity, "duration": duration_ticks},
                  world.tick)
        return {"success": True, "effect_id": len(self.active_effects) - 1}

    def trigger_resource_discovery(self, world, lat: float, lng: float,
                                    resource_type: str, amount: float) -> dict:
        """Add resources to a specific location."""
        r, c = world.resources.get_cell(lat, lng)
        layer = getattr(world.resources, resource_type, None)
        if layer is None:
            return {"success": False, "error": f"Unknown resource: {resource_type}"}

        layer[r, c] = min(100.0, layer[r, c] + amount)

        self._log("resource_discovery", {"lat": lat, "lng": lng,
                                           "resource": resource_type, "amount": amount},
                  world.tick)
        return {"success": True}

    def grant_technology(self, world, agent_id: int = None,
                         skill: str = "research", boost: float = 0.2) -> dict:
        """Boost a skill for an agent or all agents."""
        targets = []
        if agent_id is not None:
            for a in world.agents:
                if a.id == agent_id and a.alive:
                    targets.append(a)
        else:
            targets = [a for a in world.agents if a.alive]

        for agent in targets:
            agent.skills.practice(skill, boost * 10, 2.0)

        self._log("grant_technology", {"agent_id": agent_id, "skill": skill,
                                         "boost": boost, "targets": len(targets)},
                  world.tick)
        return {"success": True, "agents_affected": len(targets)}

    def modify_climate(self, world, temperature_delta: float = 0.0,
                       co2_delta: float = 0.0) -> dict:
        """Nudge global climate parameters directly."""
        world.macro.state.temperature_anomaly += temperature_delta
        world.macro.state.co2_ppm += co2_delta

        self._log("modify_climate", {"temp_delta": temperature_delta,
                                       "co2_delta": co2_delta}, world.tick)
        return {
            "success": True,
            "new_temperature": world.macro.state.temperature_anomaly,
            "new_co2": world.macro.state.co2_ppm,
        }

    # ------------------------------------------------------------------
    # Tick Update (process active effects)
    # ------------------------------------------------------------------

    def update(self, world):
        """Called each simulation tick. Processes active effects and divine messages."""
        # Process active effects (drought, plague) — only if enabled
        if self.config.enabled and self.active_effects:
            expired = []
            for i, effect in enumerate(self.active_effects):
                effect.remaining_ticks -= 1

                if effect.effect_type == "plague":
                    self._apply_plague_tick(world, effect)
                # Drought is applied once when triggered (see trigger_drought);
                # its reduced regen persists until _revert_drought restores the
                # snapshot. It must NOT be re-applied per tick — doing so
                # compounded the reduction multiplicatively and never reverted
                # cleanly (the old approximate inverse left permanent drift).

                if effect.remaining_ticks <= 0:
                    expired.append(i)
                    if effect.effect_type == "drought":
                        self._revert_drought(world, effect)

            for i in reversed(expired):
                self.active_effects.pop(i)

        # ALWAYS process divine messages — they're already in agent queues
        # (regardless of whether God Mode toggle is on/off)
        for agent in world.agents:
            if not agent.alive:
                continue
            if hasattr(agent, 'divine_messages') and agent.divine_messages:
                msg = agent.divine_messages.pop(0)
                self._process_divine_message(agent, msg, world)

    def _apply_drought(self, world, effect: ActiveEffect):
        """Reduce food and water regen in the drought area (applied once).

        Snapshots the affected cells' pre-drought regen values onto the effect so
        ``_revert_drought`` can restore them exactly (idempotent baseline pattern,
        as used by the macro/ice-age coupling). This avoids the multiplicative
        drift of the previous per-tick apply + approximate-inverse revert.
        """
        res = world.resources
        food_snap: dict[tuple[int, int], float] = {}
        water_snap: dict[tuple[int, int], float] = {}
        for r in range(res.rows):
            lat = res.lat_max - (r + 0.5) * res.cell_size_deg
            for c in range(res.cols):
                lng = res.lng_min + (c + 0.5) * res.cell_size_deg
                dist = np.sqrt((lat - effect.lat)**2 + (lng - effect.lng)**2)
                if dist <= effect.radius_deg:
                    proximity = 1.0 - dist / effect.radius_deg
                    reduction = effect.severity * proximity
                    food_snap[(r, c)] = float(res.food_regen[r, c])
                    water_snap[(r, c)] = float(res.water_regen[r, c])
                    res.food_regen[r, c] *= max(0.1, 1.0 - reduction)
                    res.water_regen[r, c] *= max(0.1, 1.0 - reduction)
        effect.params["_food_baseline"] = food_snap
        effect.params["_water_baseline"] = water_snap

    def _revert_drought(self, world, effect: ActiveEffect):
        """Restore the exact pre-drought regen rates from the saved snapshot."""
        res = world.resources
        for (r, c), val in effect.params.get("_food_baseline", {}).items():
            res.food_regen[r, c] = val
        for (r, c), val in effect.params.get("_water_baseline", {}).items():
            res.water_regen[r, c] = val

    def _apply_plague_tick(self, world, effect: ActiveEffect):
        """Apply plague health damage to agents in area."""
        for agent in world.agents:
            if not agent.alive:
                continue
            dist = np.sqrt((agent.lat - effect.lat)**2 + (agent.lng - effect.lng)**2)
            if dist <= effect.radius_deg:
                proximity = 1.0 - dist / effect.radius_deg
                damage = effect.severity * 5.0 * proximity
                # Don't kill agents below 20 health from plague alone
                if agent.health > 20:
                    agent.health -= damage
                    agent.happiness -= damage * 0.5

    # Keyword → goal mapping for divine messages
    _GOAL_KEYWORDS = {
        "eat": ["eat", "food", "hunger", "feed", "harvest", "farm"],
        "explore": ["explore", "travel", "move", "journey", "discover", "wander",
                     "north", "south", "east", "west", "go", "leave", "search"],
        "trade": ["trade", "sell", "buy", "exchange", "merchant", "market", "deal"],
        "work": ["work", "labor", "build", "produce", "craft", "mine"],
        "socialize": ["friend", "talk", "meet", "ally", "cooperate", "unite", "peace"],
        "reproduce": ["reproduce", "children", "family", "breed", "multiply", "offspring"],
        "research": ["research", "learn", "study", "knowledge", "science", "think", "wisdom"],
        "migrate": ["migrate", "flee", "escape", "run", "relocate", "move away", "danger"],
        "heal": ["heal", "rest", "recover", "health", "medicine", "cure"],
        "govern": ["govern", "lead", "rule", "organize", "law", "order", "command"],
    }

    def _parse_goal_from_message(self, text: str) -> Optional[str]:
        """Extract a goal from a divine message by keyword matching."""
        text_lower = text.lower()
        best_goal = None
        best_count = 0
        for goal, keywords in self._GOAL_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_goal = goal
        return best_goal if best_count > 0 else None

    def _process_divine_message(self, agent, msg: dict, world):
        """
        Process a divine message on an agent.

        Effects when the agent complies:
        1. Parse message for goal keywords → override agent's current goal
        2. Boost relevant skills and energy
        3. Store in memory (affects future decisions)
        4. Increase divine trust (more likely to comply next time)

        Compliance probability:
        - High cooperation + high divine_trust → likely to obey
        - High intelligence → more skeptical (questions authority)
        """
        divine_trust = getattr(agent, 'divine_trust', 0.5)
        text = msg.get("text", "")

        # Compliance probability
        compliance_prob = np.clip(
            divine_trust * 0.5 +
            agent.traits["cooperation"] * 0.3 +
            (1.0 - agent.traits["intelligence"] * 0.3) * 0.2,
            0.1, 0.95
        )
        complied = np.random.random() < compliance_prob

        goal_parsed = self._parse_goal_from_message(text)
        effect_description = ""

        if complied and goal_parsed:
            # Override agent's current goal
            agent.current_goal = goal_parsed
            agent._cached_behavior = None  # Force re-plan next tick
            agent._plan_tick = 0

            # Boost energy (divine inspiration)
            agent.energy = min(100, agent.energy + 10)
            agent.happiness = min(100, agent.happiness + 5)

            # Store the divine command in memory so needs evaluation picks it up
            agent.memory.learn_fact(f"divine_goal_{goal_parsed}", 1.0)
            agent.memory.learn_fact("divine_commanded", 1.0)

            agent.divine_trust = min(1.0, divine_trust * 1.1)
            effect_description = f"Complied! Now pursuing: {goal_parsed}"

        elif complied:
            # Message had no clear goal but agent still listened
            agent.energy = min(100, agent.energy + 5)
            agent.happiness = min(100, agent.happiness + 3)
            agent.memory.learn_fact("divine_blessing", 0.8)
            agent.divine_trust = min(1.0, divine_trust * 1.05)
            effect_description = "Received blessing (no specific command found)"

        else:
            # Refused
            agent.divine_trust = max(0.1, divine_trust * 0.85)
            effect_description = "Refused the divine message"

        agent.memory.store_episode({
            "type": "divine_message",
            "msg_type": msg.get("type", "whisper"),
            "text": text[:100],
            "complied": complied,
            "goal_parsed": goal_parsed,
            "trust_after": agent.divine_trust,
        })

        # Generate spoken reaction
        llm = getattr(world, 'llm', None)
        if llm:
            reaction = llm.generate_direct_chat(
                agent, text, "as_god",
                world.get_local_state(agent.lat, agent.lng) if hasattr(world, 'get_local_state') else {}
            )
            agent.last_dialogue = reaction.get("text")
        else:
            if complied and goal_parsed:
                agent.last_dialogue = f"*bows* I shall {goal_parsed} as commanded."
            elif complied:
                agent.last_dialogue = "I hear the divine voice..."
            else:
                agent.last_dialogue = "I question this voice. I will decide for myself."

    # ------------------------------------------------------------------
    # Logging & Status
    # ------------------------------------------------------------------

    def _log(self, intervention_type: str, params: dict, tick: int):
        if self.config.log_interventions:
            self.intervention_log.append({
                "tick": tick,
                "type": intervention_type,
                "params": params,
            })
            # Keep log bounded
            if len(self.intervention_log) > 1000:
                self.intervention_log = self.intervention_log[-500:]

    def get_status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "active_effects": [
                {
                    "type": e.effect_type,
                    "lat": round(e.lat, 2),
                    "lng": round(e.lng, 2),
                    "radius": round(e.radius_deg, 2),
                    "severity": round(e.severity, 2),
                    "remaining": e.remaining_ticks,
                    "total": e.total_ticks,
                }
                for e in self.active_effects
            ],
            "total_interventions": len(self.intervention_log),
        }

    def get_intervention_log(self) -> list[dict]:
        return self.intervention_log
