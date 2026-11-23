"""
Agent memory and skills (moved from agents.py): AgentMemory, SkillSet.
"""

import numpy as np
from typing import Optional


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
                 *, rng: np.random.RandomState):
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
