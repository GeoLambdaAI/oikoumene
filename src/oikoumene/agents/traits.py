"""
Agent personality traits (moved from agents.py): TraitType, TRAIT_NAMES,
_wrap_longitude and generate_traits.
"""

import numpy as np
from typing import Optional
from enum import Enum


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
                    *, rng: np.random.RandomState) -> dict:
    """Generate traits via crossover + mutation or random initialization.

    ``rng`` is the world's seeded ``RandomState`` (required, so trait generation
    never touches the global ``np.random``).
    """
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
