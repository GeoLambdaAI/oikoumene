"""
Historical Simulation Framework — 70,000 Years of Human Civilization.

Models:
1. PALEOCLIMATE: Ice core CO2/temperature reconstructions, Milankovitch cycles,
   ice sheet extent, sea level changes (EPICA, Vostok, Marcott et al. 2013)
2. HUMAN MIGRATION: Out of Africa → colonization of continents with timing
   (Ref: Stringer 2012, "Early Modern Human" Wikipedia)
3. GEOGRAPHIC DETERMINISM: Diamond's Guns Germs & Steel thesis —
   east-west axis diffusion advantage, domesticable species distribution,
   disease vectors from animal proximity
4. CIVILIZATION STAGES: Hunter-gatherer → Neolithic → Bronze/Iron → Classical →
   Medieval → Early Modern → Industrial → Information
5. EVOLUTION / ADAPTATION: Dawkins' evidence for evolution —
   agents accumulate adaptations to local environments over generations
6. ICE AGE DYNAMICS: Glacial cycles, ice sheet extent, permafrost,
   land bridges (Beringia, Sundaland), refugia

This module holds the eras, migration waves, spawn locations and
HistoricalSimulation, which combines the parts. The paleoclimate and ice-age
model (1, 6) lives in oikoumene.core.paleoclimate; geographic determinism, the
technology tree and local adaptation (3, 5) live in oikoumene.society.technology.

Time system: simulation year as BP (Before Present, relative to 1950 CE).
Negative values = CE years (e.g., -74 = 2024 CE).

Key scientific references:
- EPICA Community (2004): 800,000 years of CO2, Nature 429
- Marcott et al. (2013): Holocene temperature reconstruction, Science 339
- Jouzel et al. (2007): Vostok/EPICA temperature, Science 317
- Diamond (1997): Guns, Germs, and Steel
- Stringer (2012): The Origin of Our Species
- Dawkins (2009): The Greatest Show on Earth
- Marshak (2019): Earth: Portrait of a Planet (6th ed., W. W. Norton, ISBN 978-0393640137)
- Clark et al. (2009): Last Glacial Maximum ice sheets, Quat. Sci. Rev.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from oikoumene.core.paleoclimate import PaleoclimateModel
from oikoumene.society.technology import TECH_TREE, EvolutionModel, GeographicAdvantage


# ============================================================================
# Era Definitions
# ============================================================================

@dataclass
class Era:
    """Defines a historical era with its parameters."""
    name: str
    start_year_bp: float     # Start year (Before Present)
    end_year_bp: float       # End year (Before Present)
    time_scale: float        # Simulation years per tick
    max_settlement_pop: int  # Max agents per settlement
    max_governance: str      # Highest governance type available
    available_actions: list[str] = field(default_factory=list)
    description: str = ""


# Eras from oldest to most recent
ERAS = [
    Era(
        name="Out of Africa",
        start_year_bp=70000, end_year_bp=45000,
        time_scale=200.0,  # 200 years per tick
        max_settlement_pop=30,
        max_governance="tribal",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate"],
        description="Homo sapiens migrate out of East Africa into the Middle East and beyond.",
    ),
    Era(
        name="Global Colonization",
        start_year_bp=45000, end_year_bp=12000,
        time_scale=100.0,  # 100 years per tick
        max_settlement_pop=50,
        max_governance="tribal",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate", "trade", "research"],
        description="Humans colonize Europe, Asia, Australia, and eventually the Americas. "
                    "Cave art, tool sophistication, Last Glacial Maximum at ~20,000 BP.",
    ),
    Era(
        name="Neolithic Revolution",
        start_year_bp=12000, end_year_bp=5000,
        time_scale=50.0,  # 50 years per tick
        max_settlement_pop=200,
        max_governance="council",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate", "trade", "research", "work", "build_business"],
        description="Agriculture develops independently in Fertile Crescent, China, Mesoamerica. "
                    "First permanent settlements, animal domestication, pottery.",
    ),
    Era(
        name="Bronze & Iron Ages",
        start_year_bp=5000, end_year_bp=2500,
        time_scale=20.0,  # 20 years per tick
        max_settlement_pop=500,
        max_governance="republic",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate", "trade", "research", "work", "build_business",
                           "govern"],
        description="First cities (Uruk, Mohenjo-daro), writing, bronze/iron metallurgy. "
                    "Empires form: Egypt, Mesopotamia, Indus Valley, Shang China.",
    ),
    Era(
        name="Classical & Medieval",
        start_year_bp=2500, end_year_bp=500,
        time_scale=10.0,  # 10 years per tick
        max_settlement_pop=2000,
        max_governance="republic",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate", "trade", "research", "work", "build_business",
                           "govern"],
        description="Greek, Roman, Chinese, Islamic golden ages. Silk Road trade. "
                    "Medieval period. Black Death. Printing press.",
    ),
    Era(
        name="Early Modern",
        start_year_bp=500, end_year_bp=200,
        time_scale=5.0,  # 5 years per tick
        max_settlement_pop=10000,
        max_governance="democracy",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate", "trade", "research", "work", "build_business",
                           "govern"],
        description="Age of Exploration, colonialism, scientific revolution. "
                    "Guns, germs, and steel determine who conquers whom.",
    ),
    Era(
        name="Industrial",
        start_year_bp=200, end_year_bp=75,
        time_scale=2.0,  # 2 years per tick
        max_settlement_pop=50000,
        max_governance="democracy",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate", "trade", "research", "work", "build_business",
                           "govern"],
        description="Industrial revolution, fossil fuels, railways, world wars. "
                    "CO2 begins rising above Holocene levels.",
    ),
    Era(
        name="Modern",
        start_year_bp=75, end_year_bp=-200,
        time_scale=1.0 / 12.0,  # 1 month per tick (current system)
        max_settlement_pop=100000,
        max_governance="democracy",
        available_actions=["eat", "explore", "socialize", "reproduce", "heal",
                           "migrate", "trade", "research", "work", "build_business",
                           "govern"],
        description="Nuclear age, space age, information age, climate crisis. "
                    "The macro ODE system (macro.py) takes over here.",
    ),
]


def get_era(year_bp: float) -> Era:
    """Get the current era for a given year BP."""
    for era in ERAS:
        if era.start_year_bp >= year_bp > era.end_year_bp:
            return era
    return ERAS[-1]  # Default to Modern


def year_bp_to_ce(year_bp: float) -> float:
    """Convert BP year to CE/BCE year. Positive = CE, negative = BCE."""
    return 1950.0 - year_bp


def year_ce_to_bp(year_ce: float) -> float:
    """Convert CE year to BP year."""
    return 1950.0 - year_ce


# ============================================================================
# Human Migration Origins
# ============================================================================

# Initial spawn point: East Africa (where Homo sapiens originated)
ORIGIN_POINT = (2.0, 36.0)  # Kenya/Ethiopia

# Migration waypoints with approximate arrival dates
# Ref: Stringer 2012, Oppenheimer 2003
MIGRATION_WAVES = [
    {"year_bp": 70000, "lat": 2, "lng": 36, "name": "East Africa origin"},
    {"year_bp": 65000, "lat": 15, "lng": 42, "name": "Horn of Africa / Arabia"},
    {"year_bp": 60000, "lat": 25, "lng": 55, "name": "Middle East"},
    {"year_bp": 55000, "lat": 25, "lng": 75, "name": "South Asia"},
    {"year_bp": 50000, "lat": -10, "lng": 130, "name": "Australia (via Sundaland)"},
    {"year_bp": 45000, "lat": 45, "lng": 25, "name": "Europe"},
    {"year_bp": 40000, "lat": 35, "lng": 110, "name": "East Asia"},
    {"year_bp": 30000, "lat": 55, "lng": 90, "name": "Central/North Asia"},
    {"year_bp": 16000, "lat": 65, "lng": -160, "name": "Beringia crossing"},
    {"year_bp": 15000, "lat": 45, "lng": -110, "name": "North America"},
    {"year_bp": 14000, "lat": 20, "lng": -100, "name": "Central America"},
    {"year_bp": 13000, "lat": -10, "lng": -60, "name": "South America"},
]


def get_spawn_locations(year_bp: float, n_agents: int,
                        rng=None) -> list[tuple[float, float]]:
    """
    Get spawn locations for initial agents based on the current year.

    Early years: agents spawn only in East Africa.
    Later years: agents can spawn in colonized regions.

    ``rng``: the world's seeded RandomState. Threading it in makes spawn
    locations vary with the world seed (previously a hardcoded RandomState(42)
    ignored the seed entirely). Falls back to RandomState(42) for callers that
    do not supply one.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    available_regions = []

    for wave in MIGRATION_WAVES:
        if year_bp >= wave["year_bp"]:
            available_regions.append(wave)

    if not available_regions:
        available_regions = [MIGRATION_WAVES[0]]

    points = []
    for _ in range(n_agents):
        # available_regions is ordered oldest-first (East Africa origin at
        # index 0). The exponential draw concentrates mass at index 0, so most
        # agents spawn at the origin and progressively fewer at each younger
        # migration frontier — matching the demographic reality that the origin
        # sustained the largest established population.
        idx = min(len(available_regions) - 1, int(rng.exponential(1.5)))
        region = available_regions[idx]
        # Add jitter around the region center
        lat = region["lat"] + rng.normal(0, 3)
        lng = region["lng"] + rng.normal(0, 3)
        points.append((lat, lng))

    return points


# ============================================================================
# Master Historical Simulation Controller
# ============================================================================

class HistoricalSimulation:
    """
    Orchestrates the historical timeline, coordinating:
    - Paleoclimate state
    - Current era and available technologies
    - Geographic advantages
    - Migration patterns
    - Evolution/adaptation

    Integrated into World.step() via the bridge.
    """

    def __init__(self, start_year_bp: float = 70000, *, rng: np.random.RandomState):
        self.year_bp = start_year_bp
        self.paleoclimate = PaleoclimateModel()
        self.geographic = GeographicAdvantage()
        self.evolution = EvolutionModel()
        self.discovered_techs: set[str] = set()
        # Seeded RNG for stochastic tech discovery. Threading the world's
        # RandomState in keeps discovery timing (which drives
        # industrialization_level -> macro CO2/temperature) reproducible for a
        # fixed seed, instead of leaking to the unseeded global np.random.
        # Required: an unseeded fallback made runs irreproducible.
        self.rng = rng

        # Initialize with era-appropriate starting techs
        era = get_era(start_year_bp)
        for tech in TECH_TREE:
            if tech.year_bp_available >= start_year_bp and not tech.prerequisites:
                self.discovered_techs.add(tech.name)

    # Industrial-era technologies whose discovery drives anthropogenic fossil
    # emissions. industrialization_level() returns the fraction discovered, so a
    # pre-industrial civilization -> 0 and a fully developed (Information-era)
    # civilization -> 1. (renewable_energy is excluded: it mitigates rather than
    # drives fossil emissions, and is handled via the renewable fraction.)
    INDUSTRIAL_TECHS = (
        "steam_engine", "electricity", "fossil_fuel_extraction",
        "nuclear_energy", "computers", "internet",
    )

    def get_current_era(self) -> Era:
        return get_era(self.year_bp)

    def get_current_year_ce(self) -> float:
        return year_bp_to_ce(self.year_bp)

    def industrialization_level(self) -> float:
        """Fraction of industrial-era technologies the civilization has discovered.

        Drives anthropogenic fossil emissions in the macro model: 0.0 for a
        pre-industrial civilization, ramping to 1.0 once it reaches the
        Information era. Used so CO2 output scales with the civilisation's actual
        development rather than a fixed present-day rate.
        """
        discovered = sum(1 for t in self.INDUSTRIAL_TECHS if t in self.discovered_techs)
        return discovered / len(self.INDUSTRIAL_TECHS)

    def advance_time(self, n_ticks: int = 1) -> dict:
        """Advance simulation time by n ticks of the current era's time_scale."""
        era = self.get_current_era()
        years_passed = era.time_scale * n_ticks
        self.year_bp -= years_passed

        # Get updated climate
        climate = self.paleoclimate.get_climate(self.year_bp)

        # Check for newly available technologies
        new_techs = []
        for tech in TECH_TREE:
            if tech.name in self.discovered_techs:
                continue
            if tech.year_bp_available < self.year_bp:
                continue  # Not yet available
            # Check prerequisites
            if all(p in self.discovered_techs for p in tech.prerequisites):
                # Discovery probability increases over time past availability
                years_overdue = tech.year_bp_available - self.year_bp
                discovery_prob = min(0.5, years_overdue / 500)
                if self.rng.random() < discovery_prob:
                    self.discovered_techs.add(tech.name)
                    new_techs.append(tech.name)

        return {
            "year_bp": self.year_bp,
            "year_ce": year_bp_to_ce(self.year_bp),
            "era": era.name,
            "era_time_scale": era.time_scale,
            "climate": climate,
            "new_techs": new_techs,
            "n_techs": len(self.discovered_techs),
        }

    def get_climate_modifier(self, lat: float, lng: float) -> dict:
        """
        Get climate modifiers for a specific location at the current year.
        Combines global paleoclimate with local ice sheet coverage.
        """
        climate = self.paleoclimate.get_climate(self.year_bp)

        # Check ice coverage
        is_iced = self.paleoclimate.get_ice_mask(self.year_bp, lat, lng)

        # Fertility modifier from climate
        temp_offset = climate["temperature_anomaly"]
        if is_iced:
            fertility_mult = 0.0  # No agriculture under ice
            habitability = 0.0
        elif temp_offset < -4:
            # Very cold era: reduced fertility everywhere
            fertility_mult = max(0.2, 1.0 + temp_offset * 0.1)
            habitability = max(0.3, 1.0 + temp_offset * 0.1)
        else:
            fertility_mult = max(0.5, 1.0 + temp_offset * 0.05)
            habitability = 1.0

        # Agricultural potential (Diamond)
        ag_potential = self.geographic.get_agricultural_potential(lat, lng, self.year_bp)

        return {
            "temp_offset": temp_offset,
            "fertility_multiplier": fertility_mult,
            "habitability": habitability,
            "is_iced": is_iced,
            "co2_ppm": climate["co2_ppm"],
            "sea_level_m": climate["sea_level_m"],
            "agricultural_potential": ag_potential,
            "disease_resistance": self.geographic.get_disease_resistance(lat, lng, self.year_bp),
            "tech_diffusion_rate": self.geographic.get_tech_diffusion_rate(lat, lng),
        }

    def get_summary(self) -> dict:
        """Summary for UI."""
        era = self.get_current_era()
        climate = self.paleoclimate.get_climate(self.year_bp)
        year_ce = year_bp_to_ce(self.year_bp)

        if year_ce < 0:
            year_str = f"{int(abs(year_ce)):,} BCE"
        else:
            year_str = f"{int(year_ce):,} CE"

        return {
            "year_bp": round(self.year_bp, 1),
            "year_ce": round(year_ce, 1),
            "year_display": year_str,
            "era_name": era.name,
            "era_description": era.description,
            "time_scale": era.time_scale,
            "co2_ppm": round(climate["co2_ppm"], 1),
            "temperature_anomaly": round(climate["temperature_anomaly"], 2),
            "sea_level_m": round(climate["sea_level_m"], 1),
            "ice_fraction": round(climate["ice_fraction"], 3),
            "n_techs_discovered": len(self.discovered_techs),
            "techs_discovered": sorted(self.discovered_techs),
        }
