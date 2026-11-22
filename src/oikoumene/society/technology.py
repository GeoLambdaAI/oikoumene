"""
Geographic advantage, technology tree and local adaptation for the historical
scenario (moved from history.py): GeographicAdvantage, TechNode, TECH_TREE, EvolutionModel.
"""

import numpy as np
from dataclasses import dataclass, field


# ============================================================================
# Geographic Determinism (Diamond's Framework)
# ============================================================================

class GeographicAdvantage:
    """
    Implements Jared Diamond's Guns, Germs, and Steel thesis:

    1. CONTINENTAL AXIS: East-west oriented continents (Eurasia) allow faster
       diffusion of crops, animals, and technology along similar latitudes.
       North-south continents (Americas, Africa) create climate barriers.
       Ref: Diamond (1997) Chapter 10

    2. DOMESTICABLE SPECIES: The Fertile Crescent had the most domesticable
       large-seeded grasses and large mammals. Other regions had fewer.
       Ref: Diamond (1997) Chapters 5-9

    3. DISEASE VECTORS: Proximity to domesticated animals → zoonotic diseases →
       population with acquired immunity → advantage in contact with others.
       Ref: Diamond (1997) Chapter 11

    4. GEOGRAPHIC BARRIERS: Mountains, deserts, oceans slow diffusion.
       Ref: Diamond (1997) Chapters 10, 15
    """

    # Agricultural origin zones: (lat, lng, start_year_bp, crop_value)
    # Ref: Diamond (1997), Purugganan & Fuller (2009)
    AGRICULTURE_ORIGINS = [
        {"name": "Fertile Crescent", "lat": 35, "lng": 40,
         "start_bp": 11500, "crop_value": 1.0,
         "crops": ["wheat", "barley", "lentils", "peas"],
         "animals": ["sheep", "goat", "cattle", "pig"]},
        {"name": "China (Yellow River)", "lat": 35, "lng": 112,
         "start_bp": 10000, "crop_value": 0.9,
         "crops": ["rice", "millet", "soybean"],
         "animals": ["pig", "silkworm", "water buffalo"]},
        {"name": "China (Yangtze)", "lat": 30, "lng": 115,
         "start_bp": 9000, "crop_value": 0.85,
         "crops": ["rice"],
         "animals": ["water buffalo"]},
        {"name": "Mesoamerica", "lat": 18, "lng": -96,
         "start_bp": 9000, "crop_value": 0.6,
         "crops": ["maize", "squash", "beans"],
         "animals": ["turkey"]},  # Few domesticable large mammals
        {"name": "Andes", "lat": -15, "lng": -72,
         "start_bp": 8000, "crop_value": 0.5,
         "crops": ["potato", "quinoa"],
         "animals": ["llama", "alpaca"]},  # No draft animals
        {"name": "Sahel (West Africa)", "lat": 14, "lng": -2,
         "start_bp": 7000, "crop_value": 0.55,
         "crops": ["sorghum", "millet", "cowpea"],
         "animals": []},
        {"name": "Eastern N. America", "lat": 38, "lng": -85,
         "start_bp": 5000, "crop_value": 0.4,
         "crops": ["sunflower", "squash"],
         "animals": []},
        {"name": "New Guinea", "lat": -6, "lng": 145,
         "start_bp": 7000, "crop_value": 0.45,
         "crops": ["taro", "yam", "banana"],
         "animals": []},
        {"name": "India (Ganges)", "lat": 26, "lng": 82,
         "start_bp": 8000, "crop_value": 0.7,
         "crops": ["rice", "cotton", "sesame"],
         "animals": ["zebu cattle", "chicken"]},
    ]

    # Continental axis orientation (Diamond Ch. 10)
    # East-west axis = similar climate zones = easier diffusion
    AXIS_ADVANTAGE = {
        # (lat_range, lng_range): axis_multiplier
        # Eurasia: strong E-W axis from 30°N to 55°N
        "eurasia": {"lat": (25, 55), "lng": (-10, 140), "axis": 1.5},
        # Africa: N-S axis, many climate barriers
        "africa": {"lat": (-35, 35), "lng": (-18, 52), "axis": 0.7},
        # Americas: extreme N-S axis
        "n_america": {"lat": (15, 55), "lng": (-130, -60), "axis": 0.6},
        "s_america": {"lat": (-55, 10), "lng": (-80, -35), "axis": 0.5},
        # Australia: moderate, but isolated
        "australia": {"lat": (-40, -10), "lng": (112, 155), "axis": 0.8},
    }

    # Diamond Ch.10: diffusion is faster east-west (constant latitude -> similar
    # climate) than north-south. The north-south reach is this fraction of the
    # east-west reach, making the spread an ellipse elongated along latitude.
    NS_DIFFUSION_FRACTION = 0.5

    def get_agricultural_potential(self, lat: float, lng: float,
                                   year_bp: float) -> float:
        """
        Get agricultural potential for a location at a given time.

        Returns 0-1 score based on:
        - Proximity to agricultural origin zones
        - Whether agriculture has been invented there yet (year_bp)
        - Diffusion rate along continental axis
        """
        if year_bp > 12000:
            return 0.0  # No agriculture before Neolithic

        best_potential = 0.0

        for origin in self.AGRICULTURE_ORIGINS:
            if year_bp > origin["start_bp"]:
                continue  # Not invented yet

            # Offset from origin (degrees)
            dlat = lat - origin["lat"]
            dlng = lng - origin["lng"]

            # Time since invention at origin
            years_since = origin["start_bp"] - year_bp

            # Anisotropic diffusion (Diamond 1997, Ch.10): crops/tech spread faster
            # along the EAST-WEST axis (constant latitude -> similar climate & day
            # length) than NORTH-SOUTH (crossing climate zones). Base E-W speed
            # ~1 deg / 100 yr (Ammerman & Cavalli-Sforza 1971), scaled by the
            # continent's axis-orientation advantage; the N-S reach is a fixed
            # fraction of it, so the spread is an ellipse elongated east-west.
            axis_mult = self._get_axis_multiplier(lat, lng)
            ew_reach = (years_since * 0.01) * axis_mult           # east-west radius (deg)
            ns_reach = ew_reach * self.NS_DIFFUSION_FRACTION      # north-south radius (deg)

            # Normalised elliptical distance: < 1 means agriculture has reached here.
            norm = np.sqrt((dlng / max(ew_reach, 1e-6)) ** 2 +
                           (dlat / max(ns_reach, 1e-6)) ** 2)
            if norm < 1.0:
                proximity = 1.0 - norm
                potential = origin["crop_value"] * proximity
                best_potential = max(best_potential, potential)

        return float(np.clip(best_potential, 0, 1))

    def get_disease_resistance(self, lat: float, lng: float,
                               year_bp: float) -> float:
        """
        Disease resistance from proximity to domesticated animals.
        Ref: Diamond (1997) Ch. 11

        Populations near agriculture origins with many animals develop
        zoonotic diseases (smallpox, measles, influenza) and partial immunity.
        """
        if year_bp > 8000:
            return 0.0  # Too early for disease effects

        resistance = 0.0
        for origin in self.AGRICULTURE_ORIGINS:
            if year_bp > origin["start_bp"]:
                continue
            n_animals = len(origin["animals"])
            if n_animals == 0:
                continue

            dist = np.sqrt((lat - origin["lat"])**2 + (lng - origin["lng"])**2)
            years_since = origin["start_bp"] - year_bp

            # Disease resistance builds over millennia
            time_factor = min(1.0, years_since / 5000)
            animal_factor = min(1.0, n_animals / 4)

            diffusion = max(0, 1.0 - dist / (years_since * 0.01 + 1))
            resistance = max(resistance, time_factor * animal_factor * diffusion)

        return float(np.clip(resistance, 0, 1))

    def get_tech_diffusion_rate(self, lat: float, lng: float) -> float:
        """
        Technology diffusion speed multiplier based on continental axis.
        Eurasia (E-W axis) = fastest. Americas (N-S) = slowest.
        """
        return self._get_axis_multiplier(lat, lng)

    def _get_axis_multiplier(self, lat: float, lng: float) -> float:
        """Get axis advantage multiplier for a location."""
        for name, region in self.AXIS_ADVANTAGE.items():
            lat_range = region["lat"]
            lng_range = region["lng"]
            if lat_range[0] <= lat <= lat_range[1] and lng_range[0] <= lng <= lng_range[1]:
                return region["axis"]
        return 0.5  # Default for islands/remote regions


# ============================================================================
# Civilization Stage Tracker
# ============================================================================

@dataclass
class TechNode:
    """A technology or cultural advancement."""
    name: str
    era_requirement: str       # Minimum era name
    year_bp_available: float   # Earliest year this can be discovered
    prerequisites: list[str] = field(default_factory=list)
    food_multiplier: float = 1.0
    military_multiplier: float = 1.0
    trade_multiplier: float = 1.0
    description: str = ""


TECH_TREE = [
    # Paleolithic
    TechNode("fire_control", "Out of Africa", 70000,
             food_multiplier=1.3, description="Controlled use of fire for cooking, warmth"),
    TechNode("stone_tools", "Out of Africa", 70000,
             food_multiplier=1.1, military_multiplier=1.2),
    TechNode("language", "Out of Africa", 60000,
             trade_multiplier=1.5, description="Complex spoken language"),
    TechNode("cave_art", "Global Colonization", 40000,
             description="Symbolic thought, cultural transmission"),
    TechNode("bow_arrow", "Global Colonization", 30000,
             food_multiplier=1.2, military_multiplier=1.5),
    TechNode("fishing", "Global Colonization", 25000,
             food_multiplier=1.4),
    TechNode("sewing", "Global Colonization", 25000,
             description="Tailored clothing enables cold-climate survival"),

    # Neolithic
    TechNode("agriculture", "Neolithic Revolution", 12000,
             prerequisites=["stone_tools"],
             food_multiplier=3.0, description="Plant cultivation"),
    TechNode("animal_domestication", "Neolithic Revolution", 11000,
             prerequisites=["agriculture"],
             food_multiplier=2.0, military_multiplier=1.3),
    TechNode("pottery", "Neolithic Revolution", 10000,
             food_multiplier=1.2, trade_multiplier=1.2),
    TechNode("irrigation", "Neolithic Revolution", 8000,
             prerequisites=["agriculture"],
             food_multiplier=2.5),
    TechNode("writing", "Bronze & Iron Ages", 5200,
             prerequisites=["pottery", "agriculture"],
             trade_multiplier=2.0),

    # Metal Ages
    TechNode("bronze_working", "Bronze & Iron Ages", 5000,
             prerequisites=["pottery"],
             military_multiplier=2.0, trade_multiplier=1.5),
    TechNode("iron_working", "Bronze & Iron Ages", 3200,
             prerequisites=["bronze_working"],
             military_multiplier=2.5, food_multiplier=1.5),
    TechNode("wheel", "Bronze & Iron Ages", 5500,
             trade_multiplier=2.0),
    TechNode("sailing", "Bronze & Iron Ages", 4000,
             trade_multiplier=3.0),

    # Classical
    TechNode("mathematics", "Classical & Medieval", 2500,
             prerequisites=["writing"]),
    TechNode("philosophy", "Classical & Medieval", 2500,
             prerequisites=["writing"]),
    TechNode("steel", "Classical & Medieval", 2000,
             prerequisites=["iron_working"],
             military_multiplier=3.0),
    TechNode("paper", "Classical & Medieval", 2100,
             prerequisites=["writing"],
             trade_multiplier=1.5),

    # Medieval
    TechNode("gunpowder", "Classical & Medieval", 1000,
             prerequisites=["steel"],
             military_multiplier=5.0),
    TechNode("printing_press", "Classical & Medieval", 575,
             prerequisites=["paper"],
             trade_multiplier=3.0),
    TechNode("compass", "Classical & Medieval", 900,
             prerequisites=["sailing"],
             trade_multiplier=2.0),

    # Early Modern
    TechNode("ocean_navigation", "Early Modern", 500,
             prerequisites=["compass", "sailing"],
             trade_multiplier=5.0),
    TechNode("scientific_method", "Early Modern", 400,
             prerequisites=["mathematics", "printing_press"]),

    # Industrial
    TechNode("steam_engine", "Industrial", 225,
             prerequisites=["iron_working", "scientific_method"],
             trade_multiplier=5.0, food_multiplier=2.0),
    TechNode("electricity", "Industrial", 150,
             prerequisites=["steam_engine", "scientific_method"]),
    TechNode("fossil_fuel_extraction", "Industrial", 200,
             prerequisites=["steam_engine"]),

    # Modern
    TechNode("nuclear_energy", "Modern", 75,
             prerequisites=["electricity", "scientific_method"]),
    TechNode("computers", "Modern", 70,
             prerequisites=["electricity"]),
    TechNode("internet", "Modern", 30,
             prerequisites=["computers"],
             trade_multiplier=10.0),
    TechNode("renewable_energy", "Modern", 40,
             prerequisites=["electricity", "scientific_method"]),
]


# ============================================================================
# Evolution / Adaptation Model
# ============================================================================

class EvolutionModel:
    """
    Models evolutionary adaptation of agent populations to local environments.

    Inspired by Dawkins' "The Greatest Show on Earth" — evidence for
    natural selection acting on populations over generations.

    Key adaptations modeled:
    - Cold tolerance (higher in populations that lived in glacial regions)
    - Heat tolerance (higher in tropical populations)
    - Altitude adaptation (Tibetan, Andean, Ethiopian highland populations)
    - Disease resistance (from animal proximity, Diamond Ch. 11)
    - Lactose tolerance (from pastoralism, ~7500 BP in Europe)

    Ref: Dawkins (2009), Stringer (2012), Jablonski (2006)
    """

    @staticmethod
    def compute_local_adaptation(agent_traits: dict, lat: float, lng: float,
                                 generations_here: int,
                                 local_temp: float) -> dict:
        """
        Compute trait bonuses from local environmental adaptation.

        Returns dict of trait modifiers (additive, small per generation).
        Ref: Dawkins Ch. 5 (natural selection in action)
        """
        modifiers = {}

        # Cold adaptation: resilience increases in cold climates
        # Ref: Jablonski & Chaplin (2010), adaptation to UV/cold
        if local_temp < 5:
            cold_pressure = min(0.001 * generations_here, 0.1)
            modifiers["resilience_bonus"] = cold_pressure

        # Heat adaptation: energy efficiency in hot climates
        if local_temp > 25:
            heat_pressure = min(0.001 * generations_here, 0.1)
            modifiers["resilience_bonus"] = heat_pressure

        # Altitude adaptation (lat/lng near known highland populations)
        # Ref: Beall (2007) — Tibetan, Andean, Ethiopian adaptations
        highland_regions = [(30, 88, "Tibet"), (-15, -72, "Andes"), (8, 38, "Ethiopia")]
        for hlat, hlng, name in highland_regions:
            if np.sqrt((lat - hlat)**2 + (lng - hlng)**2) < 10:
                modifiers["resilience_bonus"] = min(0.001 * generations_here, 0.15)

        return modifiers
