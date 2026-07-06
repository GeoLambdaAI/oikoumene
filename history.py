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
# Paleoclimate Model
# ============================================================================

class PaleoclimateModel:
    """
    Reconstructed paleoclimate from ice core data (EPICA, Vostok).

    Provides temperature anomaly, CO2, sea level, and ice sheet extent
    for any year in the last 100,000 years.

    Data sources:
    - EPICA Dome C ice core: CO2 for 800 kyr (Lüthi et al. 2008)
    - Vostok ice core: temperature for 420 kyr (Petit et al. 1999)
    - Marcott et al. (2013): Holocene temperature stack
    - Spratt & Lisiecki (2016): sea level for 800 kyr
    - Clark et al. (2009): LGM ice sheet reconstructions
    """

    # Key paleoclimate data points: (year_bp, CO2_ppm, temp_anomaly_C, sea_level_m)
    # Ref: EPICA, Vostok, Marcott et al. 2013, Spratt & Lisiecki 2016
    PALEOCLIMATE_DATA = [
        # year_bp,  CO2,   dT(°C),  sea_level(m vs present)
        (100000,    240,   -2.0,    -20),    # Eemian cooling
        (90000,     230,   -4.0,    -40),    # Early glaciation
        (80000,     210,   -5.0,    -60),    # MIS 4
        (70000,     200,   -6.0,    -80),    # Deep glaciation, Out of Africa
        (60000,     200,   -5.5,    -70),    # MIS 3 interstadial
        (50000,     210,   -4.5,    -60),    # Continued glaciation
        (40000,     200,   -5.0,    -70),    # Neanderthal coexistence
        (30000,     200,   -6.0,    -90),    # Approaching LGM
        (26000,     185,   -7.5,    -120),   # Last Glacial Maximum begins
        (21000,     185,   -8.0,    -130),   # LGM peak: max ice extent
        (18000,     190,   -7.0,    -125),   # LGM ending
        (15000,     210,   -5.0,    -100),   # Bølling-Allerød warm period
        (12900,     240,   -3.0,    -70),    # Pre-Younger Dryas
        (12000,     235,   -5.0,    -65),    # Younger Dryas cold snap
        (11700,     260,   -1.5,    -60),    # Younger Dryas ends abruptly
        (10000,     265,   -0.5,    -40),    # Early Holocene
        (8000,      270,    0.0,    -20),    # Holocene Climatic Optimum start
        (6000,      270,    0.5,    -5),     # Holocene Climatic Optimum peak
        (4000,      270,    0.0,    -2),     # Neoglaciation begins
        (2000,      275,   -0.2,    0),      # Roman Warm Period
        (1000,      280,   -0.1,    0),      # Medieval Warm Period
        (500,       280,   -0.5,    0),      # Little Ice Age depths (1450 CE)
        (200,       283,   -0.3,    0),      # Little Ice Age (1750 CE, 200 BP)
        (75,        290,   -0.1,    0),      # 1875 CE (75 BP) — early industrial
        (0,         310,    0.3,    0.0),    # 1950 CE (0 BP) — mid-20th century
        (-74,       420,    1.2,    0.15),   # 2024 CE (-74 BP) — present day
    ]
    # Note: BP is Before Present = 1950 CE (year_bp_to_ce(bp) = 1950 - bp), so the
    # industrial ramp is 290 ppm (1875) -> 310 (1950) -> 420 (2024 = -74 BP). The
    # last point uses negative BP because the modern era extends past 1950 (see the
    # ERAS table). np.interp clamps queries beyond -74 BP to the present value.

    # Ice sheet coverage at LGM: approximate southern boundary latitudes
    # Ref: Clark et al. (2009), Marshak (2019) Ch. 22
    LGM_ICE_EXTENT = {
        "laurentide": {  # North America
            "lat_south": 40.0,  # Ice reached ~40°N (New York, Chicago)
            "lng_min": -130.0,
            "lng_max": -60.0,
        },
        "fennoscandian": {  # Northern Europe
            "lat_south": 50.0,  # Ice reached ~50°N (London, Berlin)
            "lng_min": -10.0,
            "lng_max": 60.0,
        },
        "greenland": {
            "lat_south": 60.0,
            "lng_min": -73.0,
            "lng_max": -12.0,
        },
    }

    # Land bridges exposed at low sea levels
    # Ref: Marshak (2019), Voris (2000)
    LAND_BRIDGES = [
        {"name": "Beringia", "lat": 65, "lng": -170,
         "threshold_m": -50, "connects": ("Asia", "Americas")},
        {"name": "Sundaland", "lat": -2, "lng": 110,
         "threshold_m": -40, "connects": ("Asia", "Australia_approach")},
        {"name": "Doggerland", "lat": 54, "lng": 3,
         "threshold_m": -30, "connects": ("Britain", "Europe")},
    ]

    def __init__(self):
        # Pre-sort data by year for interpolation
        self._data = sorted(self.PALEOCLIMATE_DATA, key=lambda x: -x[0])
        self._years = np.array([d[0] for d in self._data])
        self._co2 = np.array([d[1] for d in self._data])
        self._temp = np.array([d[2] for d in self._data])
        self._sea = np.array([d[3] for d in self._data])

    def get_climate(self, year_bp: float) -> dict:
        """
        Get paleoclimate state for a given year.

        Returns dict with:
        - co2_ppm: atmospheric CO2
        - temperature_anomaly: °C relative to pre-industrial (1850)
        - sea_level_m: meters relative to present
        - ice_fraction: fraction of Northern Hemisphere covered by ice
        """
        # Interpolate from data points
        co2 = float(np.interp(year_bp, self._years[::-1], self._co2[::-1]))
        temp = float(np.interp(year_bp, self._years[::-1], self._temp[::-1]))
        sea = float(np.interp(year_bp, self._years[::-1], self._sea[::-1]))

        # Ice fraction: approximate from temperature anomaly
        # At LGM (-8°C): ~30% of NH land covered
        # At present (~+1°C): ~10% (Greenland + small glaciers)
        # Ref: Clark et al. 2009
        ice_fraction = np.clip(0.10 + 0.025 * max(0, -temp), 0, 0.35)

        return {
            "co2_ppm": co2,
            "temperature_anomaly": temp,
            "sea_level_m": sea,
            "ice_fraction": ice_fraction,
        }

    def get_ice_mask(self, year_bp: float, lat: float, lng: float) -> bool:
        """Check if a location is under ice sheet for a given year."""
        climate = self.get_climate(year_bp)
        temp = climate["temperature_anomaly"]

        # No significant extra ice when warmer than -2°C anomaly
        if temp > -2.0:
            return False

        # Scale ice extent linearly between present and LGM
        # LGM at -8°C, present at 0°C
        ice_scale = np.clip(-temp / 8.0, 0, 1)

        for name, extent in self.LGM_ICE_EXTENT.items():
            if extent["lng_min"] <= lng <= extent["lng_max"]:
                # Ice boundary moves south proportionally
                present_boundary = 75.0  # Present ice starts ~75°N
                lgm_boundary = extent["lat_south"]
                boundary = present_boundary - (present_boundary - lgm_boundary) * ice_scale
                if lat >= boundary:
                    return True

        # Antarctic ice sheet (always present below ~-65°, extends to -55° at LGM)
        antarctic_boundary = -65.0 + 10.0 * ice_scale
        if lat <= antarctic_boundary:
            return True

        return False

    def is_land_bridge_active(self, bridge_name: str, year_bp: float) -> bool:
        """Check if a land bridge is exposed (sea level low enough)."""
        climate = self.get_climate(year_bp)
        for bridge in self.LAND_BRIDGES:
            if bridge["name"] == bridge_name:
                return climate["sea_level_m"] <= bridge["threshold_m"]
        return False


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

    def __init__(self, start_year_bp: float = 70000, rng=None):
        self.year_bp = start_year_bp
        self.paleoclimate = PaleoclimateModel()
        self.geographic = GeographicAdvantage()
        self.evolution = EvolutionModel()
        self.discovered_techs: set[str] = set()
        # Seeded RNG for stochastic tech discovery. Threading the world's
        # RandomState in keeps discovery timing (which drives
        # industrialization_level -> macro CO2/temperature) reproducible for a
        # fixed seed, instead of leaking to the unseeded global np.random.
        self.rng = rng if rng is not None else np.random.RandomState()

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
