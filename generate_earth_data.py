#!/usr/bin/env python3
"""
Comprehensive Earth System Data Generator.

Generates realistic global grids from first-principles Earth science:

1. ELEVATION — synthetic topo from tectonic plate boundaries & known ranges
2. TEMPERATURE — latitude + elevation lapse rate + continentality + ocean currents
3. PRECIPITATION — ITCZ, Hadley cells, monsoon, orographic, continentality
4. BIOME — Whittaker biome diagram: temperature × precipitation → biome class
5. SOIL FERTILITY — f(biome, precipitation, temperature, river proximity)
6. MINERAL RICHNESS — f(tectonic setting, mountain proximity, geology proxy)
7. FRESHWATER — f(precipitation, river proximity, lake proximity, aquifer proxy)
8. FOSSIL FUEL POTENTIAL — f(sedimentary basin locations)

All grids at 0.5° resolution (360 rows × 720 cols) covering 90°N to 90°S.

Scientific references cited inline. Run once; output saved as .npy files.

Author approach: Senior Earth Scientist + Senior Geospatial Data Scientist
"""

import json
import os
import time

import numpy as np
from shapely.geometry import shape, Point, LineString, MultiLineString
from shapely.ops import unary_union
from shapely.prepared import prep

# ============================================================================
# Constants
# ============================================================================

RESOLUTION = 0.5  # degrees per cell
LAT_MIN, LAT_MAX = -90.0, 90.0
LNG_MIN, LNG_MAX = -180.0, 180.0
ROWS = int((LAT_MAX - LAT_MIN) / RESOLUTION)  # 360
COLS = int((LNG_MAX - LNG_MIN) / RESOLUTION)   # 720

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Biome IDs (extended from TerrainType for richer classification)
class Biome:
    OCEAN = 0
    TROPICAL_RAINFOREST = 1
    TROPICAL_SAVANNA = 2
    SUBTROPICAL_DESERT = 3
    TEMPERATE_GRASSLAND = 4
    TEMPERATE_FOREST = 5
    MEDITERRANEAN = 6
    BOREAL_FOREST = 7
    TUNDRA = 8
    ALPINE = 9
    MONSOON_FOREST = 10
    ICE_SHEET = 11

    # Map to simulation TerrainType for backwards compatibility
    TO_TERRAIN = {
        0: 0,   # OCEAN
        1: 2,   # TROPICAL_RAINFOREST -> FOREST
        2: 1,   # TROPICAL_SAVANNA -> PLAINS
        3: 4,   # SUBTROPICAL_DESERT -> DESERT
        4: 1,   # TEMPERATE_GRASSLAND -> PLAINS
        5: 2,   # TEMPERATE_FOREST -> FOREST
        6: 1,   # MEDITERRANEAN -> PLAINS
        7: 2,   # BOREAL_FOREST -> FOREST
        8: 5,   # TUNDRA
        9: 3,   # ALPINE -> MOUNTAINS
        10: 2,  # MONSOON_FOREST -> FOREST
        11: 5,  # ICE_SHEET -> TUNDRA
    }

    NAMES = {
        0: "Ocean", 1: "Tropical Rainforest", 2: "Tropical Savanna",
        3: "Subtropical Desert", 4: "Temperate Grassland", 5: "Temperate Forest",
        6: "Mediterranean", 7: "Boreal Forest (Taiga)", 8: "Tundra",
        9: "Alpine", 10: "Monsoon Forest", 11: "Ice Sheet",
    }


# ============================================================================
# Helper: Coordinate Grids
# ============================================================================

def make_lat_lng_grids():
    """Create 2D grids of latitude and longitude (cell centers)."""
    lats = np.linspace(LAT_MAX - RESOLUTION / 2, LAT_MIN + RESOLUTION / 2, ROWS)
    lngs = np.linspace(LNG_MIN + RESOLUTION / 2, LNG_MAX - RESOLUTION / 2, COLS)
    lng_grid, lat_grid = np.meshgrid(lngs, lats)
    return lat_grid, lng_grid


# ============================================================================
# 1. ELEVATION MODEL
# ============================================================================

def generate_elevation(lat_grid, lng_grid, land_mask):
    """
    Synthetic elevation model combining:
    - Multi-octave Perlin-like noise for continental terrain
    - Known mountain range amplification
    - Coastal gradient (land rises from coast)

    Not as accurate as ETOPO1, but captures major features
    (Himalayas, Andes, Rockies, Alps, East African Rift).

    Ref: General plate tectonics / orogenesis principles.
    """
    print("  Generating elevation model...")
    elev = np.zeros_like(lat_grid)

    # Base continental elevation from noise
    for octave in range(5):
        freq = 2 ** octave * 0.05
        amp = 1.0 / (2 ** octave)
        phase = octave * 1.37
        noise = (
            np.sin(lat_grid * freq + phase) *
            np.cos(lng_grid * freq * 0.7 + phase * 2.1) *
            np.sin((lat_grid + lng_grid) * freq * 0.3 + phase * 0.5)
        )
        elev += noise * amp

    # Normalize to 0-0.15 range for base land (most land is lowland)
    elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-10)
    elev = elev * 0.15  # Base terrain is low: 0 to ~800m equivalent
    elev *= land_mask

    # Amplify known mountain ranges (approximate polygonal regions)
    mountain_ranges = [
        # (lat_center, lng_center, lat_extent, lng_extent, peak_height, name)
        (30, 82, 12, 20, 0.95, "Himalayas/Tibetan Plateau"),
        (45, 10, 4, 12, 0.65, "Alps"),
        (-15, -70, 40, 8, 0.80, "Andes"),
        (45, -112, 18, 12, 0.70, "Rocky Mountains"),
        (0, 35, 18, 10, 0.55, "East African Rift / Kilimanjaro"),
        (58, 60, 15, 5, 0.50, "Urals"),
        (63, 12, 10, 8, 0.55, "Scandinavian Mountains"),
        (33, -5, 5, 12, 0.50, "Atlas Mountains"),
        (42, 44, 6, 6, 0.60, "Caucasus"),
        (-5, 140, 8, 6, 0.55, "New Guinea Highlands"),
        (25, 100, 10, 15, 0.65, "Yunnan-Guizhou Plateau"),
        (-42, 170, 8, 4, 0.50, "Southern Alps NZ"),
    ]

    for lat_c, lng_c, lat_e, lng_e, peak, name in mountain_ranges:
        dist = np.sqrt(
            ((lat_grid - lat_c) / lat_e) ** 2 +
            ((lng_grid - lng_c) / lng_e) ** 2
        )
        mountain_mask = np.exp(-dist ** 2 * 3.0) * peak
        elev = np.maximum(elev, mountain_mask * land_mask)

    # Ensure ocean is low
    elev = np.where(land_mask, np.clip(elev, 0.02, 1.0), 0.0)

    return elev


# ============================================================================
# 2. TEMPERATURE MODEL
# ============================================================================

def generate_temperature(lat_grid, lng_grid, elevation, land_mask):
    """
    Mean annual temperature model.

    Components:
    - Latitude: T = 30 - 0.7 * |lat| (first-order approximation)
      Ref: Hartmann (2016) "Global Physical Climatology", Chapter 2
    - Elevation: -6.5°C per 1000m (environmental lapse rate)
      Ref: ICAO Standard Atmosphere; Barry & Chorley (2010)
    - Continentality: inland areas have more extreme temperatures
      Ref: Conrad (1946) continentality index
    - Ocean current anomalies: Gulf Stream, Kuroshio, cold currents
      Ref: Peixoto & Oort (1992) "Physics of Climate"
    """
    print("  Generating temperature model...")

    # Base temperature from latitude.
    # Calibrated to observed zonal mean annual temperatures (ERA5 reanalysis):
    # 0° ~26°C, 20° ~24°C, 30° ~20°C, 45° ~12°C, 55° ~5°C, 65° ~-3°C, 80° ~-18°C
    # Ref: Hartmann 2016 Fig 2.2, ERA5 climatology 1991-2020.
    # These coefficients reproduce the anchors across the whole range (80° ->
    # -18.6°C). The previous fit (0.15*L + 0.004*L^2) was ~7°C too warm in the
    # polar tail (-10.6°C at 80°), which left Antarctica/Greenland misclassified
    # as grassland/tundra by the T<-15 ice-sheet rule in classify_biomes().
    abs_lat = np.abs(lat_grid)
    temp = 27.0 - 0.05 * abs_lat - 0.0065 * abs_lat ** 2

    # Elevation lapse rate: -6.5°C per 1000m
    # Our elevation scale: 0-1 maps to 0-5500m (mean continental elev ~800m)
    # Only apply to land; most land is <0.3 on our scale
    # Ref: ICAO Standard Atmosphere
    elev_meters = elevation * 5500.0
    temp -= 6.5 * elev_meters / 1000.0 * land_mask

    # Continentality: compute distance to coast (simplified).
    # Inland areas: warmer summers, colder winters -> slightly lower mean.
    # Ref: Conrad (1946). distance_transform_edt measures in grid cells; a degree
    # of longitude shrinks with latitude, so scale by cos(lat) to approximate
    # ground distance (otherwise high-latitude interiors read as artificially far
    # inland). Approximate — it scales the mixed lat/lng EDT step uniformly.
    from scipy.ndimage import distance_transform_edt
    cos_lat = np.cos(np.deg2rad(np.abs(lat_grid)))
    coast_dist = distance_transform_edt(land_mask) * RESOLUTION * cos_lat  # ~degrees of ground distance
    continentality = np.clip(coast_dist / 40.0, 0, 1)  # Normalize, max ~40° inland
    temp -= continentality * 2.0  # Inland areas 0-2°C cooler (mean annual effect)

    # Ocean current warm anomalies
    # Gulf Stream: warms NW Europe +5-8°C above zonal mean
    # Ref: Seager et al. (2002), "Is the Gulf Stream responsible..."
    gulf_stream = np.exp(-(((lat_grid - 55) / 15) ** 2 + ((lng_grid + 5) / 20) ** 2)) * 6.0
    # Kuroshio: warms Japan/Korea
    kuroshio = np.exp(-(((lat_grid - 35) / 10) ** 2 + ((lng_grid - 140) / 15) ** 2)) * 3.0
    # North Atlantic Drift extends warming further
    nad = np.exp(-(((lat_grid - 65) / 10) ** 2 + ((lng_grid + 10) / 25) ** 2)) * 4.0

    # Cold current anomalies
    # Humboldt (Peru): cools west coast of S. America
    # Ref: Thiel et al. (2007)
    humboldt = np.exp(-(((lat_grid + 15) / 20) ** 2 + ((lng_grid + 78) / 8) ** 2)) * -4.0
    # Benguela (SW Africa)
    benguela = np.exp(-(((lat_grid + 20) / 15) ** 2 + ((lng_grid - 12) / 8) ** 2)) * -3.0
    # California Current
    california = np.exp(-(((lat_grid - 35) / 15) ** 2 + ((lng_grid + 122) / 8) ** 2)) * -3.0

    temp += gulf_stream + kuroshio + nad + humboldt + benguela + california

    # Ocean temperature (simplified SST)
    ocean_temp = 27.0 - 0.5 * np.abs(lat_grid) - 0.002 * lat_grid ** 2
    temp = np.where(land_mask, temp, ocean_temp)

    return temp


# ============================================================================
# 3. PRECIPITATION MODEL
# ============================================================================

def generate_precipitation(lat_grid, lng_grid, elevation, land_mask):
    """
    Mean annual precipitation model (mm/yr).

    Components:
    - ITCZ (Intertropical Convergence Zone): heavy rain belt at ~0-10°N
      Ref: Schneider et al. (2014) "Migrations of the ITCZ"
    - Hadley cell subsidence: dry at ~20-35°N/S (subtropical highs)
      Ref: Held & Hou (1980) "Nonlinear Axially Symmetric Circulations"
    - Mid-latitude storm tracks: rain at ~40-60°
      Ref: Hoskins & Valdes (1990)
    - Orographic effect: mountain windward = wet, leeward = dry
      Ref: Roe (2005) "Orographic Precipitation"
    - Monsoon regions: seasonal heavy rainfall
      Ref: Wang & Ding (2008) "Global Monsoon"
    - Continentality: drier inland
      Ref: New et al. (2002) CRU climatology
    """
    print("  Generating precipitation model...")

    abs_lat = np.abs(lat_grid)

    # Base zonal precipitation pattern
    # ITCZ: intense rain near equator (2000-3000 mm/yr)
    itcz = 2500 * np.exp(-((lat_grid - 5) / 8) ** 2)

    # Subtropical dry zones (Hadley cell descending branch)
    # ~200-400 mm/yr at 25-30°
    subtropical_dry = -1500 * np.exp(-((abs_lat - 28) / 8) ** 2)

    # Mid-latitude cyclonic rain (~600-1200 mm/yr at 45-55°)
    midlat_rain = 800 * np.exp(-((abs_lat - 50) / 12) ** 2)

    # Polar dry (~200 mm/yr)
    polar_dry = -400 * np.exp(-((abs_lat - 80) / 15) ** 2)

    precip = 800 + itcz + subtropical_dry + midlat_rain + polar_dry

    # Continentality: drier inland. cos(lat)-scaled to approximate ground
    # distance (see generate_temperature), so high-latitude interiors are not
    # treated as artificially far from the coast.
    from scipy.ndimage import distance_transform_edt
    cos_lat = np.cos(np.deg2rad(np.abs(lat_grid)))
    coast_dist = distance_transform_edt(land_mask) * RESOLUTION * cos_lat
    continentality_dry = np.clip(coast_dist / 20.0, 0, 1) * -400
    precip += continentality_dry * land_mask

    # Maritime enhancement: coastal areas get more rain
    maritime = np.exp(-coast_dist / 3.0) * 300 * land_mask
    precip += maritime

    # Monsoon regions: seasonal heavy rain
    # South Asian Monsoon
    monsoon_sa = np.exp(-(((lat_grid - 22) / 12) ** 2 + ((lng_grid - 82) / 20) ** 2)) * 1200
    # East Asian Monsoon
    monsoon_ea = np.exp(-(((lat_grid - 28) / 10) ** 2 + ((lng_grid - 115) / 15) ** 2)) * 800
    # West African Monsoon
    monsoon_wa = np.exp(-(((lat_grid - 10) / 8) ** 2 + ((lng_grid + 2) / 15) ** 2)) * 600
    # Australian Monsoon
    monsoon_au = np.exp(-(((lat_grid + 15) / 8) ** 2 + ((lng_grid - 135) / 15) ** 2)) * 500

    precip += (monsoon_sa + monsoon_ea + monsoon_wa + monsoon_au) * land_mask

    # Specific wet regions
    # Amazon basin
    amazon = np.exp(-(((lat_grid + 3) / 10) ** 2 + ((lng_grid + 60) / 15) ** 2)) * 1000
    # Congo basin
    congo = np.exp(-(((lat_grid - 0) / 8) ** 2 + ((lng_grid - 22) / 10) ** 2)) * 800
    # SE Asia
    se_asia = np.exp(-(((lat_grid - 5) / 10) ** 2 + ((lng_grid - 110) / 15) ** 2)) * 700

    precip += (amazon + congo + se_asia) * land_mask

    # Orographic enhancement (windward side of mountains)
    elev_gradient = np.gradient(elevation, axis=1)  # E-W gradient
    orographic = np.clip(-elev_gradient * 3000, -500, 800)  # Windward = wet
    precip += orographic * land_mask

    # Specific dry regions (rain shadow, continental interiors)
    # Central Asian deserts
    central_asia_dry = np.exp(-(((lat_grid - 42) / 10) ** 2 + ((lng_grid - 62) / 15) ** 2)) * -600
    # Sahara
    sahara_dry = np.exp(-(((lat_grid - 25) / 8) ** 2 + ((lng_grid - 10) / 25) ** 2)) * -500
    # Arabian
    arabian_dry = np.exp(-(((lat_grid - 23) / 6) ** 2 + ((lng_grid - 48) / 10) ** 2)) * -500
    # Australian interior
    australia_dry = np.exp(-(((lat_grid + 25) / 8) ** 2 + ((lng_grid - 135) / 12) ** 2)) * -500
    # Patagonia rain shadow
    patagonia_dry = np.exp(-(((lat_grid + 45) / 6) ** 2 + ((lng_grid + 68) / 5) ** 2)) * -400

    precip += (central_asia_dry + sahara_dry + arabian_dry + australia_dry + patagonia_dry) * land_mask

    # Clamp to physical bounds
    precip = np.clip(precip, 10, 4500)  # mm/yr; min 10mm (hyperarid), max ~4500 (Cherrapunji)
    precip = np.where(land_mask, precip, 0)

    return precip


# ============================================================================
# 4. BIOME CLASSIFICATION (Whittaker Diagram)
# ============================================================================

def classify_biomes(temperature, precipitation, elevation, land_mask):
    """
    Biome classification using the Whittaker biome diagram.

    Maps (mean_annual_temp, mean_annual_precip) -> biome type.

    Ref: Whittaker (1975) "Communities and Ecosystems"
         Ricklefs (2008) "The Economy of Nature"
         Holdridge (1947) life zone classification (alternative)

    Decision boundaries approximate the classic Whittaker diagram.
    """
    print("  Classifying biomes (Whittaker diagram)...")

    biome = np.full_like(temperature, Biome.OCEAN, dtype=int)
    T = temperature
    P = precipitation
    E = elevation

    # Process only land cells
    land = land_mask.astype(bool)

    # Ice sheet: extremely cold (< -15°C mean annual) or very high + cold
    # Ref: Only Greenland + Antarctica truly have ice sheets
    ice = land & ((T < -15) | ((E > 0.85) & (T < -5)))
    biome[ice] = Biome.ICE_SHEET

    # Tundra: -10 to 0°C, low precipitation
    tundra = land & ~ice & (T < 0) & (T >= -10)
    biome[tundra] = Biome.TUNDRA

    # Alpine: high elevation (>0.6) in non-polar regions
    alpine = land & ~ice & ~tundra & (E > 0.55) & (T < 10)
    biome[alpine] = Biome.ALPINE

    # Boreal forest (Taiga): 0-5°C, >300mm
    # Ref: Whittaker diagram boundary
    boreal = land & ~ice & ~tundra & ~alpine & (T >= 0) & (T < 5) & (P > 300)
    biome[boreal] = Biome.BOREAL_FOREST

    # Still cold but dry -> tundra
    cold_dry = land & ~ice & ~tundra & ~alpine & ~boreal & (T >= 0) & (T < 5) & (P <= 300)
    biome[cold_dry] = Biome.TUNDRA

    # Subtropical desert: warm (>15°C) and dry (<300mm)
    # Ref: Whittaker; BWh/BWk in Köppen-Geiger
    # Widened threshold because Sahara/Arabian extend to lower temps at edges
    desert = land & (T >= 15) & (P < 300)
    biome[desert] = Biome.SUBTROPICAL_DESERT

    # Also cold deserts: 5-15°C and very dry (<200mm) — Gobi, Patagonia
    cold_desert = land & (T >= 0) & (T < 15) & (P < 200) & (biome == Biome.OCEAN)
    biome[cold_desert] = Biome.SUBTROPICAL_DESERT

    # Tropical rainforest: >22°C, >1800mm
    # Ref: Whittaker diagram; Af in Köppen-Geiger
    trop_rain = land & (T >= 22) & (P >= 1800) & (biome == Biome.OCEAN)
    biome[trop_rain] = Biome.TROPICAL_RAINFOREST

    # Monsoon forest: >20°C, 1000-1800mm (seasonal)
    monsoon = land & (T >= 20) & (P >= 1000) & (P < 1800) & (biome == Biome.OCEAN)
    biome[monsoon] = Biome.MONSOON_FOREST

    # Tropical savanna: >18°C, 300-1000mm
    # Ref: Whittaker; Aw in Köppen. Widened to capture African/Brazilian savanna
    savanna = land & (T >= 18) & (P >= 300) & (P < 1000) & (biome == Biome.OCEAN)
    biome[savanna] = Biome.TROPICAL_SAVANNA

    # Mediterranean: 10-20°C, 300-900mm, dry-summer mid-latitudes (Köppen Csa/Csb).
    # abs_lat is a (ROWS, 1) column that broadcasts against the (ROWS, COLS) masks.
    abs_lat = np.abs(
        np.linspace(LAT_MAX - RESOLUTION / 2, LAT_MIN + RESOLUTION / 2, ROWS)
    )[:, None]
    med_mask = (land & (T >= 10) & (T < 20) & (P >= 300) & (P < 900)
                & (abs_lat > 28) & (abs_lat < 45) & (biome == Biome.OCEAN))
    biome[med_mask] = Biome.MEDITERRANEAN

    # Temperate forest: 5-20°C, >600mm
    # Ref: Whittaker; Cf/Cw in Köppen
    temp_forest = land & (T >= 5) & (T < 20) & (P >= 600) & (biome == Biome.OCEAN)
    biome[temp_forest] = Biome.TEMPERATE_FOREST

    # Temperate grassland/steppe: 5-20°C, 200-600mm
    # Ref: Whittaker; BSk in Köppen
    grassland = land & (T >= 5) & (T < 20) & (P >= 200) & (P < 600) & (biome == Biome.OCEAN)
    biome[grassland] = Biome.TEMPERATE_GRASSLAND

    # Fill remaining land cells (edge cases)
    remaining = land & (biome == Biome.OCEAN)
    biome[remaining] = Biome.TEMPERATE_GRASSLAND  # Default fallback

    return biome


# ============================================================================
# 5. SOIL FERTILITY / FOOD PRODUCTIVITY
# ============================================================================

def generate_fertility(biome, temperature, precipitation, land_mask):
    """
    Soil fertility / agricultural productivity index (0-1).

    Based on:
    - Biome type (tropical soils are often nutrient-poor despite growth)
    - Precipitation (need >500mm for rain-fed agriculture)
    - Temperature (optimal 15-25°C for most crops)
    - Known breadbasket regions

    Ref: FAO GAEZ (Global Agro-Ecological Zones)
         Licker et al. (2010) "Mind the gap" crop yield analysis
         Mueller et al. (2012) "Closing yield gaps"
    """
    print("  Generating soil fertility model...")

    fertility = np.zeros_like(temperature)

    # Base fertility by biome
    # Ref: FAO soil classification & productivity estimates
    biome_fertility = {
        Biome.OCEAN: 0.0,
        Biome.TROPICAL_RAINFOREST: 0.4,    # Oxisols: lush but nutrient-poor
        Biome.TROPICAL_SAVANNA: 0.55,      # Better for agriculture
        Biome.SUBTROPICAL_DESERT: 0.05,     # Almost nothing without irrigation
        Biome.TEMPERATE_GRASSLAND: 0.75,    # Mollisols: world's best agricultural soils
        Biome.TEMPERATE_FOREST: 0.65,       # Good when cleared
        Biome.MEDITERRANEAN: 0.55,          # Good with irrigation
        Biome.BOREAL_FOREST: 0.20,          # Spodosols: acidic, poor
        Biome.TUNDRA: 0.05,
        Biome.ALPINE: 0.05,
        Biome.MONSOON_FOREST: 0.60,         # Good seasonal agriculture
        Biome.ICE_SHEET: 0.0,
    }

    for biome_id, base_fert in biome_fertility.items():
        fertility[biome == biome_id] = base_fert

    # Precipitation modifier: need 400-1500mm for optimal agriculture
    # Ref: FAO GAEZ water requirement thresholds
    precip_factor = np.where(
        precipitation < 200, 0.1,
        np.where(precipitation < 500, 0.3 + 0.7 * (precipitation - 200) / 300,
        np.where(precipitation < 1500, 1.0,
        np.where(precipitation < 3000, 1.0 - 0.3 * (precipitation - 1500) / 1500,
        0.5)))  # Excessive rain = leaching
    )
    fertility *= precip_factor

    # Temperature modifier: optimal 12-28°C
    # Ref: Schlenker & Roberts (2009) nonlinear crop response
    temp_factor = np.where(
        temperature < 0, 0.0,
        np.where(temperature < 12, temperature / 12.0,
        np.where(temperature < 28, 1.0,
        np.where(temperature < 38, 1.0 - (temperature - 28) / 10.0,
        0.0)))
    )
    fertility *= temp_factor

    # Known breadbasket amplification
    # Ref: Monfreda et al. (2008) global crop harvested area
    breadbaskets = [
        (42, -95, 8, 15, 1.3, "US Great Plains/Corn Belt"),
        (50, 40, 8, 20, 1.2, "Ukrainian/Russian Black Earth"),
        (48, -105, 6, 10, 1.2, "Canadian Prairies"),
        (-30, -58, 8, 10, 1.2, "Argentine Pampas"),
        (30, 78, 8, 8, 1.3, "Indo-Gangetic Plain"),
        (33, 115, 8, 12, 1.2, "North China Plain"),
        (-25, 28, 6, 6, 1.1, "South African Highveld"),
        (48, 3, 5, 8, 1.2, "Northern France"),
    ]
    for lat_c, lng_c, lat_e, lng_e, mult, name in breadbaskets:
        lat_grid_2d = np.broadcast_to(
            np.linspace(LAT_MAX - RESOLUTION/2, LAT_MIN + RESOLUTION/2, ROWS)[:, None],
            (ROWS, COLS)
        )
        lng_grid_2d = np.broadcast_to(
            np.linspace(LNG_MIN + RESOLUTION/2, LNG_MAX - RESOLUTION/2, COLS)[None, :],
            (ROWS, COLS)
        )
        dist = np.sqrt(((lat_grid_2d - lat_c) / lat_e) ** 2 + ((lng_grid_2d - lng_c) / lng_e) ** 2)
        boost = np.exp(-dist ** 2 * 2.0) * (mult - 1.0)
        fertility += boost * land_mask

    fertility = np.clip(fertility, 0.0, 1.0) * land_mask
    return fertility


# ============================================================================
# 6. MINERAL RICHNESS
# ============================================================================

def generate_minerals(elevation, land_mask):
    """
    Mineral deposit potential index (0-1).

    Based on:
    - Proximity to mountain ranges (orogenic belts = mineral-rich)
    - Known mineral-rich geological provinces
    - Tectonic plate boundary proxy (elevation gradient)

    Ref: USGS Mineral Resources data
         Arndt et al. (2017) "Future mineral resources"
         Kesler & Simon (2015) "Mineral Resources"
    """
    print("  Generating mineral richness model...")

    minerals = np.zeros_like(elevation)

    # Mountains = orogenic mineral deposits (Cu, Au, Ag, Mo)
    # Ref: Sillitoe (2010) porphyry copper deposits
    minerals += np.clip(elevation - 0.3, 0, 1) * 0.6

    # Elevation gradients = tectonic activity = minerals
    grad_x = np.abs(np.gradient(elevation, axis=1))
    grad_y = np.abs(np.gradient(elevation, axis=0))
    tectonic = np.clip((grad_x + grad_y) * 5.0, 0, 0.5)
    minerals += tectonic

    # Known mineral-rich provinces
    lat_grid, lng_grid = make_lat_lng_grids()
    provinces = [
        (-25, 28, 5, 5, 0.9, "South African Bushveld (PGMs, Cr, V)"),
        (-20, 120, 10, 15, 0.7, "Australian Pilbara/Yilgarn (Fe, Au)"),
        (-15, -65, 10, 10, 0.7, "Andean copper belt"),
        (60, 90, 15, 30, 0.6, "Siberian mineral belt"),
        (50, -85, 10, 15, 0.6, "Canadian Shield (Ni, Cu, Au)"),
        (-10, 28, 10, 8, 0.7, "Central African Copperbelt"),
        (35, 55, 8, 10, 0.5, "Iranian/Turkish mineral belt"),
        (25, 110, 8, 8, 0.5, "Chinese rare earth deposits"),
        (-22, -50, 6, 6, 0.5, "Brazilian iron quadrangle"),
        (65, 25, 5, 10, 0.5, "Scandinavian Shield"),
    ]
    for lat_c, lng_c, lat_e, lng_e, richness, name in provinces:
        dist = np.sqrt(((lat_grid - lat_c) / lat_e) ** 2 + ((lng_grid - lng_c) / lng_e) ** 2)
        minerals += np.exp(-dist ** 2 * 2.0) * richness

    minerals = np.clip(minerals, 0.0, 1.0) * land_mask
    return minerals


# ============================================================================
# 7. FRESHWATER AVAILABILITY
# ============================================================================

def generate_freshwater(precipitation, land_mask):
    """
    Freshwater availability index (0-1).

    Based on:
    - Precipitation (primary driver)
    - River/lake proximity from Natural Earth data
    - Known aquifer regions

    Ref: Döll et al. (2003) global groundwater model
         Vörösmarty et al. (2010) global freshwater threats
    """
    print("  Generating freshwater model...")

    # Base: precipitation-driven
    # >1000mm/yr = generally water-rich; <200mm = water-stressed
    freshwater = np.clip((precipitation - 100) / 1500, 0, 1)

    # Enhance from rivers and lakes (Natural Earth data)
    river_proximity = _compute_feature_proximity(
        os.path.join(DATA_DIR, "ne_110m_rivers.geojson"), "rivers"
    )
    lake_proximity = _compute_feature_proximity(
        os.path.join(DATA_DIR, "ne_110m_lakes.geojson"), "lakes"
    )

    # River proximity boost (within ~2° of major river)
    river_boost = np.clip(1.0 - river_proximity / 3.0, 0, 1) * 0.4
    freshwater += river_boost * land_mask

    # Lake proximity boost
    lake_boost = np.clip(1.0 - lake_proximity / 2.0, 0, 1) * 0.3
    freshwater += lake_boost * land_mask

    # Known major aquifer regions
    lat_grid, lng_grid = make_lat_lng_grids()
    aquifers = [
        (35, -100, 10, 10, 0.2, "Ogallala Aquifer"),
        (25, 30, 10, 15, 0.15, "Nubian Sandstone Aquifer"),
        (-25, 135, 10, 10, 0.15, "Great Artesian Basin"),
        (55, 75, 10, 15, 0.15, "West Siberian Aquifer"),
        (28, 78, 8, 8, 0.2, "Indo-Gangetic aquifer"),
    ]
    for lat_c, lng_c, lat_e, lng_e, boost, name in aquifers:
        dist = np.sqrt(((lat_grid - lat_c) / lat_e) ** 2 + ((lng_grid - lng_c) / lng_e) ** 2)
        freshwater += np.exp(-dist ** 2 * 2.0) * boost * land_mask

    freshwater = np.clip(freshwater, 0.0, 1.0) * land_mask
    return freshwater


def _compute_feature_proximity(geojson_path, feature_type):
    """Compute distance grid to nearest GeoJSON feature."""
    if not os.path.exists(geojson_path):
        print(f"    Warning: {geojson_path} not found, using default")
        return np.full((ROWS, COLS), 999.0)

    with open(geojson_path) as f:
        data = json.load(f)

    # Rasterize: mark cells that contain or are near features
    feature_mask = np.zeros((ROWS, COLS), dtype=bool)
    lat_grid, lng_grid = make_lat_lng_grids()

    for feature in data["features"]:
        geom = shape(feature["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        # Mark cells near this feature
        bounds = geom.bounds  # (minx, miny, maxx, maxy) = (min_lng, min_lat, max_lng, max_lat)
        r_min = max(0, int((LAT_MAX - bounds[3]) / RESOLUTION) - 2)
        r_max = min(ROWS, int((LAT_MAX - bounds[1]) / RESOLUTION) + 2)
        c_min = max(0, int((bounds[0] - LNG_MIN) / RESOLUTION) - 2)
        c_max = min(COLS, int((bounds[2] - LNG_MIN) / RESOLUTION) + 2)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                pt = Point(lng_grid[r, c], lat_grid[r, c])
                if geom.distance(pt) < RESOLUTION:
                    feature_mask[r, c] = True

    # Distance transform from feature cells
    from scipy.ndimage import distance_transform_edt
    if feature_mask.any():
        dist = distance_transform_edt(~feature_mask) * RESOLUTION
    else:
        dist = np.full((ROWS, COLS), 999.0)

    return dist


# ============================================================================
# 8. FOSSIL FUEL POTENTIAL
# ============================================================================

def generate_fossil_fuels(land_mask):
    """
    Fossil fuel deposit potential (0-1).

    Based on known sedimentary basin locations.
    Ref: USGS World Petroleum Assessment
         BGR Energy Study (2019)
    """
    print("  Generating fossil fuel potential...")

    fossil = np.zeros((ROWS, COLS))
    lat_grid, lng_grid = make_lat_lng_grids()

    basins = [
        (28, 50, 8, 15, 0.9, "Persian Gulf"),
        (60, 70, 10, 20, 0.8, "West Siberia"),
        (30, -95, 10, 10, 0.7, "Gulf of Mexico / Texas"),
        (55, -115, 8, 10, 0.7, "Alberta Oil Sands"),
        (5, 5, 8, 8, 0.7, "Niger Delta"),
        (-5, -55, 8, 10, 0.5, "Pre-salt Brazil"),
        (35, 50, 6, 8, 0.6, "Caspian region"),
        (10, 108, 5, 5, 0.4, "South China Sea"),
        (-20, 45, 5, 5, 0.3, "Mozambique Channel"),
        (58, 5, 5, 5, 0.6, "North Sea"),
        (25, 80, 8, 8, 0.3, "Indian subcontinent"),
        (35, 115, 8, 10, 0.5, "Chinese coal basins"),
        (-32, 150, 5, 5, 0.4, "Australian coal"),
        (50, 5, 5, 8, 0.4, "European coal belt"),
    ]

    for lat_c, lng_c, lat_e, lng_e, richness, name in basins:
        dist = np.sqrt(((lat_grid - lat_c) / lat_e) ** 2 + ((lng_grid - lng_c) / lng_e) ** 2)
        fossil += np.exp(-dist ** 2 * 2.0) * richness

    fossil = np.clip(fossil, 0.0, 1.0)
    # Fossil fuels can be offshore too, so don't mask to land only
    return fossil


# ============================================================================
# MAIN: Generate All Grids
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("EARTH SYSTEM DATA GENERATOR")
    print(f"Resolution: {RESOLUTION}° ({ROWS} x {COLS} grid)")
    print("=" * 70)

    # Load land mask (resample from 0.25° to 0.5°)
    print("\n[1/8] Loading land mask...")
    mask_025 = np.load(os.path.join(DATA_DIR, "landmask.npy"))
    # Downsample by taking every 2nd cell (0.25° -> 0.5°)
    land_mask = mask_025[::2, ::2].astype(np.float32)
    # Handle edge: if shapes don't match exactly, crop
    land_mask = land_mask[:ROWS, :COLS]
    print(f"  Land mask: {land_mask.shape}, land fraction: {land_mask.mean()*100:.1f}%")

    lat_grid, lng_grid = make_lat_lng_grids()

    # Generate all layers
    print("\n[2/8] Elevation...")
    elevation = generate_elevation(lat_grid, lng_grid, land_mask)

    print("\n[3/8] Temperature...")
    temperature = generate_temperature(lat_grid, lng_grid, elevation, land_mask)

    print("\n[4/8] Precipitation...")
    precipitation = generate_precipitation(lat_grid, lng_grid, elevation, land_mask)

    print("\n[5/8] Biome classification...")
    biome = classify_biomes(temperature, precipitation, elevation, land_mask)

    # Convert biome to simulation terrain type
    terrain = np.vectorize(lambda b: Biome.TO_TERRAIN.get(b, 0))(biome)

    print("\n[6/8] Soil fertility...")
    fertility = generate_fertility(biome, temperature, precipitation, land_mask)

    print("\n[7/8] Mineral richness...")
    minerals = generate_minerals(elevation, land_mask)

    print("\n[8/8] Freshwater + Fossil fuels...")
    freshwater = generate_freshwater(precipitation, land_mask)
    fossil_fuels = generate_fossil_fuels(land_mask)

    # Save all grids
    print("\nSaving grids...")
    grids = {
        "elevation": elevation,
        "temperature": temperature,
        "precipitation": precipitation,
        "biome": biome.astype(np.int8),
        "terrain": terrain.astype(np.int8),
        "fertility": fertility,
        "minerals": minerals,
        "freshwater": freshwater,
        "fossil_fuels": fossil_fuels,
    }

    for name, grid in grids.items():
        path = os.path.join(DATA_DIR, f"earth_{name}.npy")
        np.save(path, grid)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name}: {grid.shape}, dtype={grid.dtype}, {size_kb:.0f} KB")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    land_cells = int(land_mask.sum())
    print(f"Land cells: {land_cells:,} / {ROWS*COLS:,} ({land_cells/(ROWS*COLS)*100:.1f}%)")
    print(f"\nBiome distribution (land only):")
    for biome_id in sorted(Biome.NAMES.keys()):
        if biome_id == 0:
            continue
        count = int((biome == biome_id).sum())
        if count > 0:
            pct = count / land_cells * 100
            print(f"  {Biome.NAMES[biome_id]:30s}: {count:6d} cells ({pct:5.1f}%)")

    print(f"\nTemperature range (land): {temperature[land_mask > 0].min():.1f}°C to {temperature[land_mask > 0].max():.1f}°C")
    print(f"Precipitation range (land): {precipitation[land_mask > 0].min():.0f} to {precipitation[land_mask > 0].max():.0f} mm/yr")
    print(f"Mean fertility (land): {fertility[land_mask > 0].mean():.3f}")
    print(f"Mean mineral richness (land): {minerals[land_mask > 0].mean():.3f}")

    elapsed = time.time() - t_start
    print(f"\nTotal generation time: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
