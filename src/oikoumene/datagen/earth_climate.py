"""
Earth system grids, part 1: grid constants, biome IDs and the climate layers.

Holds the 0.5° grid constants, ``Biome``, ``make_lat_lng_grids`` and the
elevation, temperature, precipitation and biome-classification models used by
``oikoumene.datagen.generate_earth_data``. Split out of that module unchanged.
"""

import numpy as np

# ============================================================================
# Constants
# ============================================================================

RESOLUTION = 0.5  # degrees per cell
LAT_MIN, LAT_MAX = -90.0, 90.0
LNG_MIN, LNG_MAX = -180.0, 180.0
ROWS = int((LAT_MAX - LAT_MIN) / RESOLUTION)  # 360
COLS = int((LNG_MAX - LNG_MIN) / RESOLUTION)   # 720


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
