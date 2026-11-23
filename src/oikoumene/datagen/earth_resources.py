"""
Earth system grids, part 2: soil fertility, minerals, freshwater and fossil fuels.

Resource-layer models used by ``oikoumene.datagen.generate_earth_data``. Grid
constants, ``Biome`` and ``make_lat_lng_grids`` come from
``oikoumene.datagen.earth_climate``. Split out of ``generate_earth_data`` unchanged.
"""

import json
import os

import numpy as np
from shapely.geometry import shape, Point, LineString, MultiLineString
from shapely.ops import unary_union
from shapely.prepared import prep

from oikoumene.datagen.earth_climate import (
    Biome,
    COLS,
    LAT_MAX,
    LAT_MIN,
    LNG_MAX,
    LNG_MIN,
    RESOLUTION,
    ROWS,
    make_lat_lng_grids,
)
from oikoumene.paths import DATA_DIR


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
