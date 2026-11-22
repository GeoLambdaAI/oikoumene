#!/usr/bin/env python3
"""
Download and process real-world data for present-day scenario initialization.

Run once before using Scenario B. Downloads freely available datasets,
processes to simulation grid (2° resolution), saves to data/.

Data sources (all free, no API key):
1. Population:     World Bank API SP.POP.TOTL + grid allocation
2. GDP/Economics:  World Bank Open Data REST API
3. Climate:        NOAA GML (Mauna Loa CO2) + NASA GISS (temperature)
4. Energy:         World Bank API (renewable/fossil %)
5. Military:       World Bank API (SIPRI data)
6. Conflicts:      Curated list of active conflicts (2025)
7. Agriculture:    World Bank API (arable land, yield)
8. Countries:      Natural Earth centroids + World Bank metadata

Output: data/present_day_*.{npy,json,npz}
"""

import os
import json
import time
import numpy as np
import requests
from datetime import date

from oikoumene.paths import DATA_DIR

GRID_LAT_MIN, GRID_LAT_MAX = -60.0, 75.0
GRID_LNG_MIN, GRID_LNG_MAX = -180.0, 180.0
GRID_CELL_SIZE = 2.0
GRID_ROWS = int((GRID_LAT_MAX - GRID_LAT_MIN) / GRID_CELL_SIZE)  # 68
GRID_COLS = int((GRID_LNG_MAX - GRID_LNG_MIN) / GRID_CELL_SIZE)  # 180

WB_BASE = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=400&date=2019:2025&mrv=1"

# World Bank aggregate region codes to exclude (not real countries)
WB_AGGREGATES = {
    "AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA", "ECS",
    "EMU", "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA", "IDB", "IDX",
    "INX", "LAC", "LCN", "LDC", "LIC", "LMC", "LMY", "LTE", "MEA", "MIC",
    "MNA", "NAC", "OED", "OSS", "PRE", "PSS", "PST", "SAS", "SSA", "SSF",
    "SST", "TEA", "TEC", "TLA", "TMN", "TSA", "TSS", "UMC", "WLD",
    "1A", "1W", "4E", "7E", "8S", "B8", "EU", "F1", "S1", "S2", "S3",
    "S4", "T2", "T3", "T4", "T5", "T6", "T7", "V1", "V2", "V3", "V4",
    "XC", "XD", "XE", "XF", "XG", "XH", "XI", "XJ", "XL", "XM", "XN",
    "XO", "XP", "XQ", "XT", "XU", "XY", "Z4", "Z7", "ZB", "ZF", "ZG",
    "ZH", "ZI", "ZJ", "ZQ", "ZT",
}

# Known country centroids (lat, lng) for major countries
# Used for grid allocation when Natural Earth data isn't enough
COUNTRY_CENTROIDS = {
    "CHN": (35.0, 105.0), "IND": (22.0, 79.0), "USA": (39.0, -98.0),
    "IDN": (-2.0, 118.0), "PAK": (30.0, 70.0), "BRA": (-10.0, -52.0),
    "NGA": (9.5, 8.0), "BGD": (24.0, 90.0), "RUS": (60.0, 90.0),
    "MEX": (23.0, -102.0), "JPN": (36.0, 138.0), "ETH": (9.0, 38.5),
    "PHL": (12.0, 122.0), "EGY": (26.5, 30.0), "VNM": (16.0, 108.0),
    "COD": (-2.5, 23.5), "DEU": (51.0, 10.0), "TUR": (39.0, 35.0),
    "IRN": (32.5, 53.5), "GBR": (54.0, -2.0), "FRA": (46.5, 2.5),
    "THA": (15.5, 101.0), "ITA": (42.5, 12.5), "ZAF": (-29.0, 25.0),
    "TZA": (-6.5, 35.0), "MMR": (19.8, 96.2), "KEN": (-0.5, 37.5),
    "KOR": (36.5, 128.0), "COL": (4.0, -72.0), "ESP": (40.0, -3.5),
    "ARG": (-34.0, -64.0), "DZA": (28.0, 3.0), "SDN": (15.5, 32.5),
    "UKR": (49.0, 32.0), "IRQ": (33.0, 44.0), "AFG": (33.5, 66.0),
    "POL": (52.0, 20.0), "CAN": (56.0, -96.0), "MAR": (32.0, -6.0),
    "SAU": (24.0, 45.0), "UZB": (41.5, 64.5), "PER": (-10.0, -76.0),
    "AGO": (-12.5, 18.5), "MYS": (4.0, 109.5), "MOZ": (-18.0, 35.0),
    "GHA": (7.5, -1.0), "YEM": (15.5, 48.0), "NPL": (28.0, 84.0),
    "VEN": (7.0, -66.0), "MDG": (-19.0, 47.0), "CMR": (6.0, 12.5),
    "CIV": (7.5, -5.5), "NER": (17.5, 8.0), "AUS": (-25.0, 134.0),
    "TWN": (23.5, 121.0), "MLI": (17.5, -4.0), "BFA": (12.5, -1.5),
    "LKA": (7.5, 80.5), "MWI": (-13.5, 34.0), "CHL": (-35.0, -71.0),
    "ZMB": (-15.0, 28.0), "KAZ": (48.0, 68.0), "TCD": (15.0, 19.0),
    "SOM": (5.0, 46.0), "SEN": (14.5, -14.5), "ZWE": (-19.0, 29.5),
    "GIN": (11.0, -10.5), "RWA": (-2.0, 29.5), "BDI": (-3.5, 30.0),
    "TUN": (34.0, 9.0), "BEL": (50.5, 4.5), "BOL": (-17.0, -65.0),
    "HTI": (19.0, -72.5), "CUB": (22.0, -79.5), "SSD": (7.5, 30.0),
    "DOM": (19.0, -70.0), "CZE": (50.0, 15.5), "GRC": (39.0, 22.0),
    "JOR": (31.0, 36.5), "PRT": (39.5, -8.0), "AZE": (40.5, 50.0),
    "SWE": (63.0, 16.0), "HUN": (47.0, 20.0), "BLR": (53.5, 28.0),
    "ARE": (24.0, 54.0), "HND": (15.0, -86.5), "TJK": (39.0, 71.0),
    "AUT": (47.5, 14.5), "ISR": (31.5, 35.0), "CHE": (47.0, 8.2),
    "PNG": (-6.0, 147.0), "SLE": (8.5, -12.0), "TGO": (8.5, 1.2),
    "HKG": (22.3, 114.2), "PRY": (-23.0, -58.0), "LAO": (18.0, 105.0),
    "LBY": (27.0, 17.0), "SLV": (13.8, -89.0), "NIC": (13.0, -85.0),
    "KGZ": (41.5, 75.0), "LBN": (34.0, 36.0), "TKM": (39.0, 60.0),
    "SGP": (1.3, 103.8), "DNK": (56.0, 10.0), "FIN": (64.0, 26.0),
    "NOR": (64.0, 11.0), "NZL": (-42.0, 174.0), "LBR": (6.5, -9.5),
    "PAN": (9.0, -80.0), "CRI": (10.0, -84.0), "IRL": (53.5, -8.0),
    "CAF": (7.0, 21.0), "COG": (-1.0, 15.5), "OMN": (21.5, 57.0),
    "NLD": (52.0, 5.5), "MNG": (47.0, 105.0), "BIH": (44.0, 17.8),
    "ALB": (41.0, 20.0), "LTU": (55.5, 24.0), "UGA": (1.5, 32.5),
    "GAB": (-0.5, 11.5), "JAM": (18.1, -77.3), "QAT": (25.3, 51.2),
    "NAM": (-22.0, 17.0), "BWA": (-22.0, 24.0), "ARM": (40.0, 45.0),
    "GEO": (42.0, 43.5), "MDA": (47.0, 29.0), "HRV": (45.0, 16.0),
    "KWT": (29.5, 47.8), "ERI": (15.5, 39.0), "MRT": (20.0, -10.5),
    "ECU": (-1.5, -78.5), "GTM": (15.5, -90.5), "URY": (-33.0, -56.0),
    "SVK": (48.7, 19.7), "SRB": (44.0, 21.0), "ROU": (46.0, 25.0),
    "BGR": (43.0, 25.5), "EST": (59.0, 25.0), "LVA": (57.0, 25.0),
}


# ============================================================================
# World Bank API Fetcher
# ============================================================================

def fetch_wb_indicator(indicator_code: str, label: str = "") -> dict:
    """Fetch most recent value per country from World Bank API."""
    url = WB_BASE.format(indicator=indicator_code)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    WARNING: Failed to fetch {label or indicator_code}: {e}")
        return {}

    result = {}
    if len(data) > 1 and data[1]:
        for entry in data[1]:
            iso3 = entry.get("countryiso3code", "")
            val = entry.get("value")
            if val is not None and iso3 and len(iso3) == 3 and iso3 not in WB_AGGREGATES:
                result[iso3] = val
    return result


def fetch_all_wb_indicators() -> dict:
    """Fetch all needed World Bank indicators."""
    indicators = {
        "SP.POP.TOTL": "population",
        "NY.GDP.MKTP.CD": "gdp_usd",
        "NY.GDP.PCAP.PP.CD": "gdp_per_capita_ppp",
        "SI.POV.GINI": "gini_index",
        "EG.FEC.RNEW.ZS": "renewable_pct",
        "EG.USE.COMM.FO.ZS": "fossil_fuel_pct",
        "MS.MIL.XPND.GD.ZS": "military_pct_gdp",
        "SP.DYN.LE00.IN": "life_expectancy",
        "IT.NET.USER.ZS": "internet_pct",
        "SE.ADT.LITR.ZS": "literacy_rate",
        "AG.LND.ARBL.ZS": "arable_land_pct",
        "EN.ATM.CO2E.PC": "co2_per_capita",
        "NE.TRD.GNFS.ZS": "trade_pct_gdp",
        "NV.AGR.TOTL.ZS": "agriculture_pct_gdp",
        "GB.XPD.RSDV.GD.ZS": "research_pct_gdp",
    }

    all_data = {}
    for code, label in indicators.items():
        print(f"    Fetching {label}...")
        all_data[label] = fetch_wb_indicator(code, label)
        time.sleep(0.3)  # Rate limiting courtesy

    return all_data


# ============================================================================
# Climate Data
# ============================================================================

def fetch_climate_state() -> dict:
    """Fetch current CO2 and temperature from NOAA/NASA."""
    climate = {
        "co2_ppm": 425.0,
        "temperature_anomaly": 1.3,
        "sea_level_rise_m": 0.10,
        "date_fetched": date.today().isoformat(),
    }

    # CO2 from NOAA Mauna Loa
    try:
        resp = requests.get(
            "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt",
            timeout=15)
        lines = [l for l in resp.text.split('\n') if l.strip() and not l.startswith('#')]
        last = lines[-1].split()
        climate["co2_ppm"] = float(last[1])
        print(f"    CO2: {climate['co2_ppm']} ppm ({last[0]})")
    except Exception as e:
        print(f"    WARNING: CO2 fetch failed, using default: {e}")

    # Temperature from NASA GISS
    try:
        resp = requests.get(
            "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
            timeout=15)
        lines = resp.text.strip().split('\n')
        for line in reversed(lines):
            parts = line.split(',')
            if len(parts) > 13 and parts[0].isdigit():
                jan_dec = parts[13].strip()
                if jan_dec and jan_dec != '***':
                    climate["temperature_anomaly"] = float(jan_dec)
                    print(f"    Temperature: +{climate['temperature_anomaly']}°C ({parts[0]})")
                    break
    except Exception as e:
        print(f"    WARNING: Temperature fetch failed, using default: {e}")

    return climate


# ============================================================================
# Active Conflicts (curated, LAST_UPDATED: 2025-03)
# ============================================================================

# LAST_UPDATED: 2025-03-29
ACTIVE_CONFLICTS = [
    {"name": "Russia-Ukraine War", "lat": 48.5, "lng": 37.5,
     "radius_deg": 5.0, "intensity": 0.9, "parties": ["Russia", "Ukraine"]},
    {"name": "Sudan Civil War", "lat": 15.5, "lng": 32.5,
     "radius_deg": 4.0, "intensity": 0.8, "parties": ["SAF", "RSF"]},
    {"name": "Gaza/Israel Conflict", "lat": 31.4, "lng": 34.4,
     "radius_deg": 1.5, "intensity": 0.9, "parties": ["Israel", "Hamas"]},
    {"name": "Myanmar Civil War", "lat": 19.8, "lng": 96.2,
     "radius_deg": 4.0, "intensity": 0.7, "parties": ["Junta", "NUG/EAO"]},
    {"name": "Sahel Insurgency", "lat": 14.0, "lng": 2.0,
     "radius_deg": 6.0, "intensity": 0.6, "parties": ["JNIM", "ISGS", "Govt forces"]},
    {"name": "Ethiopia (various)", "lat": 9.0, "lng": 38.7,
     "radius_deg": 3.0, "intensity": 0.5, "parties": ["Fano", "ENDF", "OLA"]},
    {"name": "Yemen Civil War", "lat": 15.4, "lng": 44.2,
     "radius_deg": 3.0, "intensity": 0.6, "parties": ["Houthis", "Saudi Coalition"]},
    {"name": "Haiti Crisis", "lat": 18.5, "lng": -72.3,
     "radius_deg": 1.0, "intensity": 0.5, "parties": ["Gangs", "Govt"]},
    {"name": "DR Congo (Eastern)", "lat": -1.5, "lng": 29.0,
     "radius_deg": 3.0, "intensity": 0.7, "parties": ["M23", "FARDC", "Militias"]},
    {"name": "Somalia (Al-Shabaab)", "lat": 2.0, "lng": 45.0,
     "radius_deg": 4.0, "intensity": 0.6, "parties": ["Al-Shabaab", "SNA/ATMIS"]},
]


# ============================================================================
# Country Data Assembly
# ============================================================================

def build_country_data(wb_data: dict) -> list[dict]:
    """Assemble per-country data from World Bank indicators."""
    pop_data = wb_data.get("population", {})
    countries = []

    for iso3, pop in pop_data.items():
        if not pop or pop < 100000:  # Skip tiny entities
            continue

        centroid = COUNTRY_CENTROIDS.get(iso3)
        if centroid is None:
            continue  # Skip countries without known centroids

        gdp = wb_data.get("gdp_usd", {}).get(iso3, 0) or 0
        gdp_pc = wb_data.get("gdp_per_capita_ppp", {}).get(iso3, 5000) or 5000
        gini = wb_data.get("gini_index", {}).get(iso3, 40) or 40
        renewable = wb_data.get("renewable_pct", {}).get(iso3, 15) or 15
        fossil = wb_data.get("fossil_fuel_pct", {}).get(iso3, 70) or 70
        military = wb_data.get("military_pct_gdp", {}).get(iso3, 2.0) or 2.0
        life_exp = wb_data.get("life_expectancy", {}).get(iso3, 65) or 65
        internet = wb_data.get("internet_pct", {}).get(iso3, 50) or 50
        literacy = wb_data.get("literacy_rate", {}).get(iso3, 80) or 80
        arable = wb_data.get("arable_land_pct", {}).get(iso3, 10) or 10
        co2_pc = wb_data.get("co2_per_capita", {}).get(iso3, 4) or 4
        trade = wb_data.get("trade_pct_gdp", {}).get(iso3, 50) or 50
        agri = wb_data.get("agriculture_pct_gdp", {}).get(iso3, 10) or 10
        research = wb_data.get("research_pct_gdp", {}).get(iso3, 0.5) or 0.5

        # Derived scores (normalized 0-1)
        tech_level = float(np.clip(
            (internet / 100) * 0.4 + (gdp_pc / 80000) * 0.3 +
            (literacy / 100) * 0.2 + (research / 4) * 0.1, 0, 1))

        military_power = float(np.clip(
            (military / 100) * (gdp / 1e12) * 10 + (pop / 1e9) * 0.1, 0, 1))

        governance = float(np.clip(
            (life_exp / 85) * 0.4 + (1 - gini / 100) * 0.3 +
            (internet / 100) * 0.3, 0, 1))

        countries.append({
            "iso3": iso3,
            "name": iso3,  # Will be enriched later
            "lat": centroid[0],
            "lng": centroid[1],
            "population": int(pop),
            "gdp_usd": gdp,
            "gdp_per_capita_ppp": gdp_pc,
            "gini_index": gini,
            "renewable_pct": renewable,
            "fossil_fuel_pct": fossil,
            "military_pct_gdp": military,
            "life_expectancy": life_exp,
            "internet_pct": internet,
            "literacy_rate": literacy,
            "arable_land_pct": arable,
            "co2_per_capita": co2_pc,
            "trade_pct_gdp": trade,
            "agriculture_pct_gdp": agri,
            "research_pct_gdp": research,
            # Derived
            "technology_level": tech_level,
            "military_power": military_power,
            "governance_quality": governance,
            "energy_profile": {"fossil": fossil, "renewable": renewable},
        })

    return countries


# ============================================================================
# Population Grid
# ============================================================================

def build_population_grid(countries: list[dict]) -> np.ndarray:
    """
    Distribute country populations onto simulation grid,
    weighted by land fertility.
    """
    # Load grids
    landmask_path = DATA_DIR / "landmask.npy"
    fertility_path = DATA_DIR / "earth_fertility.npy"

    if not landmask_path.exists():
        print("    WARNING: landmask.npy not found, using empty grid")
        return np.zeros((GRID_ROWS, GRID_COLS))

    landmask_025 = np.load(landmask_path)
    # Downsample 0.25° -> 2° (take max of 8x8 blocks)
    # From 720x1440 to 68x180 (lat range -60 to 75 = 135° / 2° = 67.5 -> 68)
    # Offset: row 0 at 90°N, but our grid starts at 75°N (row 60 in 0.25° grid)
    offset_row = int((90.0 - GRID_LAT_MAX) / 0.25)  # 60
    landmask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            r025 = offset_row + r * 8
            c025 = c * 8
            if r025 + 8 <= landmask_025.shape[0] and c025 + 8 <= landmask_025.shape[1]:
                landmask[r, c] = landmask_025[r025:r025+8, c025:c025+8].any()

    # Load fertility if available
    if fertility_path.exists():
        fert_05 = np.load(fertility_path)
        # Resample 0.5° (360x720) -> 2° (68x180)
        offset_fert_row = int((90.0 - GRID_LAT_MAX) / 0.5)
        fertility = np.zeros((GRID_ROWS, GRID_COLS))
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                r05 = offset_fert_row + r * 4
                c05 = c * 4
                if r05 + 4 <= fert_05.shape[0] and c05 + 4 <= fert_05.shape[1]:
                    fertility[r, c] = fert_05[r05:r05+4, c05:c05+4].mean()
    else:
        fertility = np.ones((GRID_ROWS, GRID_COLS)) * 0.3

    # Weight = landmask * (fertility + 0.1) to avoid zero weights
    weight = landmask.astype(float) * (fertility + 0.1)

    pop_grid = np.zeros((GRID_ROWS, GRID_COLS))

    for country in countries:
        lat, lng = country["lat"], country["lng"]
        pop = country["population"]

        # Find grid cells near this country centroid (within ~10°)
        # and distribute population proportionally to weight
        r_center = int(np.clip((GRID_LAT_MAX - lat) / GRID_CELL_SIZE, 0, GRID_ROWS - 1))
        c_center = int(np.clip((lng - GRID_LNG_MIN) / GRID_CELL_SIZE, 0, GRID_COLS - 1))

        # Radius: larger countries get wider distribution
        radius_cells = max(2, int(np.sqrt(pop / 1e7)))
        radius_cells = min(radius_cells, 10)

        local_weight = np.zeros((GRID_ROWS, GRID_COLS))
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                r = r_center + dr
                c = (c_center + dc) % GRID_COLS
                if 0 <= r < GRID_ROWS:
                    dist = np.sqrt(dr**2 + dc**2)
                    if dist <= radius_cells:
                        local_weight[r, c] = weight[r, c] * np.exp(-dist / radius_cells)

        total_w = local_weight.sum()
        if total_w > 0:
            pop_grid += (local_weight / total_w) * pop

    return pop_grid


# ============================================================================
# Resource Grid Estimation
# ============================================================================

def estimate_resource_grids(countries: list[dict]) -> dict:
    """Estimate gridded resources from country data + terrain."""
    # Load terrain if available
    terrain_path = DATA_DIR / "earth_terrain.npy"
    if terrain_path.exists():
        terrain_05 = np.load(terrain_path)
        # Resample 0.5° -> 2°
        offset = int((90.0 - GRID_LAT_MAX) / 0.5)
        terrain = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                r05, c05 = offset + r * 4, c * 4
                if r05 + 4 <= terrain_05.shape[0] and c05 + 4 <= terrain_05.shape[1]:
                    # Mode of 4x4 block
                    block = terrain_05[r05:r05+4, c05:c05+4].flatten()
                    terrain[r, c] = np.bincount(block.astype(int)).argmax()
    else:
        terrain = np.ones((GRID_ROWS, GRID_COLS), dtype=int)

    food = np.zeros((GRID_ROWS, GRID_COLS))
    minerals = np.zeros((GRID_ROWS, GRID_COLS))
    fossil = np.zeros((GRID_ROWS, GRID_COLS))
    freshwater = np.zeros((GRID_ROWS, GRID_COLS))

    # Base from terrain
    food[terrain == 1] = 50  # Plains
    food[terrain == 2] = 30  # Forest
    minerals[terrain == 3] = 60  # Mountains
    minerals[terrain == 4] = 30  # Desert (some minerals)
    freshwater[terrain == 1] = 40
    freshwater[terrain == 2] = 60
    freshwater[terrain == 5] = 30  # Tundra

    # Enhance from country data
    for country in countries:
        r = int(np.clip((GRID_LAT_MAX - country["lat"]) / GRID_CELL_SIZE, 0, GRID_ROWS - 1))
        c = int(np.clip((country["lng"] - GRID_LNG_MIN) / GRID_CELL_SIZE, 0, GRID_COLS - 1))
        radius = 3

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, (c + dc) % GRID_COLS
                if 0 <= rr < GRID_ROWS:
                    d = np.sqrt(dr**2 + dc**2)
                    if d <= radius:
                        w = np.exp(-d / radius)
                        food[rr, cc] += country["arable_land_pct"] * w
                        if country["co2_per_capita"] > 10:  # High fossil use
                            fossil[rr, cc] += 30 * w

    # Known fossil fuel hotspots
    fossil_hotspots = [
        (24, 50, 8, "Persian Gulf"), (60, 70, 6, "W Siberia"),
        (30, -95, 5, "Gulf Mexico"), (55, -115, 4, "Alberta"),
        (5, 5, 4, "Niger Delta"), (58, 5, 3, "North Sea"),
    ]
    for lat, lng, strength, name in fossil_hotspots:
        r = int(np.clip((GRID_LAT_MAX - lat) / GRID_CELL_SIZE, 0, GRID_ROWS - 1))
        c = int(np.clip((lng - GRID_LNG_MIN) / GRID_CELL_SIZE, 0, GRID_COLS - 1))
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr, cc = r + dr, (c + dc) % GRID_COLS
                if 0 <= rr < GRID_ROWS:
                    fossil[rr, cc] += strength * 10 * np.exp(-np.sqrt(dr**2+dc**2) / 2)

    return {
        "food": np.clip(food, 0, 100),
        "minerals": np.clip(minerals, 0, 100),
        "fossil_fuels": np.clip(fossil, 0, 100),
        "freshwater": np.clip(freshwater, 0, 100),
    }


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("PRESENT-DAY DATA PIPELINE")
    print(f"Date: {date.today().isoformat()}")
    print("=" * 70)

    # 1. Climate state
    print("\n[1/5] Fetching climate state (NOAA/NASA)...")
    climate = fetch_climate_state()

    # 2. World Bank indicators
    print("\n[2/5] Fetching World Bank indicators...")
    wb_data = fetch_all_wb_indicators()

    # 3. Build country data
    print("\n[3/5] Building country dataset...")
    countries = build_country_data(wb_data)
    print(f"    Countries with data: {len(countries)}")
    total_pop = sum(c["population"] for c in countries)
    total_gdp = sum(c["gdp_usd"] for c in countries)
    print(f"    Total population: {total_pop / 1e9:.2f} billion")
    print(f"    Total GDP: ${total_gdp / 1e12:.1f} trillion")

    # 4. Population grid
    print("\n[4/5] Building population grid...")
    pop_grid = build_population_grid(countries)
    print(f"    Grid sum: {pop_grid.sum() / 1e9:.2f} billion")
    print(f"    Non-zero cells: {(pop_grid > 0).sum()} / {GRID_ROWS * GRID_COLS}")

    # 5. Resource grids
    print("\n[5/5] Estimating resource grids...")
    resources = estimate_resource_grids(countries)

    # Save everything
    print("\nSaving to data/...")
    np.save(DATA_DIR / "present_day_population.npy", pop_grid)
    print(f"  present_day_population.npy: {pop_grid.shape}")

    np.savez(DATA_DIR / "present_day_resources.npz", **resources)
    print(f"  present_day_resources.npz: food, minerals, fossil_fuels, freshwater")

    with open(DATA_DIR / "present_day_climate.json", "w") as f:
        json.dump(climate, f, indent=2)
    print(f"  present_day_climate.json")

    with open(DATA_DIR / "present_day_countries.json", "w") as f:
        json.dump(countries, f, indent=2)
    print(f"  present_day_countries.json: {len(countries)} countries")

    with open(DATA_DIR / "present_day_conflicts.json", "w") as f:
        json.dump(ACTIVE_CONFLICTS, f, indent=2)
    print(f"  present_day_conflicts.json: {len(ACTIVE_CONFLICTS)} conflicts")

    metadata = {
        "generated": date.today().isoformat(),
        "climate_source": "NOAA GML (CO2) + NASA GISS (temp)",
        "country_source": "World Bank Open Data API",
        "n_countries": len(countries),
        "total_population": int(total_pop),
        "co2_ppm": climate["co2_ppm"],
        "temperature_anomaly": climate["temperature_anomaly"],
    }
    with open(DATA_DIR / "present_day_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    checks = []

    ok = 6e9 < total_pop < 10e9
    checks.append(ok)
    print(f"  Population {total_pop/1e9:.2f}B (target: 6-10B): {'PASS' if ok else 'FAIL'}")

    ok = 420 <= climate["co2_ppm"] <= 435
    checks.append(ok)
    print(f"  CO2 {climate['co2_ppm']} ppm (target: 420-435): {'PASS' if ok else 'FAIL'}")

    ok = 1.0 <= climate["temperature_anomaly"] <= 1.6
    checks.append(ok)
    print(f"  Temp +{climate['temperature_anomaly']}°C (target: 1.0-1.6): {'PASS' if ok else 'FAIL'}")

    ok = len(countries) > 130
    checks.append(ok)
    print(f"  Countries: {len(countries)} (target: >130): {'PASS' if ok else 'FAIL'}")

    print(f"\n  {sum(checks)}/{len(checks)} checks passed")
    print(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
