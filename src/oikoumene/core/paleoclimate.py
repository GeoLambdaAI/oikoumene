"""
Paleoclimate reconstruction for the historical scenario (moved from history.py):
temperature anomaly, CO2, sea level and ice sheets from ice core data.
"""

import numpy as np


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
