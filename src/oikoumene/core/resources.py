"""
Grid-based resource map over Earth (moved from world.py): ResourceMap.
"""

import numpy as np
from typing import Optional

from oikoumene.core.earth import TerrainType


# ============================================================================
# Resource System (Earth Grid)
# ============================================================================

class ResourceMap:
    """Grid-based resource map over Earth with regeneration."""

    def __init__(self, earth_grid: dict):
        self.rows = earth_grid["rows"]
        self.cols = earth_grid["cols"]
        self.lats = earth_grid["lats"]
        self.lngs = earth_grid["lngs"]
        self.cell_size_deg = earth_grid["cell_size_deg"]
        self.lat_min = earth_grid["lat_min"]
        self.lat_max = earth_grid["lat_max"]
        self.lng_min = earth_grid["lng_min"]
        self.lng_max = earth_grid["lng_max"]

        # Resource layers
        self.food = np.zeros((self.rows, self.cols))
        self.minerals = np.zeros((self.rows, self.cols))
        self.wood = np.zeros((self.rows, self.cols))
        self.water = np.zeros((self.rows, self.cols))

        # Regeneration rates
        self.food_regen = np.zeros((self.rows, self.cols))
        self.minerals_regen = np.zeros((self.rows, self.cols))
        self.wood_regen = np.zeros((self.rows, self.cols))
        self.water_regen = np.zeros((self.rows, self.cols))

        # Per-cell baselines used by World._apply_ice_age_effects to apply
        # set-from-baseline (idempotent) semantics for paleoclimate scaling,
        # rather than the multiplicative ratchet of the previous code which
        # drove cold-region food_regen to underflow over Pleistocene-scale
        # runs. Lazily snapshotted on the first paleo tick (see _apply_ice_age_effects).
        self._baseline_food: Optional[np.ndarray] = None
        self._baseline_food_regen: Optional[np.ndarray] = None
        self._baseline_wood: Optional[np.ndarray] = None
        self._baseline_wood_regen: Optional[np.ndarray] = None
        self._baseline_water: Optional[np.ndarray] = None
        # Per-cell flag tracking whether the cell has ever been ice-covered
        # in this simulation. Set when ice_mask reports ice; cleared on the
        # iced->non-iced transition so post-glacial recovery seeds fire once.
        self._was_iced: Optional[np.ndarray] = None

        # Persistent multiplicative drought factors (god-mode). These compose
        # with whichever subsystem is the authoritative regen writer for the era
        # (paleo ice-age or modern macro bridge), so a drought is not silently
        # erased when that subsystem rewrites food_regen/water_regen. 1.0 = no
        # drought; a value < 1 reduces regen. Maintained by god_mode drought
        # apply/revert and read back in _apply_ice_age_effects and
        # bridge.apply_macro_to_world.
        self.drought_food_factor = np.ones((self.rows, self.cols))
        self.drought_water_factor = np.ones((self.rows, self.cols))

    def initialize_from_terrain(self, terrain: np.ndarray, fertility: np.ndarray,
                               minerals_grid=None, freshwater_grid=None,
                               fossil_grid=None):
        """
        Set initial resources based on terrain, fertility, and Earth system data.

        Uses pre-computed mineral, freshwater, and fossil fuel grids when available.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                t = terrain[r, c]
                f = fertility[r, c]

                # Mineral richness from Earth system data or terrain default
                m = minerals_grid[r, c] if minerals_grid is not None else (0.5 if t == 3 else 0.2)
                # Freshwater from Earth system data or terrain default
                w = freshwater_grid[r, c] if freshwater_grid is not None else 0.5

                if t == TerrainType.PLAINS:
                    self.food[r, c] = 80 * f
                    self.food_regen[r, c] = 2.0 * f
                    self.water[r, c] = 60 * w
                    self.water_regen[r, c] = 1.0 * w
                    self.minerals[r, c] = 30 * m
                    self.minerals_regen[r, c] = 0.2 * m
                elif t == TerrainType.FOREST:
                    self.food[r, c] = 40 * f
                    self.food_regen[r, c] = 1.0 * f
                    self.wood[r, c] = 100 * f
                    self.wood_regen[r, c] = 1.5 * f
                    self.water[r, c] = 70 * w
                    self.water_regen[r, c] = 1.5 * w
                    self.minerals[r, c] = 20 * m
                elif t == TerrainType.MOUNTAINS:
                    self.minerals[r, c] = 100 * m
                    self.minerals_regen[r, c] = 0.5 * m
                    self.water[r, c] = 40 * w
                    self.water_regen[r, c] = 0.5 * w
                    self.food[r, c] = 10 * f
                    self.food_regen[r, c] = 0.2 * f
                elif t == TerrainType.DESERT:
                    self.minerals[r, c] = 50 * m
                    self.minerals_regen[r, c] = 0.3 * m
                    self.water[r, c] = 5 * w
                    self.water_regen[r, c] = 0.1 * w
                    self.food[r, c] = 5 * f
                    self.food_regen[r, c] = 0.1 * f
                elif t == TerrainType.TUNDRA:
                    self.food[r, c] = 10 * f
                    self.food_regen[r, c] = 0.3 * f
                    self.water[r, c] = 50 * w
                    self.water_regen[r, c] = 0.5 * w
                    self.minerals[r, c] = 40 * m
                    self.minerals_regen[r, c] = 0.3 * m

        # Capture pristine per-cell baselines NOW, from the freshly initialized
        # terrain x fertility values — before any agent harvest or macro rewrite
        # touches these arrays. (Previously these were snapshotted lazily on the
        # first ice-age tick, i.e. ~50 ticks in, capturing already-depleted and
        # macro-modified state.)
        self._baseline_food = self.food.copy()
        self._baseline_food_regen = self.food_regen.copy()
        self._baseline_wood = self.wood.copy()
        self._baseline_wood_regen = self.wood_regen.copy()
        self._baseline_water = self.water.copy()
        self._was_iced = np.zeros((self.rows, self.cols), dtype=bool)

    def get_cell(self, lat: float, lng: float) -> tuple[int, int]:
        """Convert lat/lng to grid row/col."""
        r = int(np.clip((self.lat_max - lat) / self.cell_size_deg,
                        0, self.rows - 1))
        c = int(np.clip((lng - self.lng_min) / self.cell_size_deg,
                        0, self.cols - 1))
        return r, c

    def harvest(self, lat: float, lng: float, resource_type: str, amount: float) -> float:
        r, c = self.get_cell(lat, lng)
        layer = getattr(self, resource_type, None)
        if layer is None:
            return 0.0
        available = layer[r, c]
        taken = min(available, amount)
        layer[r, c] -= taken
        return float(taken)

    def get_local(self, lat: float, lng: float) -> dict:
        r, c = self.get_cell(lat, lng)
        return {
            "food": float(self.food[r, c]),
            "minerals": float(self.minerals[r, c]),
            "wood": float(self.wood[r, c]),
            "water": float(self.water[r, c]),
        }

    def regenerate(self):
        """Regenerate resources each tick."""
        self.food = np.minimum(self.food + self.food_regen * 0.1, 100.0)
        self.minerals = np.minimum(self.minerals + self.minerals_regen * 0.05, 100.0)
        self.wood = np.minimum(self.wood + self.wood_regen * 0.08, 100.0)
        self.water = np.minimum(self.water + self.water_regen * 0.1, 100.0)

    def to_grid_data(self) -> dict:
        """Serialize for UI (only non-zero cells to save bandwidth)."""
        return {
            "food": self.food.tolist(),
            "minerals": self.minerals.tolist(),
            "wood": self.wood.tolist(),
            "water": self.water.tolist(),
            "cell_size_deg": self.cell_size_deg,
            "cols": self.cols,
            "rows": self.rows,
        }
