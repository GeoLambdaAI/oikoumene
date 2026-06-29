"""
Generate a high-resolution rasterized land/ocean mask from Natural Earth
110m land polygons using Shapely.

Output: data/landmask.npy — boolean grid at 0.25° resolution
        (720 rows x 1440 cols covering -90 to +90 lat, -180 to +180 lng)

This only needs to be run once. The resulting .npy file is loaded at
runtime by earth.py for fast O(1) land/ocean lookups.

Data source: Natural Earth (naturalearthdata.com)
             ne_110m_land.geojson — public domain
"""

import json
import numpy as np
from shapely.geometry import shape, Point, MultiPolygon
from shapely.prepared import prep
import time
import os

RESOLUTION = 0.25  # degrees per cell
LAT_MIN, LAT_MAX = -90.0, 90.0
LNG_MIN, LNG_MAX = -180.0, 180.0

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GEOJSON_PATH = os.path.join(DATA_DIR, "ne_110m_land.geojson")
OUTPUT_PATH = os.path.join(DATA_DIR, "landmask.npy")


def load_land_polygons():
    """Load and merge all land polygons from Natural Earth GeoJSON."""
    with open(GEOJSON_PATH, "r") as f:
        data = json.load(f)

    polygons = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        if geom.is_valid:
            polygons.append(geom)
        else:
            polygons.append(geom.buffer(0))  # Fix invalid geometries

    from shapely.ops import unary_union
    merged = unary_union(polygons)
    return merged


def rasterize(land_geometry, resolution=RESOLUTION):
    """
    Rasterize land polygons into a boolean grid.

    Uses Shapely's prepared geometry for fast point-in-polygon tests.
    Grid convention: row 0 = +90° lat (North Pole), last row = -90° (South Pole)
    """
    rows = int((LAT_MAX - LAT_MIN) / resolution)
    cols = int((LNG_MAX - LNG_MIN) / resolution)

    print(f"Rasterizing at {resolution}° resolution: {rows} x {cols} grid")

    # Prepare geometry for fast repeated queries
    prepared = prep(land_geometry)

    mask = np.zeros((rows, cols), dtype=np.bool_)

    t0 = time.time()
    for r in range(rows):
        lat = LAT_MAX - (r + 0.5) * resolution  # Cell center
        for c in range(cols):
            lng = LNG_MIN + (c + 0.5) * resolution
            if prepared.contains(Point(lng, lat)):  # Note: Shapely uses (x=lng, y=lat)
                mask[r, c] = True

        # Progress
        if (r + 1) % 50 == 0:
            elapsed = time.time() - t0
            pct = (r + 1) / rows * 100
            eta = elapsed / (r + 1) * (rows - r - 1)
            print(f"  {pct:.0f}% ({r+1}/{rows} rows) — {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    land_cells = mask.sum()
    total_cells = rows * cols
    land_pct = land_cells / total_cells * 100

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Land cells: {land_cells:,} / {total_cells:,} ({land_pct:.1f}%)")
    print(f"Expected: ~29% land (Earth's actual land fraction)")

    return mask


def main():
    print("Loading Natural Earth 110m land polygons...")
    land = load_land_polygons()
    print(f"  {land.geom_type} with {len(land.geoms) if hasattr(land, 'geoms') else 1} parts")

    mask = rasterize(land)

    print(f"\nSaving to {OUTPUT_PATH}")
    np.save(OUTPUT_PATH, mask)
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

    # Quick validation
    print("\nValidation:")
    def check(name, lat, lng, expected):
        r = int((LAT_MAX - lat) / RESOLUTION)
        c = int((lng - LNG_MIN) / RESOLUTION)
        result = bool(mask[r, c])
        status = "OK" if result == expected else "FAIL"
        print(f"  {name}: ({lat}, {lng}) = {'land' if result else 'ocean'} [{status}]")

    check("Paris", 48.8, 2.3, True)
    check("London", 51.5, -0.1, True)
    check("Tokyo", 35.7, 139.7, True)
    check("New York", 40.7, -74.0, True)
    check("Sydney", -33.9, 151.2, True)
    check("São Paulo", -23.5, -46.6, True)
    check("Atlantic Ocean", 30.0, -40.0, False)
    check("Pacific Ocean", 0.0, -150.0, False)
    check("Indian Ocean", -20.0, 70.0, False)
    check("Mediterranean", 36.0, 18.0, False)
    check("North Sea", 56.0, 3.0, False)


if __name__ == "__main__":
    main()
