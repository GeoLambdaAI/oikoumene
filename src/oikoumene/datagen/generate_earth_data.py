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

import os
import time

import numpy as np

from oikoumene.datagen.earth_climate import (
    Biome,
    COLS,
    RESOLUTION,
    ROWS,
    classify_biomes,
    generate_elevation,
    generate_precipitation,
    generate_temperature,
    make_lat_lng_grids,
)
from oikoumene.datagen.earth_resources import (
    generate_fertility,
    generate_fossil_fuels,
    generate_freshwater,
    generate_minerals,
)
from oikoumene.paths import DATA_DIR


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
