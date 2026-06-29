#!/usr/bin/env python3
"""
Empirical input generator — downloads authoritative open datasets and
projects them onto the simulator's 0.5° grid.

Datasets (open-licence; see docs/data_attributions.md):
  1. CHELSA V2.1     — high-resolution present-day climatology (CC0 1.0)
                       -> replaces the heuristic earth_temperature.npy /
                          earth_precipitation.npy from generate_earth_data.py.
  2. SoilGrids 2.0   — global soil properties at 250 m (CC-BY 4.0)
                       -> replaces the heuristic earth_fertility.npy.
  3. HYDE 3.2.1      — anthropogenic land-use & population, 10 ka BCE -> 2015 CE
                       (CC0 1.0)
                       -> historical validation track for Scenario A and
                          spatial initial conditions for Scenario B.
  4. HadCRUT.5.1.0.0 — Met Office / CRU global mean surface-temperature
                       anomaly time series 1850-present (Open Government
                       Licence v3.0)
                       -> second independent observational anchor next to
                          NASA GISTEMP; consumed by the macro validation tests.
  5. ETOPO 2022      — NOAA NCEI global relief model at 60 arc-sec
                       (US Government public domain; commercial use allowed)
                       -> replaces the heuristic earth_elevation.npy
                          (currently derived from tectonic-plate boundaries).
  6. UCDP-GED v25.1  — Uppsala Conflict Data Program Georeferenced Event
                       Dataset 1989-present (CC-BY 4.0)
                       -> empirical anchor for the emergent geopolitics layer
                          (replaces the "illustrative" present_day_conflicts.json).
  7. Natural Earth 10m rivers — vector hydrography upgrade (public domain)
                       -> finer river network than the existing 110m vector.

Sources, URL patterns and file IDs were verified against the official
documentation on 2026-05-25 / 2026-06-21:
  - CHELSA tech-spec PDF v1.2 (EnviDat DOI 10.16904/envidat.228);
    bucket listing https://os.unil.cloud.switch.ch/chelsa02/
  - SoilGrids WebDAV index https://files.isric.org/soilgrids/latest/data/
  - HYDE Dataverse files API at https://archaeology.datastations.nl/
  - HadCRUT.5.1.0.0 download index at
    https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/download.html
  - ETOPO 2022 THREDDS server at
    https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/
  - UCDP downloads page at https://ucdp.uu.se/downloads/
  - Natural Earth CDN at https://naciscdn.org/naturalearth/

Output goes under data/empirical/ so it coexists with the synthesized
data/earth_*.npy rasters; world.py chooses which to load.

Optional dependencies: rasterio>=1.3 + pyproj>=3.6 — declared as the
[data] extra. CHELSA and SoilGrids are read via GDAL's /vsicurl/ at the
simulator's coarse resolution so the network footprint stays small
(reads only the strips/tiles needed; no GB-scale local downloads).
HYDE ships as a single 5.3 GB ZIP; the script caches it on disk.

Install: pip install -e ".[data]"

Run once:
    python generate_empirical_inputs.py --all
    python generate_empirical_inputs.py --chelsa --vars tas pr
    python generate_empirical_inputs.py --hyde --years 1500 1700 1900 2000
    python generate_empirical_inputs.py --soilgrids
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests

DATA_DIR = Path(__file__).parent / "data"
EMP_DIR = DATA_DIR / "empirical"
EMP_DIR.mkdir(parents=True, exist_ok=True)

# Simulator target grid — matches generate_earth_data.py defaults.
TARGET_RESOLUTION_DEG = 0.5
TARGET_ROWS = int(180 / TARGET_RESOLUTION_DEG)  # 360
TARGET_COLS = int(360 / TARGET_RESOLUTION_DEG)  # 720


# ============================================================================
# Lazy heavy imports (rasterio + pyproj are an optional [data] extra)
# ============================================================================

def _import_geo():
    try:
        import rasterio
        from rasterio.enums import Resampling
        return rasterio, Resampling
    except ImportError as exc:
        raise ImportError(
            "generate_empirical_inputs.py needs the optional geo stack. "
            'Install with:  pip install -e ".[data]"   (rasterio + pyproj)'
        ) from exc


def _read_resampled(src: str, *, categorical: bool = False) -> np.ndarray:
    """
    Read a global georeferenced raster (local path OR https URL via /vsicurl/)
    and resample to the simulator grid in a single, low-bandwidth call.

    For continuous fields (temperature, precipitation, soil organic carbon)
    we use average resampling. For categorical fields (biome codes etc.) pass
    `categorical=True` to switch to mode resampling.
    """
    rasterio, Resampling = _import_geo()
    method = Resampling.mode if categorical else Resampling.average
    # /vsicurl/ streams the file via HTTP; combined with `out_shape`, rasterio
    # only fetches the strips/tiles needed to produce our coarse-grid array.
    path = f"/vsicurl/{src}" if src.startswith(("http://", "https://")) else src
    with rasterio.open(path) as ds:
        out = ds.read(
            1, out_shape=(TARGET_ROWS, TARGET_COLS), resampling=method
        )
        scale = ds.scales[0] if ds.scales else 1.0
        offset = ds.offsets[0] if ds.offsets else 0.0
    return out.astype("float32") * float(scale) + float(offset)


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Streaming download with resume-on-existence (for files we cannot stream)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  download: {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for blk in r.iter_content(chunk_size=chunk):
                if blk:
                    f.write(blk)
        tmp.rename(dest)
    return dest


# ============================================================================
# 1. CHELSA V2.1 (CC0 1.0)
# ============================================================================
# Filename convention from the CHELSA V2.1 tech specification (Karger &
# Zimmermann, document v1.2, 2021-09-10), Section 5:
#   CHELSA_<short_name>_<timeperiod>_<Version>.tif
# Verified concretely against the S3 bucket listing on 2026-05-25 — e.g.
#   https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/climatologies/tas/
#       1981-2010/CHELSA_tas_01_1981-2010_V.2.1.tif
# Twelve monthly files per variable. GeoTIFF, integer stored, scale+offset
# embedded in the file (read via rasterio.dataset.scales / .offsets).
CHELSA_BASE = (
    "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/climatologies/"
)
CHELSA_PERIOD = "1981-2010"
CHELSA_VERSION = "V.2.1"

# Variables documented in the tech spec Section 7.2 (monthly). Tuple is
# (aggregator, output description) — monthly means are aggregated to an
# annual statistic appropriate for the variable.
CHELSA_VARS = {
    "tas":    ("mean", "annual mean near-surface air temperature (°C)"),
    "tasmin": ("mean", "annual mean daily minimum air temperature (°C)"),
    "tasmax": ("mean", "annual mean daily maximum air temperature (°C)"),
    "pr":     ("sum",  "annual precipitation total (kg m^-2 yr^-1, ~ mm/yr)"),
}


def chelsa_monthly_url(var: str, month: int) -> str:
    return (
        f"{CHELSA_BASE}{var}/{CHELSA_PERIOD}/"
        f"CHELSA_{var}_{month:02d}_{CHELSA_PERIOD}_{CHELSA_VERSION}.tif"
    )


def download_chelsa(variables: Iterable[str] = ("tas", "pr")) -> dict[str, Path]:
    """
    Resample CHELSA V2.1 monthly climatology rasters to the simulator grid
    and aggregate to an annual statistic per variable. Streams via /vsicurl/
    — no GB-scale local downloads.

    For each variable in `variables`, downloads (streams) 12 monthly rasters,
    averages or sums them appropriately, and saves as
    `data/empirical/chelsa_<var>_annual.npy` (float32, in physical units).

    Returns {variable -> output path}. Licence: CC0 1.0 (public domain).
    See docs/data_attributions.md.
    """
    print("[CHELSA V2.1] (CC0 1.0)")
    results: dict[str, Path] = {}

    for var in variables:
        if var not in CHELSA_VARS:
            raise ValueError(
                f"unknown CHELSA variable: {var!r} (known: {sorted(CHELSA_VARS)})"
            )
        aggregator, descr = CHELSA_VARS[var]
        print(f"  variable: {var}  -> {descr}")

        monthly = []
        for month in range(1, 13):
            url = chelsa_monthly_url(var, month)
            print(f"    month {month:02d}: streaming {url.rsplit('/', 1)[1]}")
            monthly.append(_read_resampled(url))

        stack = np.stack(monthly)
        if aggregator == "mean":
            annual = stack.mean(axis=0)
        elif aggregator == "sum":
            annual = stack.sum(axis=0)
        else:
            raise ValueError(f"bad aggregator: {aggregator}")
        annual = annual.astype("float32")

        out = EMP_DIR / f"chelsa_{var}_annual.npy"
        np.save(out, annual)
        finite = annual[np.isfinite(annual)]
        rng = (
            f"[{finite.min():.2f}, {finite.max():.2f}]" if finite.size else "[all-NaN]"
        )
        print(
            f"  -> {out.relative_to(DATA_DIR.parent)}  "
            f"shape={annual.shape} range={rng}"
        )
        results[var] = out

    return results


# ============================================================================
# 2. SoilGrids 2.0 (CC-BY 4.0)
# ============================================================================
# WebDAV directory layout verified on 2026-05-25 at
#   https://files.isric.org/soilgrids/latest/data/
# Per-property layout (top-level dirs: bdod, cec, cfvo, clay, nitrogen, ocd,
# ocs, phh2o, sand, silt, soc, landmask). For each (property, depth, stat)
# triple there is a VRT (virtual raster) at the property root which references
# the tiled COGs below it, e.g.
#   https://files.isric.org/soilgrids/latest/data/soc/soc_0-5cm_mean.vrt
SOILGRIDS_BASE = "https://files.isric.org/soilgrids/latest/data/"


def soilgrids_vrt_url(prop: str, depth: str, stat: str = "mean") -> str:
    return f"{SOILGRIDS_BASE}{prop}/{prop}_{depth}_{stat}.vrt"


def download_soilgrids() -> Path:
    """
    Stream SoilGrids 2.0 root-zone (0-5 cm) layers via /vsicurl/, combine into
    a composite fertility index and save to data/empirical/soilgrids_fertility.npy.

    Composite (illustrative, refine against literature before publication):
       0.50 * soc_norm + 0.30 * nitrogen_norm + 0.20 * clay_norm
    where each layer is min-max-normalised to its plausible top-soil range.

    Returns the output path. Licence: CC-BY 4.0 — attribution required
    (Poggio et al. 2021, SOIL 7(1), 217-240). See docs/data_attributions.md.
    """
    print("[SoilGrids 2.0] (CC-BY 4.0)")
    layers = {
        "soc":      soilgrids_vrt_url("soc",      "0-5cm"),
        "nitrogen": soilgrids_vrt_url("nitrogen", "0-5cm"),
        "clay":     soilgrids_vrt_url("clay",     "0-5cm"),
    }
    rasters: dict[str, np.ndarray] = {}
    for key, url in layers.items():
        print(f"  layer {key}: streaming {url.rsplit('/', 1)[1]}")
        rasters[key] = _read_resampled(url)

    def _norm(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    # Plausible top-soil ranges (verify against ISRIC docs before publication):
    #   soc      ~0..200 dg/kg  (SoilGrids stores soil organic carbon in dg/kg)
    #   nitrogen ~0..500 cg/kg
    #   clay     ~0..500 g/kg
    fertility = (
        0.50 * _norm(rasters["soc"],      0.0, 200.0) +
        0.30 * _norm(rasters["nitrogen"], 0.0, 500.0) +
        0.20 * _norm(rasters["clay"],     0.0, 500.0)
    ).astype("float32")

    out = EMP_DIR / "soilgrids_fertility.npy"
    np.save(out, fertility)
    finite = fertility[np.isfinite(fertility)]
    rng = (
        f"[{finite.min():.3f}, {finite.max():.3f}]" if finite.size else "[all-NaN]"
    )
    print(
        f"  -> {out.relative_to(DATA_DIR.parent)}  "
        f"shape={fertility.shape} range={rng}"
    )
    return out


# ============================================================================
# 3. HYDE 3.2.1 (CC0 1.0)
# ============================================================================
# DANS Dataverse access pattern (standard across all Dataverse instances):
#   https://archaeology.datastations.nl/api/access/datafile/<fileId>
# File IDs and sizes verified via the Dataverse JSON files API on 2026-05-25:
#   id=5490328  HYDE3_2_1-baseline.zip            (5.3 GB)
#   id=5490329  HYDE3_2_1-anthromes.zip           (141.8 MB)
#   id=5490327  HYDE3_2_1-general_supplementary.zip ( 23.6 MB)
#   id=5396388  readme_release_HYDE3.2.1.txt
#   id=5398615  easy-migration.zip
# Inside HYDE3_2_1-baseline.zip the per-year ASC grids follow the convention
# documented in readme_release_HYDE3.2.1.txt:
#   popc_<year>AD.asc  (AD)   /  popc_<year>BC.asc  (paleo)
#   cropland_<year>AD.asc, grazing_<year>AD.asc, ...
HYDE_API = "https://archaeology.datastations.nl/api/access/datafile/{file_id}"
HYDE_FILES = {
    "baseline_zip":     5490328,    # main file, ~5.3 GB
    "anthromes_zip":    5490329,
    "supplementary":    5490327,
    "readme_txt":       5396388,
    "easy_migration":   5398615,
}


def _hyde_member_name(prefix: str, year: int) -> str:
    """ASC filename convention for HYDE time-slice rasters (see readme)."""
    if year >= 1:
        return f"{prefix}_{year}AD.asc"
    # Paleo years in HYDE 3.2.1 use BC labels for negative simulation years;
    # the convention treats year 0 as 1 BC (no year 0). Adjust here so callers
    # can pass a calendar year like -10000 for 10 ka BCE.
    bc_label = abs(year) if year < 0 else 1
    return f"{prefix}_{bc_label}BC.asc"


def download_hyde(
    years: Iterable[int] = (1500, 1700, 1900, 2000),
    layers: Iterable[str] = ("popc", "cropland", "grazing"),
) -> dict[str, dict[int, Path]]:
    """
    Download the HYDE 3.2.1 baseline ZIP (5.3 GB, cached locally) and extract
    the requested layer × year ASC grids onto the simulator grid.

    For each (layer, year) pair, the corresponding ASC raster is extracted
    in-memory, resampled to the simulator grid via /vsicurl/-equivalent
    average resampling, and saved as
    `data/empirical/hyde_<layer>/<year>.npy` (float32).

    Returns {layer -> {year -> path}}. Licence: CC0 1.0 (public domain).
    See docs/data_attributions.md.
    """
    print("[HYDE 3.2.1] (CC0 1.0)")
    hyde_raw = EMP_DIR / "hyde_raw"
    hyde_raw.mkdir(exist_ok=True)
    bundle = hyde_raw / "HYDE3_2_1-baseline.zip"
    _stream_download(HYDE_API.format(file_id=HYDE_FILES["baseline_zip"]), bundle)

    results: dict[str, dict[int, Path]] = {layer: {} for layer in layers}
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        for layer in layers:
            out_dir = EMP_DIR / f"hyde_{layer}"
            out_dir.mkdir(exist_ok=True)
            for year in years:
                target = _hyde_member_name(layer, year)
                members = [m for m in names if m.endswith(target)]
                if not members:
                    print(f"  skip {layer} {year}: {target} not found in bundle")
                    continue
                member = members[0]
                tmp = hyde_raw / target
                with zf.open(member) as src, open(tmp, "wb") as dst:
                    dst.write(src.read())
                grid = _read_resampled(str(tmp))
                out = out_dir / f"{year}.npy"
                np.save(out, grid.astype("float32"))
                results[layer][year] = out
                finite = grid[np.isfinite(grid)]
                rng = (
                    f"[{finite.min():.2f}, {finite.max():.2f}]"
                    if finite.size
                    else "[all-NaN]"
                )
                print(
                    f"  -> {out.relative_to(DATA_DIR.parent)}  "
                    f"shape={grid.shape} range={rng}"
                )
    return results


# ============================================================================
# 4. HadCRUT.5.1.0.0 (Open Government Licence v3.0)
# ============================================================================
# Met Office Hadley Centre + CRU global temperature anomaly time series.
# Base path and CSV column layout verified on 2026-05-25 against the
# canonical download index at
#   https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/download.html
# Column header (verified against the live CSV):
#   Time,Anomaly (deg C),Ensemble standard deviation (1 sigma),
#   Coverage uncertainty (1 sigma),Total uncertainty (1 sigma)
# Anomalies are relative to the 1961-1990 baseline (Met Office standard).
# Note: GISTEMP uses a 1951-1980 baseline, so direct comparison should
# account for the ~0.10-0.15 deg C baseline offset between the two.
HADCRUT5_VERSION = "HadCRUT.5.1.0.0"
HADCRUT5_BASE = (
    "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/"
    f"{HADCRUT5_VERSION}/analysis/diagnostics/"
)
HADCRUT5_ANNUAL_URL = (
    f"{HADCRUT5_BASE}{HADCRUT5_VERSION}.analysis.component_series.global.annual.csv"
)
HADCRUT5_ANNUAL_NPY = EMP_DIR / "hadcrut5_global_annual.npy"


def _drop_incomplete_trailing_years(arr: np.ndarray) -> np.ndarray:
    """Drop trailing partial-year rows from the HadCRUT5 annual series.

    HadCRUT5 publishes a provisional value for the in-progress calendar year.
    Because only part of the year is observed, that row's coverage-uncertainty
    (``sigma_cov``) is sharply inflated relative to complete years. Including it
    as if it were a finished annual mean is misleading (it can even read as a
    spurious cooling), so we drop any trailing rows whose ``sigma_cov`` exceeds
    5x the *recent* median coverage-uncertainty — leaving the record ending at
    the last *complete* annual mean. The reference uses the last ~30 years
    (modern coverage is dense and uniform, ~0.004; the median is robust to the
    single inflated tail year). A full-series median would be biased high by the
    sparse 19th-century record, so it is deliberately not used.
    """
    if len(arr) < 30:
        return arr
    ref = float(np.median(arr["sigma_cov"][-31:]))
    cut = len(arr)
    while cut > 0 and arr["sigma_cov"][cut - 1] > 5.0 * ref:
        cut -= 1
    if cut < len(arr):
        dropped = [int(y) for y in arr["year"][cut:]]
        print(f"  dropped incomplete trailing year(s): {dropped} "
              f"(coverage uncertainty >> series median {ref:.4f})")
    return arr[:cut]


def download_hadcrut5() -> Path:
    """
    Download the HadCRUT.5.1.0.0 global mean annual temperature anomaly time
    series (1850 -> latest year) and save it as a structured NumPy array at
    `data/empirical/hadcrut5_global_annual.npy` with named fields:

        year   : int       (calendar year)
        anom   : float32   (deg C, anomaly vs 1961-1990 baseline)
        sigma_ens, sigma_cov, sigma_total : float32  (1-sigma uncertainties)

    Returns the output path. Licence: Open Government Licence v3.0 — free,
    commercial use OK, attribution required (Met Office Hadley Centre + CRU).
    See docs/data_attributions.md.
    """
    print(f"[{HADCRUT5_VERSION}] (Open Government Licence v3.0)")
    hc_raw = EMP_DIR / "hadcrut5_raw"
    hc_raw.mkdir(exist_ok=True)
    csv = hc_raw / f"{HADCRUT5_VERSION}.analysis.component_series.global.annual.csv"
    _stream_download(HADCRUT5_ANNUAL_URL, csv)

    # Parse the CSV explicitly (small file, header-known columns — verified
    # against the live file on 2026-05-25). Avoid pandas to keep the lone-
    # numpy invariant of the runtime dependency set.
    with open(csv) as f:
        header = f.readline().strip().split(",")
        expected_first = ["Time", "Anomaly (deg C)"]
        if header[:2] != expected_first:
            raise RuntimeError(
                f"unexpected HadCRUT5 CSV header: {header[:2]} (expected {expected_first}). "
                "The file specification at "
                f"{HADCRUT5_BASE.replace('/diagnostics/', '/')} "
                "may have changed; verify and update the parser."
            )
        rows = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            rows.append(
                (int(parts[0]), float(parts[1]),
                 float(parts[2]), float(parts[3]), float(parts[4]))
            )

    dtype = np.dtype([
        ("year",        "i4"),
        ("anom",        "f4"),
        ("sigma_ens",   "f4"),
        ("sigma_cov",   "f4"),
        ("sigma_total", "f4"),
    ])
    arr = np.array(rows, dtype=dtype)
    arr = _drop_incomplete_trailing_years(arr)
    np.save(HADCRUT5_ANNUAL_NPY, arr)
    print(
        f"  -> {HADCRUT5_ANNUAL_NPY.relative_to(DATA_DIR.parent)}  "
        f"years={arr['year'][0]}-{arr['year'][-1]}  n={len(arr)}  "
        f"recent={arr['anom'][-3:].mean():.3f} deg C (avg of last 3 yrs)"
    )
    return HADCRUT5_ANNUAL_NPY


def load_hadcrut5_global_annual() -> Optional[np.ndarray]:
    """
    Return the saved HadCRUT5 global annual anomaly array, or None if the
    user hasn't run `generate_empirical_inputs.py --hadcrut5` yet.

    Returned array is a structured numpy array with fields
    (year, anom, sigma_ens, sigma_cov, sigma_total). Anomalies are in
    deg C relative to the 1961-1990 baseline.
    """
    if not HADCRUT5_ANNUAL_NPY.exists():
        return None
    return np.load(HADCRUT5_ANNUAL_NPY)


# ============================================================================
# 5. ETOPO 2022 (US Government public domain)
# ============================================================================
# 60-arc-second global relief (single global NetCDF file, ~395 MB).
# URL and licence verified on 2026-06-21:
#   THREDDS server returned HTTP 200 for the file; ETOPO 2022 User Guide
#   §4 (https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/docs/) states:
#   "ETOPO tiles are freely available to use for all private, academic, or
#    commercial purposes."
# Citation (verified in the User Guide):
#   NOAA NCEI. 2022. ETOPO 2022 15 Arc-Second Global Relief Model.
#   DOI: 10.25921/fd45-gt74.
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/"
    "60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)
ETOPO_NPY = EMP_DIR / "etopo_elevation.npy"


def download_etopo() -> Path:
    """
    Download the ETOPO 2022 60-arc-sec global relief (NetCDF, ~395 MB,
    cached locally), resample to the simulator grid and save as
    `data/empirical/etopo_elevation.npy` (float32, metres above EGM2008
    geoid; ocean values are negative).

    Returns the output path. Licence: US Government public domain.
    Citation required (NOAA NCEI 2022). See docs/data_attributions.md.
    """
    print("[ETOPO 2022 60s] (US Government public domain)")
    etopo_raw = EMP_DIR / "etopo_raw"
    etopo_raw.mkdir(exist_ok=True)
    nc = etopo_raw / "ETOPO_2022_v1_60s_N90W180_surface.nc"
    _stream_download(ETOPO_URL, nc)

    rasterio, Resampling = _import_geo()
    # ETOPO 2022 ships as NetCDF (HDF5). Open it via rasterio's NetCDF driver
    # and, if the file exposes subdatasets, address the elevation variable
    # directly rather than reading the file container.
    with rasterio.open(str(nc)) as probe:
        targets = list(probe.subdatasets) or [str(nc)]
    elev_target = next(
        (t for t in targets if any(k in t.lower() for k in (":z", "elev", ":band1"))),
        targets[0],
    )
    with rasterio.open(elev_target) as ds:
        elev = ds.read(
            1, out_shape=(TARGET_ROWS, TARGET_COLS), resampling=Resampling.average
        ).astype("float32")
        scale = ds.scales[0] if ds.scales else 1.0
        offset = ds.offsets[0] if ds.offsets else 0.0
    elev = elev * float(scale) + float(offset)

    np.save(ETOPO_NPY, elev)
    print(
        f"  -> {ETOPO_NPY.relative_to(DATA_DIR.parent)}  "
        f"shape={elev.shape} min={elev.min():.0f} max={elev.max():.0f} "
        f"mean={elev.mean():.0f} (m)"
    )
    return ETOPO_NPY


def load_etopo_elevation() -> Optional[np.ndarray]:
    """Return the saved ETOPO elevation grid (float32, metres) or None."""
    if not ETOPO_NPY.exists():
        return None
    return np.load(ETOPO_NPY)


# ============================================================================
# 6. UCDP-GED v25.1 (CC-BY 4.0)
# ============================================================================
# Uppsala Conflict Data Program — Georeferenced Event Dataset.
# Licence text verified on 2026-06-21 on https://ucdp.uu.se/downloads/ :
#   "All datasets are free of charge and licensed under CC BY 4.0".
# The ZIP at https://ucdp.uu.se/downloads/ged/ged251-csv.zip (29.3 MB)
# contains a single file GEDEvent_v25_1.csv.
# Citation: Davies, Pettersson & Öberg 2026 (annual update),
#           Sundberg & Melander 2013 (original GED methodology).
UCDP_GED_URL = "https://ucdp.uu.se/downloads/ged/ged251-csv.zip"
UCDP_GED_INNER_CSV = "GEDEvent_v25_1.csv"
UCDP_GED_NPY = EMP_DIR / "ucdp_ged_events.npy"


def download_ucdp_ged() -> Path:
    """
    Download UCDP-GED v25.1, extract the event CSV and save the columns
    relevant to the simulator (year, location, deaths) as a structured
    NumPy array at `data/empirical/ucdp_ged_events.npy` with fields
    `(year, latitude, longitude, best, conflict_id)`.

    Returns the output path. Licence: CC-BY 4.0 — attribution required.
    See docs/data_attributions.md.
    """
    import csv
    import zipfile

    print("[UCDP-GED v25.1] (CC-BY 4.0)")
    ged_raw = EMP_DIR / "ucdp_ged_raw"
    ged_raw.mkdir(exist_ok=True)
    bundle = ged_raw / "ged251-csv.zip"
    _stream_download(UCDP_GED_URL, bundle)

    # Extract just the events CSV — the ZIP holds nothing else of interest.
    with zipfile.ZipFile(bundle) as zf:
        members = [m for m in zf.namelist() if m.endswith(UCDP_GED_INNER_CSV)]
        if not members:
            raise RuntimeError(
                f"{UCDP_GED_INNER_CSV} not found in {bundle.name}; "
                "the UCDP-GED release schema may have changed — verify "
                f"https://ucdp.uu.se/downloads/ and update UCDP_GED_INNER_CSV."
            )
        csv_path = ged_raw / UCDP_GED_INNER_CSV
        with zf.open(members[0]) as src, open(csv_path, "wb") as dst:
            dst.write(src.read())

    # Parse only the columns we need (the full CSV has 49+ columns).
    needed = ["year", "latitude", "longitude", "best", "conflict_new_id"]
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in needed if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"UCDP-GED CSV missing expected columns: {missing}. "
                f"Available: {reader.fieldnames[:10]}…  "
                "The codebook at https://ucdp.uu.se/downloads/ may have changed."
            )
        for r in reader:
            try:
                rows.append((
                    int(r["year"]),
                    float(r["latitude"]),
                    float(r["longitude"]),
                    int(r["best"]),
                    int(r["conflict_new_id"]),
                ))
            except (ValueError, KeyError):
                continue  # incomplete row — skip silently (UCDP convention)

    dtype = np.dtype([
        ("year",        "i4"),
        ("lat",         "f4"),
        ("lng",         "f4"),
        ("best",        "i4"),
        ("conflict_id", "i4"),
    ])
    arr = np.array(rows, dtype=dtype)
    np.save(UCDP_GED_NPY, arr)
    print(
        f"  -> {UCDP_GED_NPY.relative_to(DATA_DIR.parent)}  "
        f"events={len(arr):,}  years={arr['year'].min()}-{arr['year'].max()}  "
        f"total_best_deaths={int(arr['best'].sum()):,}"
    )
    return UCDP_GED_NPY


def load_ucdp_ged_events() -> Optional[np.ndarray]:
    """Return the saved UCDP-GED structured event array or None."""
    if not UCDP_GED_NPY.exists():
        return None
    return np.load(UCDP_GED_NPY)


# ============================================================================
# 7. Natural Earth 10m rivers (public domain)
# ============================================================================
# Drop-in resolution upgrade over the shipped 110m vector
# (data/ne_110m_rivers.geojson). Licence per Natural Earth:
#   "Free of charge for any purpose, including commercial. No permission is
#    needed to use Natural Earth."  (verified previously in data_attributions.md).
# CDN host returned HTTP 200 for the ZIP on 2026-06-21 (2.0 MB).
NE_10M_RIVERS_URL = (
    "https://naciscdn.org/naturalearth/10m/physical/"
    "ne_10m_rivers_lake_centerlines.zip"
)
NE_10M_RIVERS_GEOJSON = EMP_DIR / "ne_10m_rivers_lake_centerlines.geojson"


def download_ne_10m_rivers() -> Path:
    """
    Download Natural Earth 10m rivers + lake centrelines, convert the
    shapefile to GeoJSON, and save to
    `data/empirical/ne_10m_rivers_lake_centerlines.geojson`. Drop-in
    upgrade over the shipped 110m vector. Public domain.

    Requires `pyshp` (in the `[data]` extra) for the shapefile parser.
    """
    import json
    import zipfile

    try:
        import shapefile  # pyshp
    except ImportError as exc:
        raise ImportError(
            "Natural Earth shapefile -> GeoJSON conversion needs `pyshp`. "
            'Install with:  pip install -e ".[data]"'
        ) from exc

    print("[Natural Earth 10m rivers] (public domain)")
    ne_raw = EMP_DIR / "ne_rivers_raw"
    ne_raw.mkdir(exist_ok=True)
    bundle = ne_raw / "ne_10m_rivers_lake_centerlines.zip"
    _stream_download(NE_10M_RIVERS_URL, bundle)

    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(ne_raw)

    shp = ne_raw / "ne_10m_rivers_lake_centerlines.shp"
    if not shp.exists():
        raise RuntimeError(
            f"expected {shp.name} after unzipping {bundle.name}; "
            "the Natural Earth archive structure may have changed."
        )

    with shapefile.Reader(str(shp)) as sr:
        field_names = [f[0] for f in sr.fields[1:]]  # drop deletion flag
        features = []
        for shape_rec in sr.iterShapeRecords():
            features.append({
                "type": "Feature",
                "geometry": shape_rec.shape.__geo_interface__,
                "properties": dict(zip(field_names, shape_rec.record)),
            })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(NE_10M_RIVERS_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    print(
        f"  -> {NE_10M_RIVERS_GEOJSON.relative_to(DATA_DIR.parent)}  "
        f"features={len(features):,}  "
        f"(110m baseline is ~{_count_features_safe(DATA_DIR / 'ne_110m_rivers.geojson')})"
    )
    return NE_10M_RIVERS_GEOJSON


def _count_features_safe(path: Path) -> str:
    """Best-effort feature count for the shipped 110m GeoJSON (for comparison)."""
    try:
        import json
        with open(path) as f:
            data = json.load(f)
        return f"{len(data.get('features', []))}"
    except Exception:
        return "?"


# ============================================================================
# CLI
# ============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--all", action="store_true", help="run every downloader")
    p.add_argument("--chelsa", action="store_true")
    p.add_argument("--soilgrids", action="store_true")
    p.add_argument("--hyde", action="store_true")
    p.add_argument("--hadcrut5", action="store_true",
                   help="download HadCRUT.5.1.0.0 global annual anomaly time series")
    p.add_argument("--etopo", action="store_true",
                   help="download ETOPO 2022 60-arc-sec global relief (~395 MB once)")
    p.add_argument("--ucdp-ged", action="store_true", dest="ucdp_ged",
                   help="download UCDP-GED v25.1 conflict events (CC-BY 4.0)")
    p.add_argument("--ne-rivers", action="store_true", dest="ne_rivers",
                   help="download Natural Earth 10m rivers and convert to GeoJSON")
    p.add_argument(
        "--vars", nargs="+", default=("tas", "pr"),
        help=f"CHELSA variables (known: {sorted(CHELSA_VARS)}). default: tas pr",
    )
    p.add_argument(
        "--years", nargs="+", type=int,
        default=(1500, 1700, 1900, 2000),
        help="HYDE years (negative for paleo, e.g. -10000 for 10 ka BCE)",
    )
    p.add_argument(
        "--hyde-layers", nargs="+", default=("popc", "cropland", "grazing"),
        help="HYDE layers (popc=population, cropland, grazing). default: all three",
    )
    args = p.parse_args(argv)

    if not any([args.all, args.chelsa, args.soilgrids, args.hyde,
                args.hadcrut5, args.etopo, args.ucdp_ged, args.ne_rivers]):
        p.print_help()
        return 1

    if args.all or args.chelsa:
        download_chelsa(args.vars)
    if args.all or args.soilgrids:
        download_soilgrids()
    if args.all or args.hyde:
        download_hyde(args.years, args.hyde_layers)
    if args.all or args.hadcrut5:
        download_hadcrut5()
    if args.all or args.etopo:
        download_etopo()
    if args.all or args.ucdp_ged:
        download_ucdp_ged()
    if args.all or args.ne_rivers:
        download_ne_10m_rivers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
