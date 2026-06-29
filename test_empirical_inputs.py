"""
Validation tests for the optional empirical input downloaders.

Each test SKIPS cleanly when the corresponding dataset has not been fetched
(so CI runs without forcing GB-scale network downloads), and asserts strong
realistic properties when the data IS present — these are real validation
contracts on the downloader, not smoke tests.

Run the relevant downloader once before running the corresponding test:

    python generate_empirical_inputs.py --etopo
    python generate_empirical_inputs.py --ucdp-ged
    python generate_empirical_inputs.py --ne-rivers

See docs/data_attributions.md for source URLs, licences and citations.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")
from generate_empirical_inputs import (
    NE_10M_RIVERS_GEOJSON,
    load_etopo_elevation,
    load_ucdp_ged_events,
)

DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# ETOPO 2022 — elevation realism
# ============================================================================

def test_etopo_elevation_loads_and_is_realistic():
    """
    ETOPO 2022 60-arc-sec global relief, resampled to the 0.5 deg simulator
    grid, should yield an elevation field consistent with real Earth physical
    geography: ocean-dominated mean, very high mountain cells, very deep
    ocean trench cells, no NaNs, full coverage of the 360 x 720 grid.
    """
    elev = load_etopo_elevation()
    if elev is None:
        pytest.skip(
            "ETOPO not fetched yet — run "
            "`python generate_empirical_inputs.py --etopo` "
            "(~395 MB one-time download)"
        )

    print("\n" + "=" * 70)
    print("VALIDATION: ETOPO 2022 elevation realism")
    print("=" * 70)

    # Structural sanity — must match the simulator grid the rest of the
    # code expects (generate_earth_data.py: 360 rows x 720 cols at 0.5 deg).
    assert elev.shape == (360, 720), f"unexpected shape {elev.shape}"
    assert elev.dtype == np.float32, f"unexpected dtype {elev.dtype}"
    assert np.all(np.isfinite(elev)), "ETOPO grid contains NaN/Inf cells"

    print(f"  shape: {elev.shape}  dtype: {elev.dtype}")
    print(f"  min:   {elev.min():>8.0f} m  (deepest cell — should be a trench)")
    print(f"  max:   {elev.max():>8.0f} m  (highest cell — Himalayan/Andean)")
    print(f"  mean:  {elev.mean():>8.0f} m  (should be negative — oceans dominate)")

    # Physical-geography anchors. At 0.5 deg resolution individual peaks are
    # spatially averaged so we are conservative against the well-known
    # extreme values (Everest 8,848 m, Challenger Deep -10,984 m).
    assert elev.max() > 5000, (
        f"max elevation {elev.max():.0f} m is implausibly low — "
        "no cell exceeds 5 km, so either Asia/South America were not "
        "sampled or the GeoTIFF scale/offset was misapplied."
    )
    assert elev.min() < -5000, (
        f"min elevation {elev.min():.0f} m is too shallow — "
        "no cell deeper than -5 km, so the Pacific / Atlantic trenches "
        "are missing or the file was clipped to land."
    )
    assert elev.mean() < 0, (
        f"mean elevation {elev.mean():.0f} m is positive — "
        "oceans should dominate (~71 % of Earth's surface), so the "
        "resampling may have masked the ocean cells out."
    )

    # Land fraction — should match Earth's ~29 % land coverage to within a
    # few percent (some shallow shelf cells average just above zero).
    land_frac = float(np.mean(elev > 0))
    print(f"  land fraction (elev > 0): {land_frac:.3f}  "
          "(Earth's actual ~0.29)")
    assert 0.20 < land_frac < 0.45, (
        f"land fraction {land_frac:.3f} outside [0.20, 0.45] — "
        "ETOPO resampling produced a clearly unrealistic land/sea split."
    )
    print("  PASS: ETOPO elevation field matches Earth physical geography.")


# ============================================================================
# UCDP-GED — empirical anchor for emergent geopolitics
# ============================================================================

def test_ucdp_ged_events_load_and_have_realistic_spread():
    """
    UCDP-GED v25.1 is the canonical 1989-present global conflict event
    dataset. As an empirical anchor for the simulator's emergent geopolitics
    layer it must (a) cover the full 1989-recent window, (b) hold a
    substantial event count, and (c) be globally distributed rather than
    clustered in a single region.
    """
    events = load_ucdp_ged_events()
    if events is None:
        pytest.skip(
            "UCDP-GED not fetched yet — run "
            "`python generate_empirical_inputs.py --ucdp-ged`"
        )

    print("\n" + "=" * 70)
    print("VALIDATION: UCDP-GED v25.1 empirical conflict anchor")
    print("=" * 70)

    # Structural sanity
    assert events.dtype.names == ("year", "lat", "lng", "best", "conflict_id"), (
        f"unexpected column set {events.dtype.names}"
    )
    n = len(events)
    print(f"  events: {n:,}")
    assert n > 100_000, (
        f"only {n:,} events — UCDP-GED v25.1 is a substantial dataset "
        "(>100,000 events); the downloader saved an incomplete subset."
    )

    # Temporal coverage — UCDP-GED's start year is 1989 by definition; the
    # latest year must be recent (>= 2020) since the dataset is updated
    # annually.
    y_min, y_max = int(events["year"].min()), int(events["year"].max())
    print(f"  year range: {y_min} – {y_max}")
    assert y_min == 1989, f"UCDP-GED should start in 1989, got {y_min}"
    assert y_max >= 2020, (
        f"latest year {y_max} < 2020 — release schema may have shifted "
        "or the bundle was truncated."
    )

    # Geographic spread — events must cover a realistic globe-spanning lat
    # range. Conflict in 1989-present has touched every continent, so the
    # latitude std should be substantial.
    lat_std = float(events["lat"].std())
    lng_std = float(events["lng"].std())
    print(f"  lat std: {lat_std:.2f} deg   lng std: {lng_std:.2f} deg")
    assert lat_std > 10, (
        f"latitude std {lat_std:.2f} too small — events appear clustered "
        "in one region; the CSV parser may be reading the wrong column."
    )
    assert lng_std > 30, (
        f"longitude std {lng_std:.2f} too small — events appear clustered "
        "in one region; the CSV parser may be reading the wrong column."
    )

    # Coordinates must be within plausible ranges (UCDP encodes lat in
    # [-90, 90] and lng in [-180, 180]).
    assert events["lat"].min() >= -90 and events["lat"].max() <= 90
    assert events["lng"].min() >= -180 and events["lng"].max() <= 180

    # Best-estimate deaths sanity: non-negative and at least a moderate
    # global total (millions over 36+ years is well-established).
    assert np.all(events["best"] >= 0)
    total_best = int(events["best"].sum())
    print(f"  total best-estimate deaths over the series: {total_best:,}")
    assert total_best > 500_000, (
        f"total best-estimate deaths {total_best:,} is implausibly low — "
        "the 'best' column may be misaligned."
    )
    print("  PASS: UCDP-GED loaded with realistic temporal + spatial spread.")


# ============================================================================
# Natural Earth 10 m rivers — drop-in resolution upgrade
# ============================================================================

def test_ne_10m_rivers_richer_than_shipped_110m():
    """
    The 10 m Natural Earth rivers vector should be a strict resolution
    upgrade over the shipped 110 m vector: same data publisher, more
    features, well-formed GeoJSON.
    """
    if not NE_10M_RIVERS_GEOJSON.exists():
        pytest.skip(
            "Natural Earth 10m rivers not fetched yet — run "
            "`python generate_empirical_inputs.py --ne-rivers`"
        )

    with open(NE_10M_RIVERS_GEOJSON) as f:
        gj10 = json.load(f)
    n10 = len(gj10.get("features", []))

    shipped_110 = DATA_DIR / "ne_110m_rivers.geojson"
    with open(shipped_110) as f:
        gj110 = json.load(f)
    n110 = len(gj110.get("features", []))

    print(f"\n  features: 10m={n10:,}   shipped 110m={n110:,}   "
          f"ratio={n10 / max(n110, 1):.1f}x")
    assert gj10["type"] == "FeatureCollection"
    assert n10 > n110, (
        f"10m vector ({n10}) is not richer than 110m ({n110}) — the "
        "downloader/converter likely picked up the wrong shapefile."
    )
    assert n10 > 500, (
        f"only {n10} features in 10m vector — expected >= 500. "
        "Verify the Natural Earth release."
    )

    # Validate a couple of feature geometries — must be LineString or
    # MultiLineString for a river centreline vector.
    valid_types = {"LineString", "MultiLineString"}
    sample_types = {f["geometry"]["type"] for f in gj10["features"][:50]}
    assert sample_types.issubset(valid_types), (
        f"unexpected geometry types {sample_types - valid_types}"
    )
