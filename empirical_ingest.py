"""
Empirical data ingestion — runtime-safe (pure numpy + stdlib).

Turns the downloaded empirical datasets under ``data/empirical/`` into the
structures the simulator consumes at initialization. This is deliberately kept
separate from ``generate_empirical_inputs.py`` (the downloaders, which need
rasterio/GDAL): this module imports with zero heavy dependencies, so the
simulator runtime never requires them.

Wired so far (toward v0.4):
- **UCDP-GED** georeferenced conflict events (Uppsala Conflict Data Program,
  CC-BY 4.0) → present-day active conflict zones for the geopolitics layer,
  replacing the shipped illustrative ``present_day_conflicts.json``.

Every ingester degrades gracefully: the ~GB empirical downloads are gitignored
and optional, so if a source file is absent the loader returns ``None`` and the
caller falls back to the shipped data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
EMPIRICAL_DIR = DATA_DIR / "empirical"
UCDP_EVENTS_FILE = EMPIRICAL_DIR / "ucdp_ged_events.npy"
HADCRUT5_FILE = EMPIRICAL_DIR / "hadcrut5_global_annual.npy"


def derive_conflicts_from_ucdp(
    events: np.ndarray,
    recent_years: int = 3,
    top_k: int = 15,
    min_events: int = 5,
) -> list[dict]:
    """
    Aggregate recent UCDP-GED events into present-day active conflict zones.

    ``events`` is the structured array saved by ``generate_empirical_inputs.py``
    with fields ``(year, lat, lng, best, conflict_id)`` — one row per recorded
    violent event, ``best`` being the best fatality estimate. Events from the
    most recent ``recent_years`` of *available* data (not the calendar year, so
    the ingestion is robust to the lag between the dataset and "now") are grouped
    by ``conflict_id``; each group with at least ``min_events`` events becomes a
    conflict zone. The ``top_k`` most lethal are returned.

    Returns a list of conflict dicts in the scenario schema consumed by
    ``ScenarioLoader._init_conflicts`` (keys ``lat, lng, radius_deg, intensity,
    name, parties`` plus provenance fields ``conflict_id, fatalities,
    n_events``), most lethal first. Empty list if there is no usable data.
    """
    if events is None or len(events) == 0:
        return []

    year = events["year"]
    data_max = int(year.max())
    recent = events[year >= data_max - recent_years + 1]
    if len(recent) == 0:
        return []

    conflicts: list[dict] = []
    for cid in np.unique(recent["conflict_id"]):
        ev = recent[recent["conflict_id"] == cid]
        if len(ev) < min_events:
            continue
        lat = float(np.mean(ev["lat"]))
        lng = float(np.mean(ev["lng"]))
        fatalities = int(np.sum(ev["best"]))
        # Spatial spread of the events → conflict radius (deg-equivalents),
        # bounded so a single stray event or a continent-spanning conflict_id
        # cannot produce a degenerate or absurd zone.
        spread = float(np.sqrt(np.var(ev["lat"]) + np.var(ev["lng"])))
        radius = float(np.clip(spread, 1.5, 8.0))
        conflicts.append({
            "conflict_id": int(cid),
            "lat": lat,
            "lng": lng,
            "fatalities": fatalities,
            "n_events": int(len(ev)),
            "radius_deg": radius,
            "name": f"UCDP-{int(cid)}",
            "parties": [],
        })

    if not conflicts:
        return []

    # Intensity: log-scaled fatalities normalized to the batch maximum, mapped to
    # [0.3, 1.0] so the least-lethal seeded conflict still registers and the
    # worst saturates. (Fatality counts are heavy-tailed, hence the log.)
    max_fat = max(c["fatalities"] for c in conflicts)
    denom = np.log10(max_fat + 1.0) or 1.0
    for c in conflicts:
        frac = np.log10(c["fatalities"] + 1.0) / denom
        c["intensity"] = float(np.clip(0.3 + 0.7 * frac, 0.3, 1.0))

    conflicts.sort(key=lambda c: c["fatalities"], reverse=True)
    return conflicts[:top_k]


def load_present_day_conflicts(
    recent_years: int = 3,
    top_k: int = 15,
    min_events: int = 5,
) -> Optional[list[dict]]:
    """
    Load and aggregate UCDP-GED conflicts for the present-day scenario.

    Returns the derived conflict list, or ``None`` if the UCDP-GED dataset is not
    present (so the caller can fall back to the shipped illustrative JSON). Any
    unreadable/corrupt file also yields ``None`` rather than raising, so a bad
    download can never block world initialization.
    """
    if not UCDP_EVENTS_FILE.exists():
        return None
    try:
        events = np.load(UCDP_EVENTS_FILE)
    except (OSError, ValueError):
        return None
    derived = derive_conflicts_from_ucdp(
        events, recent_years=recent_years, top_k=top_k, min_events=min_events)
    return derived or None


# ---------------------------------------------------------------------------
# HadCRUT5 — observed global-mean surface temperature (Met Office + CRU)
# ---------------------------------------------------------------------------
#
# HadCRUT5 anomalies are reported relative to the 1961-1990 climatology. The
# macro model (macro.py) works in anomalies relative to the 1850-1900
# PRE-INDUSTRIAL reference (Nordhaus/IPCC convention). We must convert, or the
# simulator would start ~0.36 degC too cold. The offset is derived FROM THE DATA
# (the 1850-1900 mean), not hardcoded, so it stays self-consistent with whatever
# HadCRUT5 release is installed. Verified: the converted 2011-2020 mean is
# 1.11 degC, matching IPCC AR6 (~1.09 degC).

_PREINDUSTRIAL_LO, _PREINDUSTRIAL_HI = 1850, 1900


def load_hadcrut5() -> Optional[np.ndarray]:
    """Load the HadCRUT5 global annual series, or ``None`` if absent/unreadable.

    Structured array with fields ``(year, anom, sigma_ens, sigma_cov,
    sigma_total)``; ``anom`` is relative to 1961-1990.
    """
    if not HADCRUT5_FILE.exists():
        return None
    try:
        return np.load(HADCRUT5_FILE)
    except (OSError, ValueError):
        return None


def _preindustrial_offset(hadcrut: np.ndarray) -> float:
    """Value to ADD to HadCRUT5 (1961-1990) anomalies to express them relative
    to the 1850-1900 pre-industrial reference. 0.0 if the window is missing."""
    yr = hadcrut["year"]
    pre = hadcrut["anom"][(yr >= _PREINDUSTRIAL_LO) & (yr <= _PREINDUSTRIAL_HI)]
    return -float(pre.mean()) if len(pre) else 0.0


def observed_temperature_anomaly(year: Optional[int] = None) -> Optional[tuple]:
    """
    Observed global-mean temperature anomaly (°C, vs 1850-1900 pre-industrial)
    from HadCRUT5. ``year=None`` returns the most recent available year.

    Returns ``(year, anomaly, sigma_total)`` or ``None`` if the dataset is
    absent or the requested year is not covered.
    """
    h = load_hadcrut5()
    if h is None or len(h) == 0:
        return None
    offset = _preindustrial_offset(h)
    if year is None:
        i = int(np.argmax(h["year"]))
    else:
        idx = np.where(h["year"] == year)[0]
        if len(idx) == 0:
            return None
        i = int(idx[0])
    return (int(h["year"][i]), float(h["anom"][i]) + offset,
            float(h["sigma_total"][i]))


def observed_temperature_series() -> Optional[list]:
    """Full observed record as a list of ``(year, anomaly_vs_preindustrial)``
    tuples (for validating the macro trajectory against observations), or
    ``None`` if the dataset is absent."""
    h = load_hadcrut5()
    if h is None or len(h) == 0:
        return None
    offset = _preindustrial_offset(h)
    return [(int(y), float(a) + offset) for y, a in zip(h["year"], h["anom"])]
