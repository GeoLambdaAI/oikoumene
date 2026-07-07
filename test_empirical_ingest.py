"""
Tests for empirical data ingestion (v0.4).

Covers UCDP-GED → present-day conflict-zone aggregation and its graceful
fallback. Uses a small synthetic UCDP-shaped array so the tests run without the
(gitignored, ~GB) real download; a final test opportunistically validates
against the real file when it is present.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

_UCDP_DTYPE = [("year", "<i4"), ("lat", "<f4"), ("lng", "<f4"),
               ("best", "<i4"), ("conflict_id", "<i4")]


def _make_events(rows):
    return np.array(rows, dtype=_UCDP_DTYPE)


def test_aggregation_groups_by_conflict_and_ranks_by_fatalities():
    from empirical_ingest import derive_conflicts_from_ucdp
    rows = []
    # Conflict 1: many lethal recent events around (50, 36).
    for _ in range(10):
        rows.append((2024, 50.0, 36.0, 100, 1))
    # Conflict 2: fewer, less lethal recent events around (-1, 30).
    for _ in range(6):
        rows.append((2023, -1.0, 30.0, 10, 2))
    # Conflict 3: too few events -> must be dropped (min_events).
    rows.append((2024, 10.0, 10.0, 500, 3))
    # Old event -> outside the recent window, must be ignored.
    rows.append((2005, 50.0, 36.0, 9999, 1))

    out = derive_conflicts_from_ucdp(_make_events(rows), recent_years=3,
                                     top_k=15, min_events=5)
    assert [c["conflict_id"] for c in out] == [1, 2], "grouping/ranking wrong"
    # Centroid of conflict 1 is its recent events only (old event excluded).
    assert abs(out[0]["lat"] - 50.0) < 1e-4 and abs(out[0]["lng"] - 36.0) < 1e-4
    # Most-lethal conflict saturates intensity; all in [0.3, 1.0].
    assert out[0]["intensity"] == 1.0
    assert all(0.3 <= c["intensity"] <= 1.0 for c in out)
    assert all(1.5 <= c["radius_deg"] <= 8.0 for c in out)


def test_schema_matches_init_conflicts():
    """Derived dicts must carry every key ScenarioLoader._init_conflicts reads."""
    from empirical_ingest import derive_conflicts_from_ucdp
    rows = [(2024, 12.0, 34.0, 50, 7) for _ in range(8)]
    out = derive_conflicts_from_ucdp(_make_events(rows))
    assert out, "expected one conflict"
    c = out[0]
    for key in ("lat", "lng", "radius_deg", "intensity", "name", "parties"):
        assert key in c


def test_top_k_and_empty_inputs():
    from empirical_ingest import derive_conflicts_from_ucdp
    rows = []
    for cid in range(20):  # 20 distinct conflicts, each with enough events
        for _ in range(6):
            rows.append((2024, float(cid), 0.0, cid + 1, cid))
    out = derive_conflicts_from_ucdp(_make_events(rows), top_k=5)
    assert len(out) == 5, "top_k not enforced"
    # Empty / None inputs are safe.
    assert derive_conflicts_from_ucdp(_make_events([])) == []
    assert derive_conflicts_from_ucdp(None) == []


def test_loader_returns_none_when_dataset_absent(monkeypatch, tmp_path):
    import empirical_ingest as E
    monkeypatch.setattr(E, "UCDP_EVENTS_FILE", tmp_path / "does_not_exist.npy")
    assert E.load_present_day_conflicts() is None


def test_present_day_scenario_uses_available_conflicts():
    """Integration: the present-day world seeds a plausible set of active
    conflicts (from UCDP-GED if present, else the shipped JSON) — and my
    conflict-realism invariant (seeded conflicts carry no nation dyad) holds."""
    from world import World
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(50)
    conflicts = w.geopolitics.active_conflicts
    assert 1 <= len(conflicts) <= 30
    for c in conflicts:
        assert -90 <= c["lat"] <= 90 and -180 <= c["lng"] <= 180
        assert c["nations"] == []  # seeded real-world conflicts are not emergent dyads


_HADCRUT_DTYPE = [("year", "<i4"), ("anom", "<f4"), ("sigma_ens", "<f4"),
                  ("sigma_cov", "<f4"), ("sigma_total", "<f4")]


def _write_hadcrut(tmp_path, rows):
    arr = np.array(rows, dtype=_HADCRUT_DTYPE)
    p = tmp_path / "hadcrut5_global_annual.npy"
    np.save(p, arr)
    return p


def test_hadcrut_preindustrial_conversion(monkeypatch, tmp_path):
    """The pre-industrial offset must be derived from the 1850-1900 window and
    applied so the returned anomaly is relative to pre-industrial, not 1961-90."""
    import empirical_ingest as E
    # 1850-1900 all at -0.4 (vs 1961-90); a modern year at +1.0 (vs 1961-90).
    rows = [(y, -0.4, 0.05, 0.05, 0.07) for y in range(1850, 1901)]
    rows.append((2025, 1.0, 0.01, 0.01, 0.02))
    monkeypatch.setattr(E, "HADCRUT5_FILE", _write_hadcrut(tmp_path, rows))

    latest = E.observed_temperature_anomaly()  # None-year -> most recent
    assert latest is not None
    year, anom, sigma = latest
    assert year == 2025
    # offset = +0.4, so 1.0 (vs 1961-90) -> 1.4 (vs pre-industrial).
    assert abs(anom - 1.4) < 1e-5
    assert abs(sigma - 0.02) < 1e-5


def test_hadcrut_specific_year_and_series(monkeypatch, tmp_path):
    import empirical_ingest as E
    rows = [(y, -0.4, 0.05, 0.05, 0.07) for y in range(1850, 1901)]
    rows += [(2000, 0.3, 0.02, 0.02, 0.03), (2025, 1.0, 0.01, 0.01, 0.02)]
    monkeypatch.setattr(E, "HADCRUT5_FILE", _write_hadcrut(tmp_path, rows))

    y2000 = E.observed_temperature_anomaly(2000)
    assert y2000 is not None and abs(y2000[1] - 0.7) < 1e-5  # 0.3 + 0.4
    assert E.observed_temperature_anomaly(1700) is None       # not covered
    series = E.observed_temperature_series()
    assert series is not None
    last_year, last_anom = series[-1]
    assert last_year == 2025 and abs(last_anom - 1.4) < 1e-5   # float32 tolerance


def test_hadcrut_absent_returns_none(monkeypatch, tmp_path):
    import empirical_ingest as E
    monkeypatch.setattr(E, "HADCRUT5_FILE", tmp_path / "missing.npy")
    assert E.load_hadcrut5() is None
    assert E.observed_temperature_anomaly() is None
    assert E.observed_temperature_series() is None


def test_present_day_temperature_matches_observation():
    """Integration: when HadCRUT5 is present, the present-day scenario's starting
    temperature equals the observed anomaly (pre-industrial baseline)."""
    from empirical_ingest import observed_temperature_anomaly
    obs = observed_temperature_anomaly()
    if obs is None:
        return  # dataset optional
    from world import World
    w = World(seed=1, scenario_id="present_day")
    w.spawn_initial_agents(30)
    assert abs(w.macro.state.temperature_anomaly - obs[1]) < 1e-6


def test_real_hadcrut_matches_ipcc_if_present():
    """The observed record, converted to pre-industrial, must reproduce IPCC AR6:
    2011-2020 mean ~1.09 degC."""
    from empirical_ingest import observed_temperature_series
    series = observed_temperature_series()
    if series is None:
        return
    decade = [a for (y, a) in series if 2011 <= y <= 2020]
    if len(decade) < 5:
        return
    mean = sum(decade) / len(decade)
    assert 1.0 <= mean <= 1.2, f"2011-2020 mean {mean:.3f} outside IPCC range"


def test_real_ucdp_file_if_present():
    """When the real UCDP-GED download exists, its top zones must be valid and
    recognizable (non-trivial fatalities, sane geography)."""
    from empirical_ingest import load_present_day_conflicts, UCDP_EVENTS_FILE
    if not UCDP_EVENTS_FILE.exists():
        return  # dataset optional; skip silently
    out = load_present_day_conflicts()
    assert out and len(out) <= 15
    assert out[0]["fatalities"] > 0
    assert all(-90 <= c["lat"] <= 90 and -180 <= c["lng"] <= 180 for c in out)
    # Ranked by fatalities, descending.
    fats = [c["fatalities"] for c in out]
    assert fats == sorted(fats, reverse=True)
