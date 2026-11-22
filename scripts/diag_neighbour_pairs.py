#!/usr/bin/env python3
"""
Diagnostic for finding B10: neighbour pre-filter in raw (lat, lng) degrees.

World.get_nearby_agents builds a cKDTree on raw (lat, lng) and queries a
Euclidean radius in degrees. This script counts, for the radii the
simulation actually queries, which agent pairs that query returns versus a
great-circle query (unit-sphere 3D tree, chord radius). The great-circle
radius uses the same degree-equivalent convention as World._distance_deg
(1 deg = 111 km, R = 6371 km). Nothing in world.py is changed.

Output: one JSON summary (default scripts/results/diag_neighbour_pairs.json)
and a printed table.

Usage:
    python scripts/diag_neighbour_pairs.py --seeds 5 --ticks 1000 --sample-every 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from oikoumene.core.world import World  # noqa: E402

log = logging.getLogger("diag_neighbour_pairs")

RADII_DEG = (3.0, 5.0, 10.0)  # radii passed to get_nearby_agents (agents.py, world.py)
KM_PER_DEG = 111.0
EARTH_RADIUS_KM = 6371.0


def unit_vectors(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """(lat, lng) in degrees -> unit-sphere xyz."""
    phi, lam = np.radians(lat), np.radians(lng)
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def chord_for_radius(radius_deg: float) -> float:
    """Chord length equivalent to a great-circle radius in degree-equivalents."""
    angle = radius_deg * KM_PER_DEG / EARTH_RADIUS_KM
    return float(2.0 * np.sin(angle / 2.0))


def pair_sets(lat: np.ndarray, lng: np.ndarray, radius_deg: float) -> tuple[set, set]:
    """Unordered pairs within radius: (raw-degree Euclidean, great-circle)."""
    raw = cKDTree(np.column_stack([lat, lng])).query_pairs(radius_deg)
    sphere = cKDTree(unit_vectors(lat, lng)).query_pairs(chord_for_radius(radius_deg))
    return raw, sphere


def classify(pairs: set, lat: np.ndarray, lng: np.ndarray) -> dict:
    """Count pairs at high latitude (either end |lat| >= 50) and across the dateline."""
    high = sum(1 for i, j in pairs if max(abs(lat[i]), abs(lat[j])) >= 50.0)
    dateline = sum(1 for i, j in pairs if abs(lng[i] - lng[j]) > 180.0)
    return {"n": len(pairs), "high_lat": high, "dateline": dateline}


def compare(lat: np.ndarray, lng: np.ndarray) -> dict:
    """Per radius: pairs found by both methods, raw-only, sphere-only."""
    out = {}
    for r in RADII_DEG:
        raw, sphere = pair_sets(lat, lng, r)
        out[str(r)] = {
            "both": classify(raw & sphere, lat, lng),
            "raw_only": classify(raw - sphere, lat, lng),
            "sphere_only": classify(sphere - raw, lat, lng),
        }
    return out


def synthetic_cases() -> dict:
    """Hand-built pairs: 60 N along a parallel, and straddling 180 deg."""
    cases = {
        "60N_4deg_lng_apart": ([60.0, 60.0], [10.0, 14.0]),   # ~222 km apart
        "60N_8deg_lng_apart": ([60.0, 60.0], [10.0, 18.0]),   # ~444 km apart
        "equator_dateline_2deg": ([0.0, 0.0], [179.0, -179.0]),  # ~222 km apart
    }
    result = {}
    for name, (lat, lng) in cases.items():
        lat_a, lng_a = np.array(lat), np.array(lng)
        result[name] = {str(r): {"raw": len(pair_sets(lat_a, lng_a, r)[0]) == 1,
                                 "sphere": len(pair_sets(lat_a, lng_a, r)[1]) == 1}
                        for r in RADII_DEG}
    return result


def add_counts(total: dict, part: dict) -> None:
    for r, groups in part.items():
        for g, counts in groups.items():
            for k, v in counts.items():
                total.setdefault(r, {}).setdefault(g, {}).setdefault(k, 0)
                total[r][g][k] += v


def run_seed(seed: int, ticks: int, sample_every: int, agents: int) -> tuple[dict, int]:
    world = World(seed=seed)
    world.logger.config.enabled = False
    world.spawn_initial_agents(agents)
    total: dict = {}
    snapshots = 0
    for t in range(1, ticks + 1):
        world.step()
        if t % sample_every:
            continue
        alive = [a for a in world.agents if a.alive]
        if len(alive) < 2:
            continue
        add_counts(total, compare(np.array([a.lat for a in alive]), np.array([a.lng for a in alive])))
        snapshots += 1
    return total, snapshots


def print_table(total: dict) -> None:
    log.info("radius | both | raw_only (hi-lat/dateline) | sphere_only (hi-lat/dateline)")
    for r, g in total.items():
        ro, so = g["raw_only"], g["sphere_only"]
        log.info("%6s | %d | %d (%d/%d) | %d (%d/%d)", r, g["both"]["n"], ro["n"], ro["high_lat"],
                 ro["dateline"], so["n"], so["high_lat"], so["dateline"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--agents", type=int, default=25)
    ap.add_argument("--out", type=Path, default=ROOT / "scripts" / "results" / "diag_neighbour_pairs.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    total: dict = {}
    snapshots = 0
    for seed in range(args.seeds):
        part, n = run_seed(seed, args.ticks, args.sample_every, args.agents)
        add_counts(total, part)
        snapshots += n
    log.info("historical scenario, %d seeds x %d ticks, %d snapshots (pairs summed over snapshots)",
             args.seeds, args.ticks, snapshots)
    print_table(total)
    synth = synthetic_cases()
    log.info("synthetic cases (pair found?): %s", json.dumps(synth))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                    "snapshots": snapshots, "pairs": total, "synthetic": synth}, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
