#!/usr/bin/env python3
"""
Behaviour fingerprint: per-tick hashes of simulation state, a safety net for
pure refactors (moves and splits must leave every hash unchanged).

Runs (World config as in app.create_world, logger disabled):
  historical  seeds 0, 1, 2   300 ticks  world_model_mode "jepa" (default)
  historical  seed 0         1000 ticks  (reaches settlements and nations)
  present_day seed 0          100 ticks

Per tick, SHA-256 over: tick, number of agents in the list, number alive,
float64 bytes of (lat, lng, energy) of every alive agent in list order,
float64 bytes of every MacroState field in declaration order, the full world
RNG state (MT19937 key, position, cached Gaussian), number of settlements and
their (lat, lng) in list order, number of nations and number of alliances
(unordered nation pairs). At the final tick of each run, a separate SHA-256
over every world-model weight array (encoder, then predictor; keys sorted).

The optional empirical datasets (UCDP-GED, HadCRUT5) are forced absent, so the
present_day run uses the shipped JSON fallback whatever is installed under
data/empirical/. Hashes are bit-exact and therefore specific to the platform
and library versions recorded in the fixture; the test skips elsewhere.

Usage:
    python scripts/fingerprint.py            # compare against the fixture
    python scripts/fingerprint.py --write    # (re)write the fixture
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import empirical_ingest  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from world import World  # noqa: E402

log = logging.getLogger("fingerprint")

FIXTURE = ROOT / "tests" / "fixtures" / "fingerprint.json"
RUNS: tuple[tuple[str, int, int], ...] = (
    ("historical", 0, 300), ("historical", 1, 300), ("historical", 2, 300), ("historical", 0, 1000),
    ("present_day", 0, 100))


def environment() -> dict[str, str]:
    """The platform facts a bit-exact hash depends on."""
    return {"system": platform.system(), "machine": platform.machine(),
            "python": ".".join(platform.python_version_tuple()[:2]),
            "numpy": np.__version__, "scipy": scipy.__version__}


@contextmanager
def empirical_data_absent() -> Iterator[None]:
    """Make the present_day loaders report 'dataset absent' (JSON fallback)."""
    saved = (empirical_ingest.load_present_day_conflicts, empirical_ingest.observed_temperature_anomaly)
    empirical_ingest.load_present_day_conflicts = lambda *args, **kwargs: None
    empirical_ingest.observed_temperature_anomaly = lambda *args, **kwargs: None
    try:
        yield
    finally:
        empirical_ingest.load_present_day_conflicts, empirical_ingest.observed_temperature_anomaly = saved


def tick_digest(world: World) -> str:
    alive = [a for a in world.agents if a.alive]
    state = world.macro.state
    h = hashlib.sha256()
    h.update(np.array([world.tick, len(world.agents), len(alive)], dtype=np.int64).tobytes())
    h.update(np.array([[a.lat, a.lng, a.energy] for a in alive], dtype=np.float64).tobytes())
    h.update(np.array([float(getattr(state, f.name)) for f in dataclasses.fields(state)],
                      dtype=np.float64).tobytes())
    _update_rng(h, world)
    _update_geopolitics(h, world)
    return h.hexdigest()


def _update_rng(h: "hashlib._Hash", world: World) -> None:
    name, key, pos, has_gauss, cached = world.rng.get_state()
    h.update(name.encode())
    h.update(np.asarray(key, dtype=np.uint32).tobytes())
    h.update(np.array([pos, has_gauss], dtype=np.int64).tobytes())
    h.update(np.array([cached], dtype=np.float64).tobytes())


def _update_geopolitics(h: "hashlib._Hash", world: World) -> None:
    nations = world.geopolitics.nations
    alliances = sum(len(n.alliances) for n in nations) // 2
    h.update(np.array([len(world.settlements), len(nations), alliances], dtype=np.int64).tobytes())
    h.update(np.array([[s.lat, s.lng] for s in world.settlements], dtype=np.float64).tobytes())


def weights_digest(world: World) -> str:
    """SHA-256 over all world-model weights: encoder then predictor, keys sorted."""
    h = hashlib.sha256()
    for module in (world.shared_world_model.encoder, world.shared_world_model.predictor):
        for name in sorted(module.params):
            h.update(name.encode())
            h.update(np.ascontiguousarray(module.params[name], dtype=np.float64).tobytes())
    return h.hexdigest()


def fingerprint_run(scenario_id: str, seed: int, ticks: int) -> dict:
    """Per-tick digests and final weight digest of one run, built as app.create_world builds it."""
    scenario = SCENARIOS[scenario_id]
    with empirical_data_absent():
        world = World(seed=seed, cell_size_deg=2.0, config={"start_year_bp": scenario.start_year_bp},
                      scenario_id=scenario_id)
        world.logger.config.enabled = False
        world.spawn_initial_agents(scenario.initial_agents)
        digests = []
        for _ in range(ticks):
            world.step()
            digests.append(tick_digest(world))
    return {"ticks": digests, "final_weights": weights_digest(world)}


def fingerprint() -> dict:
    return {"environment": environment(),
            "runs": {f"{s}/seed{seed}/{t}": fingerprint_run(s, seed, t) for s, seed, t in RUNS}}


def first_mismatches(reference: dict, fresh: dict) -> dict[str, int | str]:
    """Run key -> first differing tick (1-based), or "final_weights"; matching runs are omitted."""
    out: dict[str, int | str] = {}
    for key, ref in reference["runs"].items():
        got = fresh["runs"].get(key, {"ticks": [], "final_weights": ""})
        diff = next((i for i, (a, b) in enumerate(zip(ref["ticks"], got["ticks"])) if a != b), None)
        if diff is None and len(ref["ticks"]) != len(got["ticks"]):
            diff = min(len(ref["ticks"]), len(got["ticks"]))
        if diff is not None:
            out[key] = diff + 1
        elif ref["final_weights"] != got["final_weights"]:
            out[key] = "final_weights"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help="write the fixture instead of comparing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().setLevel(logging.WARNING)  # silence scenario init chatter
    log.setLevel(logging.INFO)
    fresh = fingerprint()
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(fresh, indent=1) + "\n")
        log.info("wrote %s", FIXTURE)
        return 0
    reference = json.loads(FIXTURE.read_text())
    if reference["environment"] != fresh["environment"]:
        log.warning("environment differs from fixture: %s vs %s", fresh["environment"], reference["environment"])
    bad = first_mismatches(reference, fresh)
    for key, tick in bad.items():
        log.info("MISMATCH %s at %s", key, tick)
    log.info("fingerprint %s", "identical" if not bad else "DIFFERS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
