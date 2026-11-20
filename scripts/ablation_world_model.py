#!/usr/bin/env python3
"""
World-model ablation: does JEPA steering change outcomes versus no steering
or random steering of the same range?

Runs the historical scenario headless (no Flask) for N seeds x 3 modes
(World config ``world_model_mode`` = jepa / none / random) and records per run:
agents alive, survival rate, mean energy, mean distance travelled per agent,
food gathered, settlements, mean action L2 norm, mean action intensity,
world-model train steps and final training loss, wall-clock time.

Metric definitions:
  survival_rate    alive at end / all agents ever created in the run
  mean_energy      mean energy of agents alive at the end (NaN if none)
  mean_distance_km great-circle displacement summed over ticks, divided by
                   the number of agents ever created
  food_gathered    sum of food returned by World.harvest_resource (agents
                   eating and businesses harvesting)
  mean_action_norm mean L2 norm of the action vector in use, over alive
                   agent-ticks (planner output, zero, or random draw)
  mean_intensity   mean behaviour intensity = clip(|action[2]|, 0.1, 1)

Outputs (in --out-dir): ablation_runs.csv (one row per run),
ablation_summary.json and ablation_summary.md (per-mode mean/std, paired
per-seed differences jepa-none and jepa-random with a paired bootstrap 95% CI
and a Wilcoxon signed-rank p-value).

Usage:
    python scripts/ablation_world_model.py --seeds 10 --ticks 1000 --agents 25
    python scripts/ablation_world_model.py --random-scale 0.4   # match jepa magnitude
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import Agent  # noqa: E402
from world import WORLD_MODEL_MODES, World  # noqa: E402

log = logging.getLogger("ablation_world_model")

EARTH_RADIUS_KM = 6371.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_120  # fixed, independent of any simulation RNG
METRICS = ("agents_alive", "survival_rate", "mean_energy", "mean_distance_km", "food_gathered",
           "settlements", "mean_action_norm", "mean_intensity", "train_steps", "final_train_loss",
           "wall_clock_s")
PAIRS = (("jepa", "none"), ("jepa", "random"))


def haversine_km(lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lng2 - lng1) / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


class RunTracker:
    """Observes a World from outside: wraps harvest_resource and train_step on the instance."""

    def __init__(self, world: World) -> None:
        self.food = 0.0
        self.last_loss = float("nan")
        self.distance_km = 0.0
        self.action_norms: list[float] = []
        self.intensities: list[float] = []
        self._wrap(world)

    def _wrap(self, world: World) -> None:
        harvest, train = world.harvest_resource, world.shared_world_model.train_step

        def tracked_harvest(lat: float, lng: float, resource_type: str, amount: float) -> float:
            got = harvest(lat, lng, resource_type, amount)
            if resource_type == "food":
                self.food += got
            return got

        def tracked_train(*args, **kwargs) -> dict:
            out = train(*args, **kwargs)
            self.last_loss = float(out.get("total_loss", float("nan")))
            return out

        world.harvest_resource = tracked_harvest
        world.shared_world_model.train_step = tracked_train

    def record_tick(self, before: dict[int, tuple[float, float]], world: World) -> None:
        moved = [(before[a.id], (a.lat, a.lng)) for a in world.agents if a.id in before]
        if moved:
            arr = np.array([[p[0], p[1], q[0], q[1]] for p, q in moved])
            self.distance_km += float(haversine_km(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]).sum())
        for a in world.agents:
            if a.alive and a._cached_behavior is not None:
                self.action_norms.append(float(np.linalg.norm(a._cached_behavior["raw_action"])))
                self.intensities.append(float(a._cached_behavior["intensity"]))


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def run_once(seed: int, mode: str, ticks: int, agents: int, random_scale: float) -> dict:
    """One headless historical run; returns the metrics row."""
    t0 = time.perf_counter()
    world = World(seed=seed, config={"start_year_bp": 70000, "world_model_mode": mode,
                                     "random_steering_scale": random_scale})
    world.logger.config.enabled = False  # no logs/ output from ablation runs
    world.spawn_initial_agents(agents)
    tracker = RunTracker(world)
    for _ in range(ticks):
        before = {a.id: (a.lat, a.lng) for a in world.agents if a.alive}
        world.step()
        tracker.record_tick(before, world)
    alive = [a for a in world.agents if a.alive]
    created = Agent._next_id  # reset to 0 in World.__init__, so this counts this run's agents
    return {
        "seed": seed, "mode": mode, "ticks": ticks, "initial_agents": agents, "random_scale": random_scale,
        "agents_alive": len(alive),
        "survival_rate": len(alive) / created if created else float("nan"),
        "mean_energy": _mean_or_nan([a.energy for a in alive]),
        "mean_distance_km": tracker.distance_km / created if created else float("nan"),
        "food_gathered": tracker.food,
        "settlements": len(world.settlements),
        "mean_action_norm": _mean_or_nan(tracker.action_norms),
        "mean_intensity": _mean_or_nan(tracker.intensities),
        "train_steps": world.shared_world_model.train_steps,
        "final_train_loss": tracker.last_loss,
        "wall_clock_s": time.perf_counter() - t0,
    }


def paired_stats(diffs: np.ndarray) -> dict:
    """Mean paired difference, paired bootstrap 95% CI, Wilcoxon signed-rank p."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "wilcoxon_p": float("nan")}
    boot_rng = np.random.RandomState(BOOTSTRAP_SEED)
    idx = boot_rng.randint(0, len(diffs), size=(BOOTSTRAP_RESAMPLES, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    if np.all(diffs == 0):
        p = 1.0  # identical outcomes: no evidence of a difference
    else:
        try:
            p = float(wilcoxon(diffs).pvalue)
        except ValueError as exc:  # e.g. too few non-zero differences
            log.warning("wilcoxon failed (%s); reporting NaN", exc)
            p = float("nan")
    return {"n": int(len(diffs)), "mean_diff": float(diffs.mean()),
            "ci_low": float(np.percentile(boot_means, 2.5)),
            "ci_high": float(np.percentile(boot_means, 97.5)), "wilcoxon_p": p}


def summarize(rows: list[dict]) -> dict:
    by_mode = {m: {r["seed"]: r for r in rows if r["mode"] == m} for m in WORLD_MODEL_MODES}
    per_mode = {m: {k: {"mean": float(np.nanmean([r[k] for r in runs.values()])),
                        "std": float(np.nanstd([r[k] for r in runs.values()], ddof=1))
                        if len(runs) > 1 else float("nan")}
                    for k in METRICS}
                for m, runs in by_mode.items() if runs}
    paired = {}
    for a, b in PAIRS:
        seeds = sorted(set(by_mode[a]) & set(by_mode[b]))
        if not seeds:
            continue
        paired[f"{a}-{b}"] = {k: paired_stats(np.array([by_mode[a][s][k] - by_mode[b][s][k] for s in seeds],
                                                        dtype=float))
                              for k in METRICS}
    return {"per_mode": per_mode, "paired": paired}


def summary_markdown(summary: dict) -> str:
    modes = list(summary["per_mode"])
    lines = ["| metric | " + " | ".join(f"{m} mean ± std" for m in modes) + " |",
             "|---|" + "---|" * len(modes)]
    for k in METRICS:
        cells = [f"{summary['per_mode'][m][k]['mean']:.4g} ± {summary['per_mode'][m][k]['std']:.3g}" for m in modes]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    for pair, stats in summary["paired"].items():
        lines += ["", f"Paired differences {pair} (n seeds = {stats[METRICS[0]]['n']})", "",
                  "| metric | mean diff | 95% bootstrap CI | Wilcoxon p |", "|---|---|---|---|"]
        for k in METRICS:
            s = stats[k]
            lines.append(f"| {k} | {s['mean_diff']:.4g} | [{s['ci_low']:.4g}, {s['ci_high']:.4g}] "
                         f"| {s['wilcoxon_p']:.3g} |")
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict], summary: dict, args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "ablation_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {k: str(v) for k, v in vars(args).items()}
    (args.out_dir / "ablation_summary.json").write_text(json.dumps({"args": meta, **summary}, indent=2))
    md = summary_markdown(summary)
    (args.out_dir / "ablation_summary.md").write_text(md)
    log.info("\n%s", md)
    log.info("wrote %s", args.out_dir)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed")
    ap.add_argument("--ticks", type=int, default=1000, help="ticks per run (~Industrial era onset)")
    ap.add_argument("--agents", type=int, default=25, help="initial agents")
    ap.add_argument("--random-scale", type=float, default=1.0,
                    help="scale of the U(-1,1)^8 random action (1.0 = full CEM bounds)")
    ap.add_argument("--modes", nargs="+", default=list(WORLD_MODEL_MODES), choices=WORLD_MODEL_MODES)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "scripts" / "results" / "ablation_world_model")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for mode in args.modes:
            row = run_once(seed, mode, args.ticks, args.agents, args.random_scale)
            log.info("seed=%d mode=%-6s alive=%d food=%.1f dist=%.0fkm norm=%.3f train=%d %.1fs",
                     seed, mode, row["agents_alive"], row["food_gathered"], row["mean_distance_km"],
                     row["mean_action_norm"], row["train_steps"], row["wall_clock_s"])
            rows.append(row)
    write_outputs(rows, summarize(rows), args)


if __name__ == "__main__":
    main()
