#!/usr/bin/env python3
"""
World-model ablation: does JEPA steering change outcomes versus controls?

Runs the historical scenario headless (no Flask) for N seeds x modes (World
config ``world_model_mode``):
  jepa    JEPA encode + CEM planner (default simulation behaviour)
  random  U(-1, 1)^8 x random_scale ("random action (matched scale)")
  const   movement a[0], a[1] = 0, a[2] = const_intensity ("no steering,
          constant intensity")
  none    all-zero action ("no action vector"; also pins intensity at 0.1)

With --pilot-seeds K > 0, a jepa pilot on K separate seeds (from
--pilot-seed-start) sets random_scale = mean jepa action norm / E||U(-1,1)^8||
and const_intensity = mean jepa intensity, before any evaluation run.

With --reuse-jepa DIR (post hoc design (iii-b)), jepa rows and loss steps are
read from DIR instead of being run, and the controls are matched per component
to those rows: const_intensity = mean jepa intensity; random draws
a[k] ~ U(-s_k, s_k) with s_0 = 2 mean|a[0]|, s_1 = 2 mean|a[1]| (same mean
absolute movement components), s_2 such that E clip(|a[2]|, 0.1, 1) equals the
jepa mean intensity, and s_3..s_7 = 0 (training-only components, unused here).

Per run (ablation_runs.csv): agents alive, survival rate, mean energy, mean
distance per agent, food gathered, settlements, mean action L2 norm, mean
|a[0]|, |a[1]|, |a[2]|, mean intensity, train steps, final and last-10-step
mean training loss, wall clock. Every training step (ablation_train_loss.csv):
tick, prediction loss, regulariser, total loss.

Metric definitions:
  survival_rate    alive at end / all agents ever created in the run
  mean_energy      mean energy of agents alive at the end (NaN if none)
  mean_distance_km great-circle displacement summed over ticks / agents created
  food_gathered    sum of food returned by World.harvest_resource
  mean_action_norm mean L2 norm of the action vector in use, over alive agent-ticks
  mean_intensity   mean behaviour intensity = clip(|a[2]|, 0.1, 1)

Summary (ablation_summary.json/.md): per-mode mean/std; paired per-seed
differences jepa minus each control with a paired bootstrap 95% CI, Wilcoxon
signed-rank p, Holm-adjusted p across the six outcome metrics, and the
smallest paired mean difference detectable with 80% power (paired t, observed
SD of differences) at alpha = 0.05 and at the strictest Holm level 0.05/6.

wall_clock_s is measured per run; with --workers > 1 runs share the CPU, so
treat it as indicative only. Results do not depend on --workers.

Usage:
    python scripts/ablation_world_model.py --seeds 30 --ticks 1000 --pilot-seeds 5
    python scripts/ablation_world_model.py --train-interval 2 --pilot-seeds 5
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, t as t_dist, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from oikoumene.agents.agents import Agent  # noqa: E402
from oikoumene.core.world import (DEFAULT_TRAIN_INTERVAL_TICKS, WORLD_MODEL_MODE_LABELS,  # noqa: E402
                   WORLD_MODEL_MODES, World)

log = logging.getLogger("ablation_world_model")

EARTH_RADIUS_KM = 6371.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_120  # fixed, independent of any simulation RNG
UNIFORM_NORM_SEED = 8        # fixed RNG for the Monte Carlo E||U(-1,1)^8||
OUTCOME_METRICS = ("agents_alive", "survival_rate", "mean_energy", "mean_distance_km",
                   "food_gathered", "settlements")
DIAGNOSTIC_METRICS = ("mean_action_norm", "mean_abs_a0", "mean_abs_a1", "mean_abs_a2", "mean_intensity",
                      "train_steps", "final_train_loss", "train_loss_last10", "wall_clock_s")
METRICS = OUTCOME_METRICS + DIAGNOSTIC_METRICS
CONTROLS = ("random", "const", "none")
POWER = 0.8
ALPHA = 0.05


def performance_cores() -> int:
    """Performance-core count on macOS (sysctl), else os.cpu_count()."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                 capture_output=True, text=True, check=True, timeout=5)
            return max(1, int(out.stdout.strip()))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            log.warning("sysctl perf-core query failed (%s); using os.cpu_count()", exc)
    return os.cpu_count() or 1


def haversine_km(lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lng2 - lng1) / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


class RunTracker:
    """Observes a World from outside: wraps harvest_resource and train_step on the instance."""

    def __init__(self, world: World) -> None:
        self.food = 0.0
        self.losses: list[dict] = []
        self.distance_km = 0.0
        self.actions: list[np.ndarray] = []
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
            self.losses.append({"tick": world.tick, "step": world.shared_world_model.train_steps,
                                "pred_loss": float(out["pred_loss"]), "reg_loss": float(out["reg_loss"]),
                                "total_loss": float(out["total_loss"])})
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
                self.actions.append(np.asarray(a._cached_behavior["raw_action"], dtype=float))
                self.intensities.append(float(a._cached_behavior["intensity"]))

    def action_metrics(self) -> dict:
        if not self.actions:
            return {k: float("nan") for k in ("mean_action_norm", "mean_abs_a0", "mean_abs_a1", "mean_abs_a2")}
        acts = np.array(self.actions)
        return {"mean_action_norm": float(np.linalg.norm(acts, axis=1).mean()),
                "mean_abs_a0": float(np.abs(acts[:, 0]).mean()),
                "mean_abs_a1": float(np.abs(acts[:, 1]).mean()),
                "mean_abs_a2": float(np.abs(acts[:, 2]).mean())}


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def run_once(seed: int, mode: str, ticks: int, agents: int, random_scale: float = 1.0,
             const_intensity: float = 0.31, train_interval: int = DEFAULT_TRAIN_INTERVAL_TICKS,
             loss_log: list | None = None) -> dict:
    """One headless historical run; returns the metrics row (loss steps appended to loss_log)."""
    t0 = time.perf_counter()
    world = World(seed=seed, config={"start_year_bp": 70000, "world_model_mode": mode,
                                     "random_steering_scale": random_scale, "const_intensity": const_intensity,
                                     "train_interval_ticks": train_interval})
    world.logger.config.enabled = False  # no logs/ output from ablation runs
    world.spawn_initial_agents(agents)
    tracker = RunTracker(world)
    for _ in range(ticks):
        before = {a.id: (a.lat, a.lng) for a in world.agents if a.alive}
        world.step()
        tracker.record_tick(before, world)
    alive = [a for a in world.agents if a.alive]
    created = Agent._next_id  # reset to 0 in World.__init__, so this counts this run's agents
    totals = [s["total_loss"] for s in tracker.losses]
    if loss_log is not None:
        loss_log.extend({"seed": seed, "mode": mode, **s} for s in tracker.losses)
    return {
        "seed": seed, "mode": mode, "mode_label": WORLD_MODEL_MODE_LABELS[mode], "ticks": ticks,
        "initial_agents": agents, "random_scale": random_scale, "const_intensity": const_intensity,
        "train_interval": train_interval,
        "agents_alive": len(alive),
        "survival_rate": len(alive) / created if created else float("nan"),
        "mean_energy": _mean_or_nan([a.energy for a in alive]),
        "mean_distance_km": tracker.distance_km / created if created else float("nan"),
        "food_gathered": tracker.food,
        "settlements": len(world.settlements),
        **tracker.action_metrics(),
        "mean_intensity": _mean_or_nan(tracker.intensities),
        "train_steps": world.shared_world_model.train_steps,
        "final_train_loss": totals[-1] if totals else float("nan"),
        "train_loss_last10": float(np.mean(totals[-10:])) if totals else float("nan"),
        "wall_clock_s": time.perf_counter() - t0,
    }


def _run_job(job: dict) -> tuple[dict, list]:
    losses: list = []
    row = run_once(**job, loss_log=losses)
    return row, losses


def run_many(jobs: list[dict], workers: int) -> tuple[list[dict], list[dict]]:
    """Run jobs (kwargs for run_once) in order; results do not depend on ``workers``."""
    if workers <= 1:
        results = [_run_job(j) for j in jobs]
    else:
        with get_context("spawn").Pool(workers) as pool:
            results = pool.map(_run_job, jobs, chunksize=1)
    rows = [r for r, _ in results]
    losses = [s for _, ls in results for s in ls]
    return rows, losses


def expected_uniform_norm(dim: int = 8, n: int = 200_000) -> float:
    """Monte Carlo E||U(-1,1)^dim|| with a fixed RNG (~1.6125 for dim 8)."""
    draws = np.random.RandomState(UNIFORM_NORM_SEED).uniform(-1.0, 1.0, (n, dim))
    return float(np.linalg.norm(draws, axis=1).mean())


def pilot(args: argparse.Namespace) -> dict:
    """jepa pilot on separate seeds; returns matched random_scale and const_intensity."""
    seeds = list(range(args.pilot_seed_start, args.pilot_seed_start + args.pilot_seeds))
    jobs = [dict(seed=s, mode="jepa", ticks=args.ticks, agents=args.agents,
                 train_interval=args.train_interval) for s in seeds]
    rows, _ = run_many(jobs, args.workers)
    norm = float(np.nanmean([r["mean_action_norm"] for r in rows]))
    intensity = float(np.nanmean([r["mean_intensity"] for r in rows]))
    result = {"seeds": seeds, "jepa_mean_action_norm": norm, "jepa_mean_intensity": intensity,
              "expected_uniform_norm": expected_uniform_norm(),
              "jepa_mean_abs_a2": float(np.nanmean([r["mean_abs_a2"] for r in rows]))}
    result["random_scale"] = norm / result["expected_uniform_norm"]
    result["const_intensity"] = float(np.clip(intensity, 0.1, 1.0))
    log.info("pilot: jepa norm %.4f, intensity %.4f -> random_scale %.4f, const_intensity %.4f",
             norm, intensity, result["random_scale"], result["const_intensity"])
    return result


def uniform_scale_for_intensity(target: float) -> float:
    """s with E clip(|U(-s, s)|, 0.1, 1) = target, i.e. (0.005 + s^2 / 2) / s = target for 0.1 <= s <= 1."""
    s = target + float(np.sqrt(target**2 - 0.01)) if target >= 0.1 else float("nan")
    if not 0.1 <= s <= 1.0:
        raise ValueError(f"no uniform scale in [0.1, 1] gives mean intensity {target}")
    return s


def match_per_component(jepa_rows: list[dict]) -> dict:
    """Per-component random scales and const intensity matched to jepa evaluation rows."""
    m0, m1, intensity = (float(np.mean([r[k] for r in jepa_rows]))
                         for k in ("mean_abs_a0", "mean_abs_a1", "mean_intensity"))
    scales = [2.0 * m0, 2.0 * m1, uniform_scale_for_intensity(intensity)] + [0.0] * 5
    log.info("per-component match: jepa |a0| %.4f |a1| %.4f intensity %.4f -> scales %s",
             m0, m1, intensity, np.round(scales, 4).tolist())
    return {"source": "jepa evaluation rows (reused)", "seeds": sorted(r["seed"] for r in jepa_rows),
            "jepa_mean_abs_a0": m0, "jepa_mean_abs_a1": m1, "jepa_mean_intensity": intensity,
            "random_scale": scales, "const_intensity": intensity}


def _parse_cell(value: str) -> object:
    try:
        return float(value)
    except ValueError:
        return json.loads(value) if value.startswith("[") else value


def load_reused(run_dir: Path, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """jepa rows and loss steps from an earlier run; checks they match this configuration."""
    with open(run_dir / "ablation_runs.csv", newline="") as f:
        rows = [{k: _parse_cell(v) for k, v in r.items()} for r in csv.DictReader(f) if r["mode"] == "jepa"]
    with open(run_dir / "ablation_train_loss.csv", newline="") as f:
        losses = [{k: _parse_cell(v) for k, v in r.items()} for r in csv.DictReader(f) if r["mode"] == "jepa"]
    for r in rows + losses:
        r["seed"] = int(r["seed"])
    wanted = set(range(args.seed_start, args.seed_start + args.seeds))
    config = {(r["ticks"], r["initial_agents"], r["train_interval"]) for r in rows}
    if {r["seed"] for r in rows} != wanted or config != {(args.ticks, args.agents, args.train_interval)}:
        raise ValueError(f"reused jepa runs in {run_dir} do not match seeds/ticks/agents/train interval")
    return rows, losses


def min_detectable_difference(sd: float, n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest |mean paired difference| a two-sided paired t-test detects with ``power``."""
    if n < 2 or not np.isfinite(sd) or sd == 0:
        return float("nan")
    df = n - 1
    tcrit = t_dist.ppf(1 - alpha / 2, df)

    def power_gap(delta: float) -> float:
        nc = delta * np.sqrt(n) / sd
        achieved = 1 - nct.cdf(tcrit, df, nc) + nct.cdf(-tcrit, df, nc)
        # nct.cdf can return NaN at very large noncentrality (small df, strict alpha),
        # where power is effectively 1.
        return (achieved if np.isfinite(achieved) else 1.0) - power

    # Upper bracket: noncentrality tcrit + 5 gives power ~1; larger nc makes nct.cdf return NaN.
    return float(brentq(power_gap, 0.0, (tcrit + 5.0) * sd / np.sqrt(n)))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values; NaN p-values are left out and stay NaN."""
    valid = sorted((p, k) for k, p in pvalues.items() if np.isfinite(p))
    m, running, adjusted = len(valid), 0.0, {k: float("nan") for k in pvalues}
    for i, (p, k) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted


def paired_stats(diffs: np.ndarray) -> dict:
    """Mean paired difference, paired bootstrap 95% CI, Wilcoxon p, 80%-power MDE."""
    diffs = diffs[~np.isnan(diffs)]
    nan = float("nan")
    if len(diffs) == 0:
        return {"n": 0, "mean_diff": nan, "sd_diff": nan, "ci_low": nan, "ci_high": nan,
                "wilcoxon_p": nan, "mde80": nan, "mde80_holm": nan}
    boot_rng = np.random.RandomState(BOOTSTRAP_SEED)
    boot_means = diffs[boot_rng.randint(0, len(diffs), size=(BOOTSTRAP_RESAMPLES, len(diffs)))].mean(axis=1)
    if np.all(diffs == 0):
        p = 1.0  # identical outcomes: no evidence of a difference
    else:
        try:
            p = float(wilcoxon(diffs).pvalue)
        except ValueError as exc:  # e.g. too few non-zero differences
            log.warning("wilcoxon failed (%s); reporting NaN", exc)
            p = nan
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else nan
    return {"n": int(len(diffs)), "mean_diff": float(diffs.mean()), "sd_diff": sd,
            "ci_low": float(np.percentile(boot_means, 2.5)), "ci_high": float(np.percentile(boot_means, 97.5)),
            "wilcoxon_p": p, "mde80": min_detectable_difference(sd, len(diffs)),
            "mde80_holm": min_detectable_difference(sd, len(diffs), alpha=ALPHA / len(OUTCOME_METRICS))}


def _paired_block(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b))
    stats = {k: paired_stats(np.array([a[s][k] - b[s][k] for s in seeds], dtype=float)) for k in METRICS}
    holm = holm_adjust({k: stats[k]["wilcoxon_p"] for k in OUTCOME_METRICS})
    for k in METRICS:
        stats[k]["holm_p"] = holm.get(k, float("nan"))
    return stats


def summarize(rows: list[dict]) -> dict:
    by_mode = {m: {r["seed"]: r for r in rows if r["mode"] == m} for m in WORLD_MODEL_MODES}
    per_mode = {m: {k: {"mean": float(np.nanmean([r[k] for r in runs.values()])),
                        "std": float(np.nanstd([r[k] for r in runs.values()], ddof=1))
                        if len(runs) > 1 else float("nan")}
                    for k in METRICS}
                for m, runs in by_mode.items() if runs}
    paired = {f"jepa-{c}": _paired_block(by_mode["jepa"], by_mode[c])
              for c in CONTROLS if by_mode["jepa"] and by_mode[c]}
    return {"per_mode": per_mode, "paired": paired}


def summary_markdown(summary: dict, header: str) -> str:
    modes = list(summary["per_mode"])
    lines = [header, "", "| metric | " + " | ".join(f"{m} ({WORLD_MODEL_MODE_LABELS[m]}) mean ± sd" for m in modes)
             + " |", "|---|" + "---|" * len(modes)]
    for k in METRICS:
        cells = [f"{summary['per_mode'][m][k]['mean']:.4g} ± {summary['per_mode'][m][k]['std']:.3g}" for m in modes]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    for pair, stats in summary["paired"].items():
        control = pair.split("-", 1)[1]
        lines += ["", f"Paired differences {pair} [{WORLD_MODEL_MODE_LABELS[control]}] "
                      f"(n seeds = {stats[METRICS[0]]['n']}; Holm across the {len(OUTCOME_METRICS)} outcome metrics)",
                  "", "| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |",
                  "|---|---|---|---|---|---|---|"]
        for k in METRICS:
            s = stats[k]
            lines.append(f"| {k} | {s['mean_diff']:.4g} | [{s['ci_low']:.4g}, {s['ci_high']:.4g}] | "
                         f"{s['wilcoxon_p']:.3g} | {s['holm_p']:.3g} | {s['mde80']:.4g} | {s['mde80_holm']:.4g} |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_raw(rows: list[dict], losses: list[dict], out_dir: Path) -> None:
    """Write per-run and per-training-step CSVs (before any statistics can fail)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ablation_runs.csv", rows)
    _write_csv(out_dir / "ablation_train_loss.csv", losses)


def write_summary(summary: dict, args: argparse.Namespace, pilot_result: dict | None) -> None:
    meta = {k: str(v) for k, v in vars(args).items()}
    (args.out_dir / "ablation_summary.json").write_text(
        json.dumps({"args": meta, "note": args.note, "pilot": pilot_result, **summary}, indent=2))
    header = (f"Config: {args.agents} agents, {args.ticks} ticks, train interval {args.train_interval}, "
              f"seeds {args.seed_start}..{args.seed_start + args.seeds - 1}, "
              f"random_scale {np.round(args.random_scale, 4).tolist()}, const_intensity {args.const_intensity:.4f}"
              + (f"\n\nNote: {args.note}" if args.note else "")
              + (f"\n\nPilot: {json.dumps(pilot_result)}" if pilot_result else ""))
    md = summary_markdown(summary, header)
    (args.out_dir / "ablation_summary.md").write_text(md)
    log.info("\n%s", md)
    log.info("wrote %s", args.out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=10, help="number of evaluation seeds")
    ap.add_argument("--seed-start", type=int, default=0, help="first evaluation seed")
    ap.add_argument("--ticks", type=int, default=1000, help="ticks per run (~Industrial era onset)")
    ap.add_argument("--agents", type=int, default=25, help="initial agents")
    ap.add_argument("--random-scale", type=float, default=1.0,
                    help="scale of the U(-1,1)^8 random action (1.0 = full CEM bounds)")
    ap.add_argument("--const-intensity", type=float, default=0.31,
                    help="a[2] (= intensity) in const mode; default is the jepa mean from a pilot run")
    ap.add_argument("--train-interval", type=int, default=DEFAULT_TRAIN_INTERVAL_TICKS,
                    help="ticks between world-model training steps")
    ap.add_argument("--pilot-seeds", type=int, default=0,
                    help="if > 0, a jepa pilot on this many seeds sets --random-scale and --const-intensity")
    ap.add_argument("--pilot-seed-start", type=int, default=1000, help="first pilot seed (kept apart from evaluation)")
    ap.add_argument("--modes", nargs="+", default=list(WORLD_MODEL_MODES), choices=WORLD_MODEL_MODES)
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: performance cores)")
    ap.add_argument("--reuse-jepa", type=Path, default=None,
                    help="read jepa rows from this earlier output dir and match controls per component to them")
    ap.add_argument("--note", default="", help="free text recorded in the summary")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "scripts" / "results" / "ablation_world_model")
    args = ap.parse_args(argv)
    if args.workers is None:
        args.workers = performance_cores()
    if args.reuse_jepa is not None and args.pilot_seeds > 0:
        ap.error("--reuse-jepa and --pilot-seeds are alternative matching designs")
    return args


def prepare_controls(args: argparse.Namespace) -> tuple[dict | None, list[dict], list[dict]]:
    """Set args.random_scale / const_intensity from a pilot or reused jepa rows; returns (matching, rows, losses)."""
    if args.reuse_jepa is not None:
        rows, losses = load_reused(args.reuse_jepa, args)
        args.modes = [m for m in args.modes if m != "jepa"]
        matching = match_per_component(rows)
    elif args.pilot_seeds > 0:
        rows, losses, matching = [], [], pilot(args)
    else:
        return None, [], []
    args.random_scale, args.const_intensity = matching["random_scale"], matching["const_intensity"]
    return matching, rows, losses


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pilot_result, reused_rows, reused_losses = prepare_controls(args)
    jobs = [dict(seed=s, mode=m, ticks=args.ticks, agents=args.agents, random_scale=args.random_scale,
                 const_intensity=args.const_intensity, train_interval=args.train_interval)
            for s in range(args.seed_start, args.seed_start + args.seeds) for m in args.modes]
    log.info("%d runs on %d workers", len(jobs), args.workers)
    rows, losses = run_many(jobs, args.workers)
    for r in rows:
        log.info("seed=%d mode=%-6s alive=%d food=%.1f dist=%.0fkm norm=%.3f train=%d %.1fs", r["seed"], r["mode"],
                 r["agents_alive"], r["food_gathered"], r["mean_distance_km"], r["mean_action_norm"],
                 r["train_steps"], r["wall_clock_s"])
    rows, losses = reused_rows + rows, reused_losses + losses
    write_raw(rows, losses, args.out_dir)
    write_summary(summarize(rows), args, pilot_result)


if __name__ == "__main__":
    main()
