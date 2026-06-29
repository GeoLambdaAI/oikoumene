#!/usr/bin/env python3
"""
Sobol global sensitivity analysis for the macro layer.

Decomposes the variance of three 2100 BAU outputs — temperature anomaly,
atmospheric CO2 concentration, and global population — into the
contributions of eight key physical, carbon-cycle, emissions, and
demographic parameters. Uses Saltelli sampling and the Sobol variance
decomposition (SALib).

Outputs:
  scripts/results/sensitivity_N<N>.json — full first-order (S1) and total
    (ST) indices with confidence intervals.
  stdout: a formatted markdown table ready to drop into docs/validation.md.

The eight parameters and their priors are grounded in published
literature ranges (see PARAMETERS below). Each prior is a flat U(lo, hi)
spanning roughly the IPCC AR6 "likely" envelope for that quantity, or
the comparable consensus envelope for the carbon-cycle and demographic
parameters.

Usage:
    python scripts/sensitivity.py             # N=64 (recommended for docs)
    python scripts/sensitivity.py --n 128     # tighter CIs, longer runtime
    python scripts/sensitivity.py --n 4       # smoke test only

Requires: SALib. Install with: pip install -e ".[sensitivity]"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure top-level repo is importable when run as `python scripts/sensitivity.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro import MacroModel  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# Parameter definitions — bounds grounded in published literature ranges
# ============================================================================

PARAMETERS = [
    # (name, bounds [lo, hi], description / source)
    ("FORCING_COEFF",               (5.00, 5.70),
     "CO2 radiative forcing coefficient (Myhre 1998 1-sigma)"),
    ("CLIMATE_FEEDBACK",            (0.80, 1.60),
     "climate feedback parameter lambda (IPCC AR6 likely range; gives ECS 2.5-4.5)"),
    ("OCEAN_HEAT_CAPACITY",         (5.0,  10.0),
     "surface-layer ocean heat capacity (Held et al. 2010)"),
    ("DEEP_OCEAN_COUPLING",         (0.50, 1.00),
     "surface-deep ocean coupling (Gregory 2000)"),
    ("NATURAL_ABSORPTION_RATE",     (0.40, 0.60),
     "natural CO2 sink fraction (Friedlingstein 2024 decadal range)"),
    ("ABSORPTION_TEMP_SENSITIVITY", (0.03, 0.10),
     "sink-weakening per degC warming"),
    ("BASE_EMISSION_RATE",          (38.0, 46.0),
     "baseline 2025 CO2 emission rate, GtCO2/yr (GCP 2024 +-10%)"),
    ("POP_GROWTH_BASE",             (0.005, 0.013),
     "base population growth rate (UN WPP envelope)"),
]

# Outputs evaluated at end-of-2100 BAU trajectory
OUTPUT_NAMES = ("T_2100_degC", "CO2_2100_ppm", "Pop_2100_B")


def build_problem() -> dict:
    """SALib `problem` dict for Saltelli sampling and Sobol analysis."""
    return {
        "num_vars": len(PARAMETERS),
        "names":    [p[0] for p in PARAMETERS],
        "bounds":   [list(p[1]) for p in PARAMETERS],
    }


# ============================================================================
# Model evaluation
# ============================================================================

def run_bau_once(param_values: dict[str, float]) -> tuple[float, float, float]:
    """
    Run one BAU 2025-2100 trajectory with the given MacroModel parameter
    overrides and return (T_2100, CO2_2100, pop_2100).
    """
    model = MacroModel(config={"dt_years": 1.0 / 12.0})
    # Instance attributes shadow the class-level defaults — verified that
    # macro.py reads constants via self.<NAME> (e.g. self.FORCING_COEFF).
    for name, value in param_values.items():
        setattr(model, name, float(value))

    while model.state.year < 2100.5:
        model.step({})

    s = model.state
    return (
        float(s.temperature_anomaly),
        float(s.co2_ppm),
        float(s.global_population_billions),
    )


def evaluate_model(
    problem: dict, samples: np.ndarray, verbose: bool = True
) -> np.ndarray:
    """Evaluate the BAU run for every Saltelli sample; return (M, 3) array."""
    n_runs = len(samples)
    outputs = np.empty((n_runs, len(OUTPUT_NAMES)), dtype=np.float64)
    names = problem["names"]
    t0 = time.perf_counter()
    for i, row in enumerate(samples):
        params = {n: v for n, v in zip(names, row)}
        outputs[i] = run_bau_once(params)
        if verbose and (i + 1) % max(1, n_runs // 10) == 0:
            dt = time.perf_counter() - t0
            eta = dt * (n_runs - i - 1) / (i + 1)
            print(f"  evaluated {i+1:>5d}/{n_runs}  "
                  f"elapsed {dt:5.0f}s  eta {eta:5.0f}s")
    return outputs


# ============================================================================
# Reporting
# ============================================================================

def format_markdown_table(
    problem: dict,
    indices_per_output: dict[str, dict],
    n_samples: int,
) -> str:
    """Produce a markdown table summarizing S1 and ST indices per output."""
    names = problem["names"]
    lines = []
    lines.append(
        f"### Sobol sensitivity indices (Saltelli N={n_samples}, "
        f"M={n_samples * (2 * len(names) + 2)} model evaluations)"
    )
    lines.append("")
    header_cols = ["Parameter"]
    for out in OUTPUT_NAMES:
        header_cols.append(f"{out} S1")
        header_cols.append(f"{out} ST")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "---|" * len(header_cols))

    for i, name in enumerate(names):
        row = [name]
        for out in OUTPUT_NAMES:
            s1 = indices_per_output[out]["S1"][i]
            st = indices_per_output[out]["ST"][i]
            row.append(f"{s1:+.2f}")
            row.append(f"{st:+.2f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "*S1 = first-order index (fraction of output variance explained by "
        "this parameter alone). ST = total index (S1 plus interaction "
        "effects). Negative-but-near-zero values reflect Monte-Carlo "
        "noise; treat |S1| < 0.05 as not significant at this N.*"
    )
    return "\n".join(lines)


def print_human_summary(
    indices_per_output: dict[str, dict],
    problem: dict,
) -> None:
    """Print a human-readable per-output ranking of total Sobol indices."""
    names = problem["names"]
    for out in OUTPUT_NAMES:
        st = indices_per_output[out]["ST"]
        order = np.argsort(st)[::-1]
        print(f"\n{out}: total-index ranking")
        for rank, idx in enumerate(order, 1):
            print(f"  {rank}. {names[idx]:<28s}  ST={st[idx]:+.3f}  "
                  f"S1={indices_per_output[out]['S1'][idx]:+.3f}")


# ============================================================================
# CLI
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--n", type=int, default=64,
                   help="Saltelli base sample size (default 64; "
                   "total model runs = N*(2D+2))")
    p.add_argument("--seed", type=int, default=1,
                   help="random seed for Saltelli sampling (default 1)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-run progress messages")
    args = p.parse_args(argv)

    # Lazy SALib imports — the runtime simulator does not need SALib.
    try:
        from SALib.sample import sobol as salib_sample
        from SALib.analyze import sobol as salib_analyze
    except ImportError as exc:
        print(
            "SALib is required. Install with:  pip install -e \".[sensitivity]\"",
            file=sys.stderr,
        )
        return 1

    problem = build_problem()
    D = problem["num_vars"]
    total_runs = args.n * (2 * D + 2)
    print(f"Sobol sensitivity analysis — N={args.n}, D={D}, "
          f"total model evaluations = {total_runs}")

    print("\nSampling Saltelli design ...")
    samples = salib_sample.sample(problem, args.n, seed=args.seed)
    assert samples.shape == (total_runs, D), samples.shape

    print(f"Evaluating MacroModel BAU 2025-2100 for {total_runs} parameter sets ...")
    outputs = evaluate_model(problem, samples, verbose=not args.quiet)

    print("\nComputing Sobol indices ...")
    indices_per_output: dict[str, dict] = {}
    for j, out_name in enumerate(OUTPUT_NAMES):
        Si = salib_analyze.analyze(problem, outputs[:, j], print_to_console=False)
        indices_per_output[out_name] = {
            "S1":     [float(x) for x in Si["S1"]],
            "S1_conf": [float(x) for x in Si["S1_conf"]],
            "ST":     [float(x) for x in Si["ST"]],
            "ST_conf": [float(x) for x in Si["ST_conf"]],
        }

    # Save JSON for reproducibility
    json_path = RESULTS_DIR / f"sensitivity_N{args.n}.json"
    payload = {
        "n_samples":  args.n,
        "total_runs": total_runs,
        "seed":       args.seed,
        "parameters": [
            {"name": n, "bounds": list(b), "description": d}
            for n, b, d in PARAMETERS
        ],
        "outputs":    list(OUTPUT_NAMES),
        "indices":    indices_per_output,
        "output_means_and_stds": {
            out: {"mean": float(outputs[:, j].mean()),
                  "std":  float(outputs[:, j].std())}
            for j, out in enumerate(OUTPUT_NAMES)
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved indices to {json_path.relative_to(ROOT)}")

    print_human_summary(indices_per_output, problem)
    print()
    print(format_markdown_table(problem, indices_per_output, args.n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
