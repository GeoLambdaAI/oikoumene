"""
Figure generation pipeline for Oikoumene simulation runs.

Reads ``logs/{run_id}/timeseries.csv`` and produces paper-ready PNGs in
``logs/{run_id}/figures/``. Runs without arguments use the most recent
run; pass ``--run-id YYYYMMDD_HHMMSS`` to target a specific one.

Figures produced:
    1. macro_climate.png       CO2, temperature, sea level (3 panels)
    2. macro_resources.png     fossil, minerals, freshwater, food (4 panels)
    3. macro_society.png       population, GDP, inequality, welfare (4 panels)
    4. agents_population.png   alive agents + births/deaths over time
    5. geopolitics.png         nation count, active conflicts, trade volume
    6. ipcc_validation.png     CO2 + temp with SSP reference bands

Requires:
    pip install -e ".[viz]"   # adds matplotlib + pandas

Usage:
    python scripts/figures.py                        # latest run
    python scripts/figures.py --run-id 20260414_161426
    python scripts/figures.py --list                 # list all runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    _VIZ_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    _VIZ_AVAILABLE = False


def _require_viz() -> None:
    """Exit with a friendly message if matplotlib/pandas are not installed."""
    if not _VIZ_AVAILABLE:
        sys.stderr.write(
            "Missing visualization dependencies. Install with:\n"
            "    pip install matplotlib pandas\n"
            "or:\n"
            "    pip install -e '.[viz]'\n"
        )
        sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

# IPCC AR6 SSP reference values at 2100 — for validation overlay
IPCC_REFERENCES = {
    "SSP1-2.6": {"co2_ppm": 445, "temperature": 1.8, "color": "#2ca02c"},
    "SSP2-4.5": {"co2_ppm": 603, "temperature": 2.7, "color": "#ff7f0e"},
    "SSP3-7.0": {"co2_ppm": 867, "temperature": 3.6, "color": "#d62728"},
    "SSP5-8.5": {"co2_ppm": 1135, "temperature": 4.4, "color": "#8c564b"},
}


def list_runs() -> list[Path]:
    if not LOGS_DIR.exists():
        return []
    return sorted(
        (d for d in LOGS_DIR.iterdir() if d.is_dir() and (d / "timeseries.csv").exists()),
        key=lambda p: p.name,
    )


def latest_run() -> Path | None:
    runs = list_runs()
    return runs[-1] if runs else None


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    csv_path = run_dir / "timeseries.csv"
    meta_path = run_dir / "metadata.json"
    df = pd.read_csv(csv_path)
    with meta_path.open() as f:
        meta = json.load(f)
    return df, meta


def style_axes(ax, title: str, ylabel: str, xlabel: str = "Tick") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def fig_macro_climate(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(df["tick"], df["macro_co2_ppm"], color="#d62728", linewidth=1.5)
    axes[0].axhline(280, color="grey", linestyle="--", alpha=0.5, label="Pre-industrial")
    axes[0].legend(fontsize=8)
    style_axes(axes[0], "Atmospheric CO₂", "ppm")

    axes[1].plot(df["tick"], df["macro_temperature"], color="#ff7f0e", linewidth=1.5)
    axes[1].axhline(1.5, color="green", linestyle="--", alpha=0.5, label="Paris 1.5 °C")
    axes[1].axhline(2.0, color="orange", linestyle="--", alpha=0.5, label="Paris 2.0 °C")
    axes[1].legend(fontsize=8)
    style_axes(axes[1], "Temperature anomaly", "°C vs. pre-industrial")

    axes[2].plot(df["tick"], df["macro_sea_level_m"], color="#1f77b4", linewidth=1.5)
    style_axes(axes[2], "Sea level rise", "metres above 2000")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_macro_resources(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    pairs = [
        ("macro_fossil_fuels", "Fossil fuel reserves", "fraction remaining"),
        ("macro_minerals", "Mineral reserves", "fraction remaining"),
        ("total_water", "Total freshwater (grid sum)", "units"),
        ("macro_food_index", "Food production index", "0-1"),
    ]
    for ax, (col, title, ylabel) in zip(axes, pairs):
        if col in df.columns:
            ax.plot(df["tick"], df[col], linewidth=1.5)
            style_axes(ax, title, ylabel)
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"missing: {col}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_macro_society(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    pairs = [
        ("macro_population_B", "Population (macro)", "billions"),
        ("macro_gdp_index", "GDP index (vs. 2025)", "index"),
        ("macro_inequality", "Inequality index", "0-1"),
        ("macro_welfare", "Human welfare index", "0-1"),
    ]
    for ax, (col, title, ylabel) in zip(axes, pairs):
        if col in df.columns:
            ax.plot(df["tick"], df[col], linewidth=1.5, color="#2ca02c")
            style_axes(ax, title, ylabel)
        else:
            ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_agents_population(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["tick"], df["population"], color="#1f77b4", linewidth=1.5)
    style_axes(axes[0], "Alive agents", "count")

    if "births_this_interval" in df.columns and "deaths_this_interval" in df.columns:
        axes[1].plot(df["tick"], df["births_this_interval"], label="Births",
                     color="#2ca02c", linewidth=1)
        axes[1].plot(df["tick"], df["deaths_this_interval"], label="Deaths",
                     color="#d62728", linewidth=1)
        axes[1].legend(fontsize=8)
        style_axes(axes[1], "Births & deaths per interval", "agents")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_geopolitics(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if "n_nations" in df.columns:
        axes[0].plot(df["tick"], df["n_nations"], color="#9467bd", linewidth=1.5)
        style_axes(axes[0], "Emergent nations", "count")

    if "n_active_conflicts" in df.columns:
        axes[1].plot(df["tick"], df["n_active_conflicts"], color="#d62728", linewidth=1.5)
        style_axes(axes[1], "Active conflicts", "count")

    if "total_trade_volume" in df.columns:
        axes[2].plot(df["tick"], df["total_trade_volume"], color="#1f77b4", linewidth=1.5)
        style_axes(axes[2], "Total trade volume", "units")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ipcc_validation(df: pd.DataFrame, out: Path) -> None:
    """Overlay simulation CO₂ and temperature with IPCC SSP reference points."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df["tick"], df["macro_co2_ppm"], label="Oikoumene BAU",
                 color="black", linewidth=1.8)
    final_x = df["tick"].iloc[-1]
    for name, ref in IPCC_REFERENCES.items():
        axes[0].scatter(final_x, ref["co2_ppm"], color=ref["color"],
                        s=80, zorder=5, label=name)
    axes[0].legend(fontsize=8, loc="upper left")
    style_axes(axes[0], "CO₂ vs. IPCC SSP scenarios at 2100", "ppm")

    axes[1].plot(df["tick"], df["macro_temperature"], label="Oikoumene BAU",
                 color="black", linewidth=1.8)
    for name, ref in IPCC_REFERENCES.items():
        axes[1].scatter(final_x, ref["temperature"], color=ref["color"],
                        s=80, zorder=5, label=name)
    axes[1].axhspan(2.0, 5.0, alpha=0.1, color="grey", label="Validation band")
    axes[1].legend(fontsize=8, loc="upper left")
    style_axes(axes[1], "Temperature vs. IPCC SSP scenarios at 2100", "°C")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all(run_dir: Path) -> None:
    _require_viz()
    print(f"Loading run: {run_dir.name}")
    df, meta = load_run(run_dir)
    print(f"  rows: {len(df)}  scenario: {meta.get('scenario_name', '?')}  seed: {meta.get('seed', '?')}")

    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    figures = [
        ("macro_climate.png", fig_macro_climate),
        ("macro_resources.png", fig_macro_resources),
        ("macro_society.png", fig_macro_society),
        ("agents_population.png", fig_agents_population),
        ("geopolitics.png", fig_geopolitics),
        ("ipcc_validation.png", fig_ipcc_validation),
    ]

    for name, func in figures:
        path = out_dir / name
        try:
            func(df, path)
            print(f"  + {path.relative_to(REPO_ROOT)}")
        except Exception as exc:
            print(f"  ! {name} failed: {exc}", file=sys.stderr)

    print(f"\nDone. {len(figures)} figures in {out_dir.relative_to(REPO_ROOT)}/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", help="Specific run ID; default = latest")
    parser.add_argument("--list", action="store_true", help="List available runs and exit")
    args = parser.parse_args()

    if args.list:
        runs = list_runs()
        if not runs:
            print(f"No runs found in {LOGS_DIR}")
            return 1
        print(f"Available runs in {LOGS_DIR}:")
        for r in runs:
            print(f"  {r.name}")
        return 0

    if args.run_id:
        run_dir = LOGS_DIR / args.run_id
        if not run_dir.exists():
            print(f"Run not found: {run_dir}", file=sys.stderr)
            return 1
    else:
        run_dir = latest_run()
        if run_dir is None:
            print(f"No runs in {LOGS_DIR}. Run the simulation first.", file=sys.stderr)
            return 1

    generate_all(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
