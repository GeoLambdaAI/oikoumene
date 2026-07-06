"""
Scientific Simulation Logger — Persistent time-series recording for analysis.

Records comprehensive state snapshots at configurable intervals to CSV + JSON.
Every row captures a complete cross-section of the simulation at one point in time:
macro state, geopolitical summary, population statistics, resource levels, and events.

Output files per run (each run gets its own subdirectory logs/{run_id}/):
- {run_id}/timeseries.csv   — One row per logging interval, all metrics as columns
- {run_id}/metadata.json    — Run config, seed, scenario, timestamps, column descriptions
- {run_id}/snapshots/       — Full JSON state dumps at checkpoint intervals (optional)

Designed for:
- Post-hoc analysis in pandas, R, or Excel
- Run comparison (overlay two CSVs)
- Reproducibility (metadata includes seed + all config)
"""

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np


LOGS_DIR = Path(__file__).parent / "logs"


@dataclass
class LoggerConfig:
    """Configuration for simulation logging."""
    enabled: bool = True
    log_interval_ticks: int = 1        # Log every N ticks
    snapshot_interval_ticks: int = 100  # Full JSON snapshot every N ticks (0=disabled)
    log_dir: str = str(LOGS_DIR)
    run_id: Optional[str] = None       # Auto-generated if None
    flush_interval: int = 10           # Flush CSV to disk every N writes


# All columns in the time-series CSV, grouped by domain.
# Each tuple: (column_name, description, unit)
COLUMN_SCHEMA = [
    # --- Time ---
    ("tick", "Simulation tick counter", ""),
    ("year_display", "Human-readable year (e.g., '68,000 BCE' or '2026 CE')", ""),
    ("year_bp", "Year before present (1950 CE baseline)", "years BP"),
    ("year_ce", "Year in CE (negative = BCE)", "years CE"),
    ("era_name", "Current historical era", ""),
    ("era_time_scale", "Simulated years per tick in current era", "years/tick"),

    # --- Population ---
    ("population", "Number of alive agents", "agents"),
    ("total_born", "Cumulative agents ever created", "agents"),
    ("births_this_interval", "Births since last log", "agents"),
    ("deaths_this_interval", "Deaths since last log", "agents"),
    ("avg_age", "Mean age of alive agents", "ticks"),
    ("max_generation", "Highest generation number alive", ""),

    # --- Agent vitals (population averages) ---
    ("avg_energy", "Mean energy of alive agents", "0-100"),
    ("avg_health", "Mean health of alive agents", "0-100"),
    ("avg_wealth", "Mean wealth of alive agents", "units"),
    ("avg_happiness", "Mean happiness of alive agents", "0-100"),
    ("median_wealth", "Median wealth (inequality indicator)", "units"),
    ("wealth_gini", "Gini coefficient of wealth distribution", "0-1"),
    ("std_wealth", "Standard deviation of wealth", "units"),

    # --- Agent traits (population means) ---
    ("avg_intelligence", "Mean intelligence trait", "0-1"),
    ("avg_cooperation", "Mean cooperation trait", "0-1"),
    ("avg_ambition", "Mean ambition trait", "0-1"),
    ("avg_curiosity", "Mean curiosity trait", "0-1"),

    # --- Agent actions (distribution) ---
    ("pct_eating", "% of agents currently eating", "%"),
    ("pct_working", "% of agents currently working", "%"),
    ("pct_trading", "% of agents currently trading", "%"),
    ("pct_exploring", "% of agents currently exploring", "%"),
    ("pct_socializing", "% of agents currently socializing", "%"),
    ("pct_reproducing", "% of agents currently reproducing", "%"),
    ("pct_researching", "% of agents currently researching", "%"),
    ("pct_governing", "% of agents currently governing", "%"),
    ("pct_migrating", "% of agents currently migrating", "%"),

    # --- Economy ---
    ("n_businesses", "Number of active businesses", ""),
    ("n_settlements", "Number of settlements", ""),

    # --- Macro state (Club of Rome ODE) ---
    ("macro_co2_ppm", "Atmospheric CO2 concentration", "ppm"),
    ("macro_temperature", "Global temperature anomaly vs pre-industrial", "deg C"),
    ("macro_sea_level_m", "Sea level rise above 2000 baseline", "m"),
    ("macro_fossil_fuels", "Fossil fuel reserves remaining", "0-1"),
    ("macro_minerals", "Mineral reserves remaining", "0-1"),
    ("macro_pollution", "Persistent pollution index", "0-1"),
    ("macro_population_B", "Macro model population", "billions"),
    ("macro_gdp_index", "GDP relative to 2025", "index"),
    ("macro_inequality", "Global inequality index", "0-1"),
    ("macro_social_tension", "Social tension (Earth4All)", "0-1"),
    ("macro_technology", "Technology level multiplier", "x"),
    ("macro_renewable_frac", "Renewable energy fraction", "0-1"),
    ("macro_food_index", "Food production index", "0-1"),
    ("macro_welfare", "Human welfare index", "0-1"),

    # --- Geopolitics ---
    ("n_nations", "Number of emergent nations", ""),
    ("n_active_conflicts", "Number of active armed conflicts", ""),
    ("total_trade_volume", "Sum of all bilateral trade", "units"),
    ("avg_diplomatic_trust", "Mean inter-nation trust", "-1 to 1"),
    ("conflict_intensity", "Mean intensity of active conflicts", "0-1"),

    # --- Resources (grid totals) ---
    ("total_food", "Total food across all grid cells", "units"),
    ("total_minerals", "Total minerals across all grid cells", "units"),
    ("total_wood", "Total wood across all grid cells", "units"),
    ("total_water", "Total freshwater across all grid cells", "units"),

    # --- Spatial ---
    ("agent_lat_mean", "Mean latitude of all agents", "degrees"),
    ("agent_lng_mean", "Mean longitude of all agents", "degrees"),
    ("agent_spread_deg", "Std dev of agent positions (geographic spread)", "degrees"),

    # --- JEPA world model ---
    ("world_model_train_steps", "Shared world model training steps", ""),
    ("world_model_buffer_size", "Experience buffer entries", ""),

    # --- Performance ---
    ("tick_duration_ms", "Wall-clock time for this tick", "ms"),
]


class SimulationLogger:
    """
    Persistent scientific logger for simulation runs.

    Usage:
        logger = SimulationLogger(config)
        logger.start_run(world)  # writes metadata, opens CSV

        # In World.step():
        logger.log_tick(world, stats, tick_duration_ms)

        logger.end_run()  # flushes, closes files
    """

    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or LoggerConfig()
        self._csv_file = None
        self._csv_writer = None
        self._run_id: str = ""
        self._run_dir: Path = Path(".")
        self._write_count: int = 0
        self._last_total_born: int = 0
        self._last_population: int = 0
        self._tick_start_time: float = 0.0
        self.is_running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_run(self, world) -> str:
        """Initialize a new logging run. Returns the run_id."""
        if not self.config.enabled:
            return ""

        # Generate run ID
        self._run_id = self.config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = Path(self.config.log_dir) / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata = {
            "run_id": self._run_id,
            "started_at": datetime.now().isoformat(),
            "scenario_id": world.scenario.id,
            "scenario_name": world.scenario.name,
            "seed": world.seed,
            "cell_size_deg": world.cell_size_deg,
            "start_year_bp": world.history.year_bp,
            "initial_agents": len([a for a in world.agents if a.alive]),
            "log_interval_ticks": self.config.log_interval_ticks,
            "snapshot_interval_ticks": self.config.snapshot_interval_ticks,
            "columns": [
                {"name": c[0], "description": c[1], "unit": c[2]}
                for c in COLUMN_SCHEMA
            ],
        }
        with open(self._run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Open CSV
        csv_path = self._run_dir / "timeseries.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([c[0] for c in COLUMN_SCHEMA])
        self._write_count = 0

        self.is_running = True
        return self._run_id

    def end_run(self):
        """Finalize and close the run. Idempotent: safe to call when no run is
        active (e.g. on repeated resets) — it simply no-ops."""
        if not self.is_running and self._csv_file is None:
            return
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
        self.is_running = False

        # Update metadata with end info
        if self._run_dir is None:
            return
        meta_path = self._run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["ended_at"] = datetime.now().isoformat()
            meta["total_rows"] = self._write_count
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    def tick_start(self):
        """Call at the beginning of World.step() to measure tick duration."""
        self._tick_start_time = time.perf_counter()

    def log_tick(self, world, stats: dict):
        """
        Log one row if the current tick matches the logging interval.
        Call at the end of World.step().
        """
        if not self.config.enabled or not self.is_running:
            return
        if world.tick % self.config.log_interval_ticks != 0:
            return

        tick_ms = (time.perf_counter() - self._tick_start_time) * 1000

        row = self._build_row(world, stats, tick_ms)
        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._write_count += 1

            if self._write_count % self.config.flush_interval == 0:
                self._csv_file.flush()

        # Optional full snapshot
        if (self.config.snapshot_interval_ticks > 0 and
                world.tick % self.config.snapshot_interval_ticks == 0):
            self._write_snapshot(world)

    # ------------------------------------------------------------------
    # Row builder — extracts all metrics from world state
    # ------------------------------------------------------------------

    def _build_row(self, world, stats: dict, tick_ms: float) -> list:
        """Build one CSV row with all metrics."""
        alive = [a for a in world.agents if a.alive]
        n_alive = len(alive)

        # History
        hist = stats.get("history", {})
        year_bp = hist.get("year_bp", world.history.year_bp)
        year_ce = hist.get("year_ce", 1950 - year_bp)

        # Births/deaths since last log
        total_born = stats.get("total_born", len(world.agents))
        births = total_born - self._last_total_born
        deaths = births - (n_alive - self._last_population)
        self._last_total_born = total_born
        self._last_population = n_alive

        # Agent vitals
        if n_alive > 0:
            energies = np.array([a.energy for a in alive])
            healths = np.array([a.health for a in alive])
            wealths = np.array([a.wealth for a in alive])
            happinesses = np.array([a.happiness for a in alive])
            ages = np.array([a.age for a in alive])
            lats = np.array([a.lat for a in alive])
            lngs = np.array([a.lng for a in alive])

            avg_energy = float(np.mean(energies))
            avg_health = float(np.mean(healths))
            avg_wealth = float(np.mean(wealths))
            avg_happiness = float(np.mean(happinesses))
            avg_age = float(np.mean(ages))
            median_wealth = float(np.median(wealths))
            std_wealth = float(np.std(wealths))

            # Gini coefficient
            sorted_w = np.sort(wealths)
            n = len(sorted_w)
            if n > 1 and sorted_w.sum() > 0:
                index = np.arange(1, n + 1)
                wealth_gini = float((2 * np.sum(index * sorted_w) - (n + 1) * np.sum(sorted_w)) /
                                    (n * np.sum(sorted_w)))
            else:
                wealth_gini = 0.0

            # Trait averages
            avg_intel = float(np.mean([a.traits["intelligence"] for a in alive]))
            avg_coop = float(np.mean([a.traits["cooperation"] for a in alive]))
            avg_ambition = float(np.mean([a.traits["ambition"] for a in alive]))
            avg_curiosity = float(np.mean([a.traits["curiosity"] for a in alive]))

            # Action distribution
            actions = {}
            for a in alive:
                act = a.current_action
                actions[act] = actions.get(act, 0) + 1
            pct = lambda act: round(actions.get(act, 0) / n_alive * 100, 1)

            # Spatial spread
            lat_mean = float(np.mean(lats))
            lng_mean = float(np.mean(lngs))
            spread = float(np.sqrt(np.std(lats)**2 + np.std(lngs)**2))

            max_gen = int(max(a.generation for a in alive))
        else:
            avg_energy = avg_health = avg_wealth = avg_happiness = avg_age = 0
            median_wealth = std_wealth = wealth_gini = 0
            avg_intel = avg_coop = avg_ambition = avg_curiosity = 0
            pct = lambda act: 0
            lat_mean = lng_mean = spread = 0
            max_gen = 0

        # Macro
        macro = stats.get("macro", {})

        # Geopolitics
        geo = stats.get("geopolitics", {})

        # Resources
        total_food = float(world.resources.food.sum())
        total_minerals = float(world.resources.minerals.sum())
        total_wood = float(world.resources.wood.sum())
        total_water = float(world.resources.water.sum())

        # World model
        wm = world.shared_world_model.get_understanding()

        return [
            # Time
            world.tick,
            hist.get("year_display", ""),
            round(year_bp, 1),
            round(year_ce, 1),
            hist.get("era_name", ""),
            hist.get("time_scale", 0),

            # Population
            n_alive,
            total_born,
            max(0, births),
            max(0, deaths),
            round(avg_age, 1),
            max_gen,

            # Vitals
            round(avg_energy, 2),
            round(avg_health, 2),
            round(avg_wealth, 2),
            round(avg_happiness, 2),
            round(median_wealth, 2),
            round(wealth_gini, 4),
            round(std_wealth, 2),

            # Traits
            round(avg_intel, 3),
            round(avg_coop, 3),
            round(avg_ambition, 3),
            round(avg_curiosity, 3),

            # Actions
            pct("eat"), pct("work"), pct("trade"), pct("explore"),
            pct("socialize"), pct("reproduce"), pct("research"),
            pct("govern"), pct("migrate"),

            # Economy
            stats.get("businesses", 0),
            stats.get("settlements", 0),

            # Macro
            macro.get("co2_ppm", 0),
            macro.get("temperature", 0),
            macro.get("sea_level_m", 0),
            macro.get("fossil_fuels", 0),
            macro.get("minerals", 0),
            macro.get("pollution", 0),
            macro.get("population_B", 0),
            macro.get("gdp_index", 0),
            macro.get("inequality", 0),
            macro.get("social_tension", 0),
            macro.get("technology", 0),
            macro.get("renewable_frac", 0),
            macro.get("food_index", 0),
            macro.get("welfare", 0),

            # Geopolitics
            geo.get("nations", 0),
            geo.get("active_conflicts", 0),
            round(geo.get("trade_volume", 0), 2),
            round(geo.get("avg_trust", 0), 3),
            round(geo.get("conflict_intensity", 0), 3),

            # Resources
            round(total_food, 1),
            round(total_minerals, 1),
            round(total_wood, 1),
            round(total_water, 1),

            # Spatial
            round(lat_mean, 2),
            round(lng_mean, 2),
            round(spread, 2),

            # JEPA
            wm.get("train_steps", 0),
            wm.get("buffer_size", 0),

            # Performance
            round(tick_ms, 1),
        ]

    def _write_snapshot(self, world):
        """Write full JSON state snapshot for checkpoint/analysis."""
        snap_dir = self._run_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)

        state = world.get_full_state()
        # Remove heavy fields that are redundant with timeseries
        state.pop("stats_history", None)
        state.pop("dialogues", None)

        path = snap_dir / f"tick_{world.tick:08d}.json"
        with open(path, "w") as f:
            json.dump(state, f, default=_json_default)

    # ------------------------------------------------------------------
    # Status & file access
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "is_running": self.is_running,
            "run_id": self._run_id,
            "rows_written": self._write_count,
            "log_interval": self.config.log_interval_ticks,
            "log_dir": str(self._run_dir) if self.is_running else "",
        }

    def get_csv_path(self) -> Optional[str]:
        """Return path to current CSV file for download."""
        if self.is_running and self._run_dir:
            path = self._run_dir / "timeseries.csv"
            if path.exists():
                return str(path)
        return None

    def list_runs(self) -> list[dict]:
        """List all completed runs."""
        log_dir = Path(self.config.log_dir)
        if not log_dir.exists():
            return []
        runs = []
        for d in sorted(log_dir.iterdir()):
            if d.is_dir():
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["csv_exists"] = (d / "timeseries.csv").exists()
                    runs.append(meta)
        return runs


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    return str(obj)
