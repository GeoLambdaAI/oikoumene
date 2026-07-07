"""
Scenario Management — Configures simulation for different starting conditions.

Scenario A: "70,000 Years of Human History" (existing behavior)
Scenario B: "Present Day → Future" (real-world data initialization)

Both scenarios use the same World, Agent, MacroModel, GeopoliticalSystem
classes. The scenario only changes INITIAL CONDITIONS, not simulation logic.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""
    id: str
    name: str
    description: str
    start_date: str
    initial_agents: int = 25
    macro_active_from_start: bool = False
    use_history_timeline: bool = True
    start_year_bp: int = 70000


SCENARIOS = {
    "historical": ScenarioConfig(
        id="historical",
        name="70,000 Years of Human History",
        description="Start as small bands in East Africa. Migrate, settle, "
                    "build civilizations through the Stone Age to the modern era.",
        start_date="70000 BP",
        initial_agents=25,
        macro_active_from_start=False,
        use_history_timeline=True,
        start_year_bp=70000,
    ),
    "present_day": ScenarioConfig(
        id="present_day",
        name="Present Day → Future",
        description=f"Start from real-world data ({date.today().year}). "
                    f"Current population, economy, climate, and conflicts. "
                    f"Simulate the future.",
        start_date=date.today().isoformat(),
        initial_agents=300,
        macro_active_from_start=True,
        use_history_timeline=False,
        start_year_bp=-76,  # ~2026 CE in BP
    ),
}


# Alliance clusters for present-day scenario
NATO_MEMBERS = {
    "USA", "GBR", "FRA", "DEU", "CAN", "ITA", "ESP", "TUR", "POL", "NLD",
    "BEL", "NOR", "DNK", "PRT", "CZE", "GRC", "HUN", "BGR", "ROU", "HRV",
    "ALB", "LTU", "LVA", "EST", "SVK",
}
EU_MEMBERS = {
    "FRA", "DEU", "ITA", "ESP", "NLD", "BEL", "AUT", "SWE", "FIN", "DNK",
    "IRL", "PRT", "GRC", "CZE", "HUN", "BGR", "ROU", "HRV", "SVK", "POL",
    "LTU", "LVA", "EST",
}
BRICS_MEMBERS = {"BRA", "RUS", "IND", "CHN", "ZAF", "EGY", "ETH", "IRN", "ARE", "SAU"}


class ScenarioLoader:
    """Loads scenario data and configures World for selected scenario."""

    def configure_world(self, world, scenario: ScenarioConfig):
        """Apply scenario configuration to an initialized World."""
        if scenario.id == "present_day":
            self._configure_present_day(world, scenario)

    def _configure_present_day(self, world, scenario: ScenarioConfig):
        """Initialize world from real-world data files."""
        print("  Loading present-day data...")

        # 1. Load climate -> set MacroState
        climate = self._load_json("present_day_climate.json", {
            "co2_ppm": 425.0, "temperature_anomaly": 1.3, "sea_level_rise_m": 0.10
        })
        self._init_macro(world.macro, climate)

        # 1b. Anchor the starting temperature to the latest HadCRUT5 observation
        #     (converted to the pre-industrial baseline the macro model uses),
        #     falling back to the JSON value when the dataset is absent.
        from empirical_ingest import observed_temperature_anomaly
        obs = observed_temperature_anomaly()
        if obs is not None:
            obs_year, obs_anom, _ = obs
            world.macro.state.temperature_anomaly = obs_anom
            print(f"    Temperature: anchored to HadCRUT5 {obs_year} "
                  f"= {obs_anom:+.2f}°C vs pre-industrial")
        # Recompute derived macro fields (radiative_forcing, social_tension,
        # food/welfare indices) from the fully-seeded state so they are correct
        # from tick 0 rather than the class defaults until the first macro step.
        world.macro._compute_derived()

        # 2. Load resources -> override ResourceMap
        self._init_resources(world)

        # 3. Load population + countries -> spawn agents
        countries = self._load_json("present_day_countries.json", [])
        pop_grid = self._load_npy("present_day_population.npy")
        self._spawn_agents(world, pop_grid, countries, scenario.initial_agents)

        # 4. Create nations from countries
        self._init_nations(world, countries)

        # 5. Load conflicts — prefer real UCDP-GED anchoring; fall back to the
        #    shipped illustrative set when the (gitignored, optional) empirical
        #    dataset is absent.
        from empirical_ingest import load_present_day_conflicts
        conflicts = load_present_day_conflicts()
        if conflicts:
            print(f"    Conflicts: {len(conflicts)} zones anchored from UCDP-GED")
        else:
            conflicts = self._load_json("present_day_conflicts.json", [])
        self._init_conflicts(world, conflicts)

        print(f"  Present-day init: {len(world.agents)} agents, "
              f"{len(world.geopolitics.nations)} nations, "
              f"{len(world.geopolitics.active_conflicts)} conflicts")

    def _init_macro(self, macro, climate: dict):
        """Set MacroState from real climate data."""
        s = macro.state
        s.co2_ppm = climate.get("co2_ppm", 425.0)
        s.temperature_anomaly = climate.get("temperature_anomaly", 1.3)
        s.sea_level_rise_m = climate.get("sea_level_rise_m", 0.10)
        s.year = float(date.today().year) + date.today().month / 12.0
        s.global_population_billions = 8.1
        s.fossil_fuels = 0.82
        s.minerals_global = 0.88
        s.renewable_fraction = 0.20
        s.technology_level = 1.0
        s.inequality_index = 0.62
        s.persistent_pollution = 0.35
        s.global_gdp_index = 1.0

    def _init_resources(self, world):
        """Override ResourceMap with real-world resource estimates."""
        res_path = DATA_DIR / "present_day_resources.npz"
        if not res_path.exists():
            return

        data = np.load(res_path)
        # Map the downloaded grids onto the world's ResourceMap
        # They may have slightly different dimensions due to rounding
        for resource_name in ["food", "minerals", "freshwater"]:
            if resource_name in data:
                src = data[resource_name]
                dst = getattr(world.resources, resource_name if resource_name != "freshwater" else "water", None)
                if dst is not None:
                    # Copy what fits
                    rows = min(src.shape[0], dst.shape[0])
                    cols = min(src.shape[1], dst.shape[1])
                    dst[:rows, :cols] = src[:rows, :cols]

        if "fossil_fuels" in data:
            src = data["fossil_fuels"]
            rows = min(src.shape[0], world.resources.minerals.shape[0])
            cols = min(src.shape[1], world.resources.minerals.shape[1])
            # Add fossil fuel deposits to minerals as a proxy
            world.resources.minerals[:rows, :cols] += src[:rows, :cols] * 0.3

    def _spawn_agents(self, world, pop_grid: Optional[np.ndarray],
                      countries: list, n_agents: int):
        """Spawn agents proportional to population distribution."""
        from agents import Agent
        from earth import is_land

        if pop_grid is None or pop_grid.sum() == 0:
            # Fallback: use existing spawn logic
            from earth import find_land_spawn_points
            for lat, lng in find_land_spawn_points(n_agents, world.seed):
                agent = Agent(lat, lng, rng=world.rng)
                agent.energy = 80 + world.rng.random() * 20
                agent.wealth = 30
                world.agents.append(agent)
            return

        # Normalize population grid to probability distribution
        pop_flat = pop_grid.flatten()
        total = pop_flat.sum()
        if total <= 0:
            return
        prob = pop_flat / total

        # Sample agent locations weighted by population
        rows, cols = pop_grid.shape
        indices = world.rng.choice(len(pop_flat), size=n_agents, p=prob)

        # Build country lookup: iso3 -> country_data
        country_lookup = {c["iso3"]: c for c in countries}

        for idx in indices:
            r, c = divmod(idx, cols)
            lat = 75.0 - (r + world.rng.random()) * 2.0
            lng = -180.0 + (c + world.rng.random()) * 2.0

            if not is_land(lat, lng):
                # Jitter until on land
                for _ in range(5):
                    lat += world.rng.normal(0, 1)
                    lng += world.rng.normal(0, 1)
                    if is_land(lat, lng):
                        break

            agent = Agent(lat, lng, rng=world.rng)

            # Find nearest country and set stats from data
            nearest_country = self._find_nearest_country(lat, lng, countries)
            if nearest_country:
                cd = nearest_country
                gdp_pc = cd.get("gdp_per_capita_ppp", 10000)
                agent.wealth = float(np.clip(gdp_pc / 500, 5, 200))
                agent.energy = float(np.clip(50 + cd.get("life_expectancy", 65) / 2, 60, 95))

                # Traits from country data
                agent.traits["intelligence"] = float(np.clip(
                    cd.get("literacy_rate", 80) / 120 * 0.5 +
                    cd.get("internet_pct", 50) / 120 * 0.3 +
                    cd.get("research_pct_gdp", 0.5) / 3 * 0.2, 0.1, 0.95))
                agent.traits["cooperation"] = float(np.clip(
                    1.0 - cd.get("gini_index", 40) / 80, 0.1, 0.95))
                agent.traits["ambition"] = float(np.clip(
                    cd.get("trade_pct_gdp", 50) / 120, 0.1, 0.95))

                # Skills from country profile
                agent.skills.skills["research"] = float(np.clip(
                    cd.get("research_pct_gdp", 0.5) / 3, 0.01, 0.8))
                agent.skills.skills["trading"] = float(np.clip(
                    cd.get("trade_pct_gdp", 50) / 100, 0.01, 0.8))
                agent.skills.skills["farming"] = float(np.clip(
                    cd.get("agriculture_pct_gdp", 10) / 30, 0.01, 0.6))

                # Add noise ±15%
                for trait in agent.traits:
                    agent.traits[trait] *= (0.85 + world.rng.random() * 0.30)
                    agent.traits[trait] = float(np.clip(agent.traits[trait], 0.05, 0.95))

            world.agents.append(agent)

    def _init_nations(self, world, countries: list):
        """Create NationStates from real country data."""
        from geopolitics import NationState

        for i, cd in enumerate(countries):
            if cd["population"] < 1_000_000:
                continue  # Skip very small countries

            wealth_bn = cd.get("gdp_usd", 0) / 1e9
            mil_spending = cd.get("military_pct_gdp", 2.0) / 100
            nation = NationState(
                id=i + 1,
                name=cd.get("name", cd["iso3"]),
                settlement_ids=[],
                center_lat=cd["lat"],
                center_lng=cd["lng"],
                population=cd["population"],
                total_wealth=wealth_bn,
                # Warm-start the deterrence/parity dimension: seed military mass
                # from military spending × GDP instead of leaving it at 0 (which
                # made every dyad look militarily identical and killed the
                # power-asymmetry signal). This shifts conflict risk toward
                # genuine near-peer rivalries, per Bremer (1992) parity effect.
                total_military=mil_spending * wealth_bn,
                technology_level=1.0 + cd.get("technology_level", 0.5),
                military_spending=mil_spending,
                carbon_policy=(cd.get("renewable_pct", 15) - 50) / 50,
                trade_openness=float(np.clip(cd.get("trade_pct_gdp", 50) / 100, 0.1, 1.0)),
                research_spending=cd.get("research_pct_gdp", 0.5) / 100,
                seeded=True,
            )

            # These are scenario-anchored real countries: they start with no
            # member settlements and are defined by their centre coordinates and
            # seeded aggregate stats. Marking them `seeded=True` keeps the
            # geopolitical system from pruning them as "empty" and from zeroing
            # their seeded population/wealth. Emergent settlements that grow near
            # a nation centre are still absorbed into it by proximity, at which
            # point its aggregates become agent-driven.
            world.geopolitics.nations.append(nation)
            world.geopolitics.relation_graph.add_node(nation.id)
            world.geopolitics.trade_graph.add_node(nation.id)

        # Initialize alliance clusters
        nation_by_iso = {}
        for i, cd in enumerate(countries):
            if cd["population"] >= 1_000_000:
                nation_by_iso[cd["iso3"]] = i + 1

        # Advance the emergent-nation id counter past every seeded id so a nation
        # that later forms organically cannot collide with a seeded country.
        from geopolitics import GeopoliticalSystem
        if world.geopolitics.nations:
            GeopoliticalSystem._next_nation_id = max(
                GeopoliticalSystem._next_nation_id,
                max(n.id for n in world.geopolitics.nations),
            )

        self._create_alliance_cluster(world, nation_by_iso, NATO_MEMBERS, "NATO")
        self._create_alliance_cluster(world, nation_by_iso, EU_MEMBERS, "EU")
        self._create_alliance_cluster(world, nation_by_iso, BRICS_MEMBERS, "BRICS")

    def _create_alliance_cluster(self, world, nation_by_iso: dict,
                                  members: set, name: str):
        """Create alliance edges between member nations."""
        member_ids = [nation_by_iso[iso] for iso in members if iso in nation_by_iso]
        for i, na_id in enumerate(member_ids):
            for nb_id in member_ids[i+1:]:
                world.geopolitics.relation_graph.add_edge(na_id, nb_id, weight=0.4)
                world.geopolitics.relation_graph.add_edge(nb_id, na_id, weight=0.4)
                # Find nations and update their alliance sets
                for n in world.geopolitics.nations:
                    if n.id == na_id:
                        n.alliances.add(nb_id)
                    elif n.id == nb_id:
                        n.alliances.add(na_id)

    def _init_conflicts(self, world, conflicts: list):
        """Initialize active conflicts from data."""
        for c in conflicts:
            world.geopolitics.active_conflicts.append({
                "nations": [],
                "nation_names": c.get("parties", []),
                "lat": c["lat"],
                "lng": c["lng"],
                "radius": c.get("radius_deg", 3.0),
                "intensity": c.get("intensity", 0.5),
                "duration": 0,
                "cause": c.get("name", "ongoing"),
            })

    def _find_nearest_country(self, lat: float, lng: float,
                               countries: list) -> Optional[dict]:
        """Find nearest country to a lat/lng coordinate."""
        best_dist = float('inf')
        best = None
        for c in countries:
            d = (lat - c["lat"])**2 + (lng - c["lng"])**2
            if d < best_dist:
                best_dist = d
                best = c
        return best

    def _load_json(self, filename: str, default):
        """Load JSON from data dir with fallback."""
        path = DATA_DIR / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        print(f"    WARNING: {filename} not found, using defaults")
        return default

    def _load_npy(self, filename: str) -> Optional[np.ndarray]:
        """Load numpy array from data dir."""
        path = DATA_DIR / filename
        if path.exists():
            return np.load(path)
        print(f"    WARNING: {filename} not found")
        return None
