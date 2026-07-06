"""
Macro World Dynamics — System Dynamics ODE layer.

Inspired by:
- World3 / PyWorld3 (Meadows et al., "Limits to Growth", 1972/2004)
- Earth4All (Club of Rome / Dixson-Decleve et al., 2022) — social tension model
- Simple energy balance model (IPCC AR6 WG1, Chapter 7)
- Friedlingstein et al. (2024) — Global Carbon Budget

This does NOT replace agents. It provides the global context that agents
experience: rising temperatures, depleting resources, accumulating pollution.
Agent actions feed back into this model via the bridge.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class MacroState:
    """
    Global state vector — updated each simulation step.

    All values are physically meaningful with documented units.
    Initial conditions represent approximately 2025 baseline values.
    """
    year: float = 2025.0

    # --- Climate ---
    co2_ppm: float = 425.0                  # Atmospheric CO2 (ppm), source: NOAA 2024
    temperature_anomaly: float = 1.3        # deg C above pre-industrial, source: NASA GISS
    deep_ocean_temp: float = 0.3            # Deep ocean temperature anomaly (deg C)
    sea_level_rise_m: float = 0.21          # Meters above 2000 baseline, source: IPCC AR6

    # --- Resources (fractions remaining, 0-1) ---
    fossil_fuels: float = 0.85              # Oil + gas + coal reserves fraction
    minerals_global: float = 0.90           # Metals, rare earths fraction
    freshwater_stress: float = 0.25         # Fraction of regions under stress

    # --- Pollution ---
    persistent_pollution: float = 0.30      # Normalized index (0-1)
    ocean_acidification: float = 0.15       # pH drop proxy (0-1)

    # --- Socioeconomic ---
    global_population_billions: float = 8.1  # Source: UN WPP 2024
    global_gdp_index: float = 1.0           # Relative to 2025 = 1.0
    inequality_index: float = 0.65          # Gini-like (0=equal, 1=extreme)
    social_tension: float = 0.25            # Earth4All-inspired (0-1)
    technology_level: float = 1.0           # Multiplier, grows over time
    renewable_fraction: float = 0.15        # Fraction of energy from renewables

    # --- Derived (computed each step, not ODE state) ---
    food_production_index: float = 1.0      # Relative to 2025
    human_welfare_index: float = 0.65       # Composite (0-1)
    radiative_forcing: float = 2.7          # W/m^2, computed from CO2


# State vector indices for ODE system
_IDX = {
    "co2": 0,
    "temp": 1,
    "deep_temp": 2,
    "fossil": 3,
    "minerals": 4,
    "pollution": 5,
    "ocean_acid": 6,
    "population": 7,
    "gdp": 8,
    "inequality": 9,
    "tech": 10,
    "renewable": 11,
    "freshwater": 12,
    "sea_level": 13,
}
_N_STATES = len(_IDX)


class MacroModel:
    """
    System dynamics model governing global-scale variables.

    The ODE system evolves continuously. Each macro tick we advance
    the ODEs by dt_years and update the MacroState.

    Key feedback loops (from Limits to Growth / Earth4All):
    1. Population -> resource demand -> depletion -> food decline -> population
    2. Industrial output -> pollution -> agriculture damage -> food -> population
    3. CO2 -> temperature -> agriculture -> food -> social tension
    4. Technology -> efficiency -> slower depletion (balancing loop)
    5. Inequality -> social tension -> governance instability -> conflict

    References in each equation comment.
    """

    # --- Physical constants ---
    CO2_PREINDUSTRIAL = 280.0       # ppm, IPCC AR6
    FORCING_COEFF = 5.35            # W/m^2, Myhre et al. 1998
    CLIMATE_SENSITIVITY = 3.0       # deg C per 2xCO2, IPCC AR6 WG1 Table 7.SM.1
    OCEAN_HEAT_CAPACITY = 7.0       # W*yr/m^2/degC, Held et al. 2010 (lower end)
    # CLIMATE_FEEDBACK is chosen so the equilibrium response to 2xCO2 equals
    # CLIMATE_SENSITIVITY: F_2x = 5.35 * ln(2) = 3.708 W/m^2; lambda = F_2x / ECS.
    # FIX (v0.2): Previous value 1.1 produced an emergent ECS of 3.37 deg C,
    # at the upper end of the IPCC likely range and inconsistent with the
    # declared CLIMATE_SENSITIVITY constant.
    CLIMATE_FEEDBACK = 1.236        # W/m^2/degC, calibrated to ECS=3.0 (IPCC AR6 best)
    DEEP_OCEAN_COUPLING = 0.7       # W/m^2/degC, Gregory 2000
    DEEP_OCEAN_CAPACITY = 100.0     # W*yr/m^2/degC (much larger heat reservoir)

    # --- Carbon cycle ---
    # Decadal mean fraction of emissions taken up by combined land+ocean sinks.
    # Friedlingstein et al. 2024 Global Carbon Budget: 2014-2023 mean is ~57%
    # absorbed (43% airborne fraction). 0.50 is a conservative central value
    # accounting for inter-annual variability and the projected decline of
    # sinks with continued warming. FIX (v0.2): previous 0.44 was at the
    # lower end of the interannual range (single-year airborne fraction), not
    # the appropriate decadal-mean value for a long-horizon model.
    NATURAL_ABSORPTION_RATE = 0.50  # Fraction absorbed, Friedlingstein 2024 decadal
    ABSORPTION_TEMP_SENSITIVITY = 0.06  # Reduction per degC warming (weakening sink)

    # --- Resource depletion ---
    FOSSIL_HALF_DEPLETION = 80.0    # Years at current rate, World3 calibration
    MINERAL_HALF_DEPLETION = 120.0  # Years at current rate

    # --- Socioeconomic ---
    BASE_EMISSION_RATE = 42.0       # GtCO2/yr baseline 2025, source: GCP 2024
    # Pre-industrial land-use / early-agriculture CO2 source (deforestation; the
    # "early Anthropocene", Ruddiman 2003). Small relative to industrial fossil
    # emissions; present even before industrialisation and faded out as industry
    # takes over (see _ode_system). GtCO2/yr at present-day population scale.
    LAND_USE_EMISSION_RATE = 3.0
    # Conversion: 1 ppm CO2 corresponds to ~7.81 GtCO2 in atmosphere
    # (derived from atmospheric mass / molar mass ratios; equivalently 2.13 GtC).
    # IPCC AR6 WG1 Annex VII. -> 1 GtCO2 = 0.1280 ppm.
    PPM_PER_GTCO2 = 0.1280          # ppm per GtCO2, IPCC AR6 WG1 Annex VII
    POP_GROWTH_BASE = 0.009         # Base population growth rate, UN WPP
    TECH_GROWTH_BASE = 0.015        # Base tech growth per year

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.state = MacroState()
        self.dt_years = self.config.get("dt_years", 1.0 / 12.0)  # ~1 month per step
        self._last_feedback: dict = {}

    def _state_to_vector(self) -> np.ndarray:
        """Pack MacroState into ODE state vector."""
        y = np.zeros(_N_STATES)
        y[_IDX["co2"]] = self.state.co2_ppm
        y[_IDX["temp"]] = self.state.temperature_anomaly
        y[_IDX["deep_temp"]] = self.state.deep_ocean_temp
        y[_IDX["fossil"]] = self.state.fossil_fuels
        y[_IDX["minerals"]] = self.state.minerals_global
        y[_IDX["pollution"]] = self.state.persistent_pollution
        y[_IDX["ocean_acid"]] = self.state.ocean_acidification
        y[_IDX["population"]] = self.state.global_population_billions
        y[_IDX["gdp"]] = self.state.global_gdp_index
        y[_IDX["inequality"]] = self.state.inequality_index
        y[_IDX["tech"]] = self.state.technology_level
        y[_IDX["renewable"]] = self.state.renewable_fraction
        y[_IDX["freshwater"]] = self.state.freshwater_stress
        y[_IDX["sea_level"]] = self.state.sea_level_rise_m
        return y

    def _vector_to_state(self, y: np.ndarray):
        """Unpack ODE state vector into MacroState."""
        # Physically-motivated bounds. The lower bounds must sit BELOW plausible
        # pre-industrial values so the paleo->macro handoff (which seeds a
        # negative Little-Ice-Age anomaly ~-0.3 deg C and ~283 ppm CO2) is not
        # snapped upward on the first ODE step, which would reintroduce exactly
        # the discontinuity the handoff exists to prevent. Glacial CO2 minimum
        # ~170 ppm; Last Glacial Maximum anomaly ~-6 deg C. Upper bounds guard
        # against numerical blow-up (NaN/inf) from a diverging integration.
        self.state.co2_ppm = float(np.clip(y[_IDX["co2"]], 150.0, 5000.0))
        self.state.temperature_anomaly = float(np.clip(y[_IDX["temp"]], -10.0, 30.0))
        self.state.deep_ocean_temp = float(np.clip(y[_IDX["deep_temp"]], -5.0, 20.0))
        self.state.fossil_fuels = np.clip(y[_IDX["fossil"]], 0.0, 1.0)
        self.state.minerals_global = np.clip(y[_IDX["minerals"]], 0.0, 1.0)
        self.state.persistent_pollution = np.clip(y[_IDX["pollution"]], 0.0, 1.0)
        self.state.ocean_acidification = np.clip(y[_IDX["ocean_acid"]], 0.0, 1.0)
        self.state.global_population_billions = max(0.1, y[_IDX["population"]])
        self.state.global_gdp_index = max(0.01, y[_IDX["gdp"]])
        self.state.inequality_index = np.clip(y[_IDX["inequality"]], 0.0, 1.0)
        self.state.technology_level = max(0.5, y[_IDX["tech"]])
        self.state.renewable_fraction = np.clip(y[_IDX["renewable"]], 0.0, 1.0)
        self.state.freshwater_stress = np.clip(y[_IDX["freshwater"]], 0.0, 1.0)
        self.state.sea_level_rise_m = max(0.0, y[_IDX["sea_level"]])

        # Compute derived quantities
        self._compute_derived()

    def _compute_derived(self):
        """Compute derived state variables (not part of ODE)."""
        s = self.state

        # Radiative forcing from CO2: F = 5.35 * ln(CO2/CO2_0), Myhre et al. 1998
        s.radiative_forcing = self.FORCING_COEFF * np.log(
            max(s.co2_ppm, 280.0) / self.CO2_PREINDUSTRIAL
        )

        # Food production index: affected by temperature, pollution, water
        # Schlenker & Roberts 2009: crop yields decline above optimal temperature
        temp_effect = max(0.3, 1.0 - 0.12 * max(0, s.temperature_anomaly - 1.5) ** 1.5)
        pollution_effect = max(0.5, 1.0 - 0.3 * s.persistent_pollution)
        water_effect = max(0.4, 1.0 - 0.5 * s.freshwater_stress)
        s.food_production_index = temp_effect * pollution_effect * water_effect

        # Human welfare index: composite HDI-like measure
        # Inspired by Earth4All wellbeing indicator
        food_component = min(1.0, s.food_production_index)
        wealth_component = min(1.0, s.global_gdp_index / (s.global_population_billions / 8.0))
        equality_component = 1.0 - s.inequality_index
        env_component = max(0, 1.0 - s.persistent_pollution - 0.1 * s.temperature_anomaly)
        s.human_welfare_index = np.clip(
            0.3 * food_component + 0.25 * wealth_component +
            0.25 * equality_component + 0.2 * env_component,
            0.0, 1.0
        )

        # Social tension: Earth4All formulation
        # Dixson-Decleve et al. 2022: tension rises with inequality, food insecurity,
        # environmental degradation, and unmet expectations
        food_insecurity = max(0, 1.0 - s.food_production_index)
        env_stress = s.persistent_pollution + 0.1 * max(0, s.temperature_anomaly - 1.5)
        expectation_gap = max(0, 0.5 - s.human_welfare_index)
        s.social_tension = np.clip(
            0.3 * s.inequality_index +
            0.25 * food_insecurity +
            0.2 * env_stress +
            0.25 * expectation_gap,
            0.0, 1.0
        )

    def _ode_system(self, t: float, y: np.ndarray,
                    feedback: dict) -> np.ndarray:
        """
        Coupled ODE system for macro world dynamics.

        Args:
            t: time (years since start)
            y: state vector
            feedback: agent feedback dict from bridge
        Returns:
            dy/dt: time derivatives
        """
        dy = np.zeros(_N_STATES)

        # Unpack state
        co2 = max(280.0, y[_IDX["co2"]])
        temp = y[_IDX["temp"]]
        deep_temp = y[_IDX["deep_temp"]]
        fossil = np.clip(y[_IDX["fossil"]], 0.0, 1.0)
        minerals = np.clip(y[_IDX["minerals"]], 0.0, 1.0)
        pollution = np.clip(y[_IDX["pollution"]], 0.0, 1.0)
        ocean_acid = np.clip(y[_IDX["ocean_acid"]], 0.0, 1.0)
        pop = max(0.1, y[_IDX["population"]])
        gdp = max(0.01, y[_IDX["gdp"]])
        inequality = np.clip(y[_IDX["inequality"]], 0.0, 1.0)
        tech = max(0.5, y[_IDX["tech"]])
        renewable = np.clip(y[_IDX["renewable"]], 0.0, 1.0)
        freshwater = np.clip(y[_IDX["freshwater"]], 0.0, 1.0)
        sea_level = y[_IDX["sea_level"]]

        # Agent feedback signals (with defaults for standalone runs)
        emission_multiplier = feedback.get("emission_multiplier", 1.0)
        extraction_multiplier = feedback.get("extraction_multiplier", 1.0)
        renewable_investment = feedback.get("renewable_investment", 0.0)
        conflict_intensity = feedback.get("conflict_intensity", 0.0)
        agent_pop_factor = feedback.get("population_factor", 1.0)
        research_boost = feedback.get("research_boost", 0.0)

        # ================================================================
        # 1. CARBON CYCLE & CO2
        # ================================================================

        # Emissions: base rate scaled by fossil use, population, GDP
        # Baseline: ~42 GtCO2/yr in 2025, source: Global Carbon Project 2024
        # Note: (1-renewable)^0.6 because many sectors can't easily switch
        # (cement, steel, aviation, agriculture), so emissions decline slower
        # than renewable fraction suggests. Source: IEA Net Zero 2021
        # Anthropogenic CO2 has two components so emissions emerge from the
        # civilization's development instead of being a fixed present-day rate:
        #  (a) Industrial fossil emissions, gated by `industrialization` in [0,1]
        #      (the civ's industrial-tech development — see
        #      HistoricalSimulation.industrialization_level). A pre-industrial
        #      civ emits ~0 fossil CO2.
        #  (b) A small land-use term (deforestation / early agriculture — the
        #      "early Anthropocene", Ruddiman 2003) that exists even before
        #      industry and fades as industrialisation takes over.
        # industrialization defaults to 1.0 when no agent feedback is supplied
        # (standalone validation / present-day), so the calibrated 2025 rate and
        # the IPCC BAU checks are preserved exactly (industrial=full, land_use=0).
        industrialization = float(feedback.get("industrialization", 1.0))
        fossil_share = max(0.05, (1.0 - renewable) ** 0.6)
        industrial_emissions = (
            self.BASE_EMISSION_RATE *
            fossil_share *                # Non-renewable energy share
            (pop / 8.1) *                 # Population scaling
            (gdp ** 0.7) *               # Economic activity (sublinear: decoupling)
            emission_multiplier *         # Agent activity modifier
            (1.0 / (tech ** 0.5)) *      # Technology reduces emissions (sqrt: diminishing)
            min(1.0, fossil / 0.3 + 0.3) # Supply constraint only when very depleted
        ) * industrialization
        land_use_emissions = (
            self.LAND_USE_EMISSION_RATE * (pop / 8.1) * (1.0 - industrialization)
        )
        fossil_emissions = industrial_emissions + land_use_emissions

        # Natural absorption: ~50% of emissions, weakens with warming
        # Friedlingstein et al. 2024: carbon sink efficiency declining
        absorption_efficiency = self.NATURAL_ABSORPTION_RATE * (
            1.0 - self.ABSORPTION_TEMP_SENSITIVITY * max(0, temp - 1.0)
        )
        natural_absorption = fossil_emissions * max(0.2, absorption_efficiency)

        # Net CO2 change in ppm/yr.
        # FIX (v0.2): Previous code divided by 3.67 (GtCO2 -> GtC) and then
        # multiplied by 0.128, but 0.128 is already ppm/GtCO2 — the conversion
        # was applied twice, underestimating the CO2 rise by a factor of 3.67.
        # Verified against current Mauna Loa observations (~2.5-3 ppm/yr at
        # 42 GtCO2/yr * ~50% airborne fraction).
        net_emissions_gtco2 = fossil_emissions - natural_absorption
        dy[_IDX["co2"]] = net_emissions_gtco2 * self.PPM_PER_GTCO2

        # ================================================================
        # 2. CLIMATE: Two-layer energy balance model
        # ================================================================
        # dT/dt = (1/C) * [F - lambda*T - gamma*(T - T_deep)]
        # Source: Held et al. 2010, Gregory 2000, IPCC AR6 Chapter 7

        forcing = self.FORCING_COEFF * np.log(co2 / self.CO2_PREINDUSTRIAL)

        # Surface temperature
        dy[_IDX["temp"]] = (1.0 / self.OCEAN_HEAT_CAPACITY) * (
            forcing -
            self.CLIMATE_FEEDBACK * temp -
            self.DEEP_OCEAN_COUPLING * (temp - deep_temp)
        )

        # Deep ocean temperature
        dy[_IDX["deep_temp"]] = (1.0 / self.DEEP_OCEAN_CAPACITY) * (
            self.DEEP_OCEAN_COUPLING * (temp - deep_temp)
        )

        # Sea level rise: thermal expansion + ice melt proxy
        # ~3.6 mm/yr current rate, accelerating with temperature
        # Source: IPCC AR6 WG1 Chapter 9
        dy[_IDX["sea_level"]] = 0.0036 * (1.0 + 0.5 * max(0, temp - 1.0))

        # ================================================================
        # 3. RESOURCE DEPLETION
        # ================================================================

        # Fossil fuel depletion: smooth decline curve
        # Source: Hubbert 1956, calibrated per World3 (Meadows 2004)
        # At baseline rate, ~80 years to deplete half of remaining
        demand_factor = (
            (pop / 8.1) *
            gdp *
            (1.0 - renewable) *           # Renewables reduce fossil demand
            extraction_multiplier *        # Agent behavior
            (1.0 / tech)                  # Technology increases efficiency
        )
        extraction_rate = (
            (1.0 / self.FOSSIL_HALF_DEPLETION) *
            demand_factor *
            fossil                         # Extraction proportional to remaining stock
        )
        dy[_IDX["fossil"]] = -extraction_rate

        # Mineral depletion: similar but slower, partially recyclable
        mineral_demand = (
            (pop / 8.1) * gdp *
            extraction_multiplier *
            (1.0 - 0.3 * min(1.0, tech / 3.0))  # Recycling improves with tech
        )
        mineral_extraction = (
            (1.0 / self.MINERAL_HALF_DEPLETION) *
            mineral_demand *
            minerals                       # Proportional to remaining
        )
        dy[_IDX["minerals"]] = -mineral_extraction

        # Freshwater stress: worsens with population, temperature, pollution
        # Source: Schewe et al. 2014, water stress projections
        dy[_IDX["freshwater"]] = (
            0.002 * (pop / 8.1) +                     # Population pressure
            0.003 * max(0, temp - 1.5) +              # Climate impact on hydrology
            0.001 * pollution -                        # Pollution contaminates water
            0.002 * tech / 3.0 * (1.0 - freshwater)  # Tech helps (desalination etc.)
        )

        # ================================================================
        # 4. POLLUTION
        # ================================================================

        # Persistent pollution: generated by industry, decays slowly
        # Source: World3 pollution sector (Meadows 2004)
        pollution_generation = (
            0.01 * gdp * (pop / 8.1) *
            (1.0 - renewable * 0.5) *     # Renewables are cleaner
            (1.0 / tech)                  # Tech reduces pollution per GDP
        )
        pollution_decay = 0.02 * pollution  # ~50 year half-life for persistent pollutants
        dy[_IDX["pollution"]] = pollution_generation - pollution_decay

        # Ocean acidification: directly tied to CO2 absorption
        # Source: IPCC AR6 WG1 Chapter 5
        dy[_IDX["ocean_acid"]] = (
            0.0005 * max(0, co2 - 350) / 100.0 -  # Increases with CO2
            0.001 * ocean_acid                      # Very slow natural buffering
        )

        # ================================================================
        # 5. SOCIOECONOMIC DYNAMICS
        # ================================================================

        # Population growth: logistic with carrying capacity affected by food
        # Source: UN WPP 2024, Vollset et al. 2020 (Lancet)
        food_index = max(0.3, 1.0 - 0.12 * max(0, temp - 1.5) ** 1.5) * \
                     max(0.5, 1.0 - 0.3 * pollution)
        carrying_capacity = 12.0 * food_index * (1.0 - freshwater * 0.3)
        # Demographic transition: growth slows with wealth and education (tech proxy)
        demographic_transition = max(0.1, 1.0 - 0.3 * tech / 3.0 - 0.2 * gdp / 2.0)
        pop_growth = (
            self.POP_GROWTH_BASE *
            demographic_transition *
            pop * (1.0 - pop / carrying_capacity) *   # Logistic growth
            agent_pop_factor -                         # Agent reproduction rate
            0.001 * conflict_intensity * pop            # Conflict casualties
        )
        dy[_IDX["population"]] = pop_growth

        # GDP growth: endogenous, affected by tech, resources, climate, conflict
        # Source: Nordhaus DICE model (GDP growth sector)
        resource_drag = 1.0 - 0.3 * max(0, 1.0 - fossil) - 0.2 * max(0, 1.0 - minerals)
        climate_damage = 1.0 - 0.01 * temp ** 2  # Nordhaus damage function: D = a*T^2
        conflict_damage = 1.0 - 0.15 * conflict_intensity
        gdp_growth = (
            0.025 *                       # Base growth ~2.5%/yr
            tech / 1.0 *                  # Technology drives growth
            max(0.3, resource_drag) *     # Resource constraints
            max(0.5, climate_damage) *    # Climate damage
            max(0.5, conflict_damage) *   # Conflict costs
            gdp
        )
        dy[_IDX["gdp"]] = gdp_growth

        # Inequality: Kuznets curve with globalization and policy effects
        # Source: Piketty (2014), Earth4All inequality sector
        globalization_effect = 0.002 * gdp  # Growth can increase inequality
        redistribution = -0.005 * tech / 3.0 * (1.0 - inequality)  # Tech enables redistribution
        crisis_effect = 0.01 * conflict_intensity  # Conflict increases inequality
        dy[_IDX["inequality"]] = globalization_effect + redistribution + crisis_effect

        # Technology: S-curve growth with diminishing returns
        # Source: Nordhaus (endogenous tech change), Romer (1990)
        tech_ceiling = 5.0  # Max tech multiplier
        tech_growth = (
            self.TECH_GROWTH_BASE *
            (1.0 + research_boost) *      # Agent research investment
            tech *                         # Current level enables further growth
            (1.0 - tech / tech_ceiling) *  # Logistic ceiling (diminishing returns)
            (1.0 - 0.2 * conflict_intensity)  # Conflict disrupts R&D
        )
        dy[_IDX["tech"]] = tech_growth

        # Renewable energy transition
        # Source: Way et al. 2022 (empirical cost learning curves)
        # Slower in early decades due to infrastructure inertia
        infra_inertia = 0.3 + 0.7 * min(1.0, renewable / 0.3)  # Ramps up after 30%
        renewable_growth = (
            0.008 * tech / 1.0 *          # Technology drives adoption
            infra_inertia *               # Infrastructure inertia (slow start)
            (1.0 - renewable) *           # Logistic saturation
            (1.0 + renewable_investment) *  # Agent/policy investment
            (1.0 + 0.3 * max(0, co2 - 450) / 100)  # CO2 urgency drives adoption
        )
        dy[_IDX["renewable"]] = renewable_growth

        return dy

    def step(self, agent_feedback: dict = None) -> MacroState:
        """
        Advance macro model by one time step.

        Args:
            agent_feedback: dict from bridge.aggregate_agent_feedback()
                Keys: emission_multiplier, extraction_multiplier,
                      renewable_investment, conflict_intensity,
                      population_factor, research_boost
        Returns:
            Updated MacroState
        """
        if agent_feedback is None:
            agent_feedback = {}
        self._last_feedback = agent_feedback

        y0 = self._state_to_vector()

        # Solve ODE for one time step
        sol = solve_ivp(
            fun=lambda t, y: self._ode_system(t, y, agent_feedback),
            t_span=(0, self.dt_years),
            y0=y0,
            method='RK23',        # Fast, good enough for this timescale
            max_step=self.dt_years,
            rtol=1e-3,
            atol=1e-6,
        )

        if sol.success:
            self._vector_to_state(sol.y[:, -1])
        else:
            # Fallback: Euler step
            dy = self._ode_system(0, y0, agent_feedback)
            self._vector_to_state(y0 + dy * self.dt_years)

        self.state.year += self.dt_years

        return self.state

    def get_summary(self) -> dict:
        """Return summary dict for UI/stats."""
        s = self.state
        return {
            "year": round(s.year, 1),
            "co2_ppm": round(s.co2_ppm, 1),
            "temperature": round(s.temperature_anomaly, 2),
            "sea_level_m": round(s.sea_level_rise_m, 3),
            "fossil_fuels": round(s.fossil_fuels, 3),
            "minerals": round(s.minerals_global, 3),
            "pollution": round(s.persistent_pollution, 3),
            "population_B": round(s.global_population_billions, 2),
            "gdp_index": round(s.global_gdp_index, 3),
            "inequality": round(s.inequality_index, 3),
            "social_tension": round(s.social_tension, 3),
            "technology": round(s.technology_level, 3),
            "renewable_frac": round(s.renewable_fraction, 3),
            "food_index": round(s.food_production_index, 3),
            "welfare": round(s.human_welfare_index, 3),
        }
