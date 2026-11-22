"""
Paleo mixin for World (world.py): migration-frontier spawning, ice-age effects and
the macro hand-off at Industrial onset; plus the UI paleodemography helper
_paleo_population_billions and its _PALEO_POP_TABLE.

Expects on the host class: the attributes agents, history, macro, resources and rng;
and the method get_nearby_agents() (SpatialMixin).
"""

import numpy as np

from oikoumene.agents.agents import Agent


# ============================================================================
# Paleodemography (UI-only display helper)
# ============================================================================
#
# Piecewise-linear interpolation of canonical global-population estimates
# for paleo-era display in the right-sidebar Global State panel. The
# MacroModel ODE only activates in Industrial+ era (year_bp < 200) and
# carries no population dynamics for earlier periods, so without this
# the UI shows the MacroState.population default (8.1 B, year-2025
# baseline) frozen across the entire 70,000-yr history view.
#
# Sources (all canonical references in paleodemography):
#   - McEvedy & Jones (1978), Atlas of World Population History (Penguin).
#     Standard reference for AD-era population back to ~10 kBP.
#   - Biraben (2003), An essay concerning mankind's evolution,
#     Population & Societies 394, 1-4.
#   - Klein Goldewijk et al. (2010), HYDE 3.1: Long-term dynamic modeling
#     of global population and built-up area, The Holocene 20, 565-573.
#
# Deep-paleo values (>10 kBP) carry order-of-magnitude uncertainty and
# are best-estimates within the literature envelope; they are intended
# only for qualitative UI display, not quantitative modelling.
_PALEO_POP_TABLE = [
    (70000, 0.0005),  # ~500k, MIS 4; H. sapiens dispersal phase
    (50000, 0.001),   # ~1M, Upper Paleolithic transition
    (21000, 0.002),   # ~2M, Last Glacial Maximum (lower-bound estimate)
    (10000, 0.005),   # ~5M, end-Pleistocene (McEvedy & Jones)
    (5000,  0.050),   # ~50M, mid-Holocene (McEvedy & Jones, 3000 BCE)
    (2000,  0.170),   # ~170M, ~50 BCE (McEvedy & Jones)
    (1000,  0.265),   # ~265M, 950 CE
    (500,   0.425),   # ~425M, 1450 CE
    (200,   0.770),   # ~770M, 1750 CE (industrial revolution onset)
]


def _paleo_population_billions(year_bp: float) -> float:
    """Linear interpolation through `_PALEO_POP_TABLE`. UI display only."""
    table = _PALEO_POP_TABLE
    if year_bp >= table[0][0]:
        return table[0][1]
    if year_bp <= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        y1, p1 = table[i]
        y2, p2 = table[i + 1]
        if y1 >= year_bp >= y2:
            t = (y1 - year_bp) / (y1 - y2)
            return p1 + t * (p2 - p1)
    return table[-1][1]


class PaleoMixin:
    def _spawn_migration_frontier(self):
        """
        Spawn agents at the frontier of human migration based on
        historical migration waves from history.py.

        Represents population growth at migration frontiers — new groups
        splitting off from existing populations and pushing into new territories.
        """
        from oikoumene.core.history import MIGRATION_WAVES
        from oikoumene.core.earth import is_land

        year_bp = self.history.year_bp
        alive = [a for a in self.agents if a.alive]

        # Don't spawn if population is already large
        if len(alive) > 200:
            return

        # Find migration waves that should have reached by now
        for wave in MIGRATION_WAVES:
            if year_bp > wave["year_bp"]:
                continue  # Not yet reached

            # Check if any agent is already near this wave point
            wave_lat, wave_lng = wave["lat"], wave["lng"]
            nearby = self.get_nearby_agents(wave_lat, wave_lng, 10.0)
            if nearby:
                continue  # Already have agents there

            # Spawn 1-3 frontier agents near this wave point
            n_spawn = self.rng.randint(1, 4)
            for _ in range(n_spawn):
                lat = wave_lat + self.rng.normal(0, 3)
                lng = wave_lng + self.rng.normal(0, 3)
                if is_land(lat, lng) and not self.history.paleoclimate.get_ice_mask(year_bp, lat, lng):
                    agent = Agent(lat, lng, rng=self.rng)
                    agent.energy = 80 + self.rng.random() * 20
                    agent.wealth = 5 + self.rng.random() * 10
                    self.agents.append(agent)

    def _apply_ice_age_effects(self):
        """
        Apply paleoclimate ice sheet coverage to terrain and resources.

        Paleoclimate trajectory (temperature_anomaly, ice_mask) comes from
        history.PaleoclimateModel, which is calibrated to EPICA/Vostok ice
        cores and Clark et al. (2009) LGM ice-sheet reconstructions.

        Two coupled bugs in the previous implementation:

        (i)  food_regen was scaled multiplicatively each call:
                self.resources.food_regen[r, c] *= cold_factor
             That ratchet compounds over the thousands of paleo ticks in a
             Pleistocene-spanning run. With cold_factor < 1 sustained for
             40 000+ paleo applications, food_regen underflows to ~0 in
             every cold cell, regardless of whether the climate later
             warms.

        (ii) Cells that became ice-covered had their food, food_regen,
             wood, wood_regen, and water set to zero, but were never
             restored when the ice mask later retreated. Post-glacial
             cells therefore stayed at zero productivity permanently
             (e.g. northern Europe and Canada from ~21 000 BP to the
             Modern cutoff), which is inconsistent with the paleoclimate
             record of recolonization after deglaciation.

        Fix: snapshot per-cell baselines on first call, then for each cell
        each tick set values from the baselines (idempotent, non-ratcheting)
        and seed post-glacial recovery on the iced->non-iced transition
        using a per-cell _was_iced flag. The cold_factor formula itself is
        unchanged — at the LGM temperature anomaly of -8 degC it yields
        ~36% of baseline productivity, within the paleo-NPP envelope of
        Adams & Faure (1998) and Crowley & Baum (1997).
        """
        year_bp = self.history.year_bp
        climate = self.history.paleoclimate.get_climate(year_bp)
        temp_offset = climate["temperature_anomaly"]

        # Cold-era productivity factor (unchanged).
        if temp_offset < -2:
            cold_factor = max(0.3, 1.0 + temp_offset * 0.08)
        else:
            cold_factor = 1.0

        res = self.resources
        # Baselines and _was_iced are captured once in initialize_from_terrain
        # (pristine terrain x fertility), so nothing to snapshot here.

        # Vectorised ice-sheet application. This was a per-cell Python double loop
        # that called get_ice_mask() — which itself recomputes the climate — for
        # every one of the ~12k grid cells, costing ~0.29 s every 50 ticks and
        # throttling the sim through the glacial paleo era. The numpy form below is
        # cell-for-cell identical but runs in well under a millisecond.
        rows, cols = res.rows, res.cols
        lat = (res.lat_max - (np.arange(rows) + 0.5) * res.cell_size_deg)[:, None]  # (rows,1)
        lng = (res.lng_min + (np.arange(cols) + 0.5) * res.cell_size_deg)[None, :]  # (1,cols)

        # Whole-grid ice mask, matching PaleoclimateModel.get_ice_mask: no extra
        # ice when warmer than -2 deg C; otherwise continental ice sheets (whose
        # southern boundary moves with ice_scale) plus the Antarctic sheet.
        ice = np.zeros((rows, cols), dtype=bool)
        if temp_offset <= -2.0:
            ice_scale = float(np.clip(-temp_offset / 8.0, 0, 1))
            for extent in self.history.paleoclimate.LGM_ICE_EXTENT.values():
                boundary = 75.0 - (75.0 - extent["lat_south"]) * ice_scale
                in_lng = (lng >= extent["lng_min"]) & (lng <= extent["lng_max"])
                ice |= in_lng & (lat >= boundary)
            antarctic_boundary = -65.0 + 10.0 * ice_scale
            ice |= lat <= antarctic_boundary
        not_ice = ~ice

        # Recovery applies to cells iced on a previous call that are now ice-free
        # (computed before _was_iced is updated below).
        recovered = not_ice & res._was_iced

        # Under ice: zero biological productivity; mark for recovery tracking.
        res.food[ice] = 0
        res.food_regen[ice] = 0
        res.wood[ice] = 0
        res.wood_regen[ice] = 0
        res.water[ice] = 0
        res._was_iced[ice] = True

        # Post-glacial recovery on the iced->non-iced transition: seed a fraction
        # of baseline so natural regen refills the cell over subsequent ticks.
        res.food[recovered] = res._baseline_food[recovered] * 0.1
        res.wood[recovered] = res._baseline_wood[recovered] * 0.1
        res.water[recovered] = res._baseline_water[recovered] * 0.5
        res._was_iced[recovered] = False

        # Non-iced cells: rebuild regen from baselines (idempotent, non-ratcheting
        # — the fix for bug (i)). cold_factor scales food regen; wood regen is the
        # bare baseline (not cold-scaled in the original model). Compose with any
        # active god-mode drought so this rewrite does not erase it.
        res.food_regen[not_ice] = (
            res._baseline_food_regen[not_ice] * cold_factor
            * res.drought_food_factor[not_ice]
        )
        res.wood_regen[not_ice] = res._baseline_wood_regen[not_ice]

    def _hand_off_macro_from_paleo(self) -> None:
        """Seed the macro ODE from the paleoclimate trajectory at Industrial onset.

        In a historical run the macro model is dormant through the paleo era; when
        it first activates it would otherwise start from the present-day MacroState
        defaults (1.3 deg C, 425 ppm), producing a sharp discontinuity. Instead we
        hand off continuously: climate carries the lower paleo values forward, the
        socioeconomic state is reset to pre-industrial, and the macro then evolves
        the climate upward as the civilization industrialises (emissions are
        gated by industrialization_level()).
        """
        climate = self.history.paleoclimate.get_climate(self.history.year_bp)
        s = self.macro.state

        # Climate continuity (no jump): carry the paleo values forward.
        s.year = self.history.get_current_year_ce()
        s.co2_ppm = float(climate["co2_ppm"])
        s.temperature_anomaly = float(climate["temperature_anomaly"])
        s.deep_ocean_temp = float(climate["temperature_anomaly"]) * 0.2  # deep ocean lags
        s.sea_level_rise_m = max(0.0, float(climate["sea_level_m"]))

        # Pre-industrial socioeconomic state at handoff (Industrial onset).
        s.global_population_billions = float(_paleo_population_billions(self.history.year_bp))
        s.global_gdp_index = max(0.02, s.global_population_billions / 8.1)  # pre-industrial economy
        s.fossil_fuels = 1.0          # reserves untapped before industry
        s.minerals_global = 1.0
        s.renewable_fraction = 0.0    # no industrial energy infrastructure yet
        s.persistent_pollution = 0.0
        s.ocean_acidification = 0.0
        s.technology_level = 1.0      # neutral emission-intensity baseline; grows from here

        # Recompute derived fields (radiative_forcing, food_production_index,
        # social_tension, human_welfare_index) from the just-seeded state.
        # Without this they keep their present-day (2025) defaults until the
        # next macro step, which the UI and agent observations would read for up
        # to macro_update_interval-1 ticks after the handoff.
        self.macro._compute_derived()
