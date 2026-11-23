"""
Views mixin for World (world.py): read-only views of the world state: era-aware
summaries, the full UI state and the JEPA backend status. Changing the JEPA
backend lives in world_jepa.JepaBackendMixin.

Expects on the host class: the attributes agents, businesses, geopolitics, god_mode,
history, jepa_preset, llm, logger, macro, macro_always_active, recent_dialogues,
scenario, settlements, shared_world_model, stats_history and tick.
"""

from oikoumene.core.history import TECH_TREE
from oikoumene.core.world_jepa import _torch_available
from oikoumene.core.world_paleo import _paleo_population_billions


class ViewsMixin:
    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _build_era_aware_summaries(self) -> tuple[dict, dict, dict]:
        """
        Build (history, macro, geopolitics) summary dicts with era-aware
        climate sourcing. Used by both `step()` (for the websocket "tick"
        emit) and `get_full_state()` (for the "full_state" emit) so both
        paths deliver identical payloads to the right-sidebar UI.

        Modern era (year_bp < 200 or scenario.macro_active_from_start):
            - history.{co2_ppm, temperature_anomaly, sea_level_m,
                       year_ce, year_bp, year_display} are overridden
              with MacroModel.state values, since the macro ODE is the
              canonical source of truth in the Industrial+ era.
            - macro = MacroModel.get_summary()  (full set of fields)

        Paleo era (year_bp >= 200):
            - history kept verbatim (PaleoclimateModel: EPICA/Vostok +
              Clark et al. 2009).
            - macro populated with paleoclimate-derived climate fields
              and a paleopopulation interpolation (McEvedy & Jones 1978;
              Biraben 2003; HYDE 3.1) so the panel evolves with year_bp.
              Industrial-era fields (fossil_fuels, renewable_frac,
              persistent_pollution) carry their pre-industrial physical
              values; technology is normalised to the tech-tree size.

        Geopolitics: settlement count is always injected so the Nations
        tab reflects pre-nation tribal activity in paleo era (where
        nations/conflicts/trade are zero by design until settlements
        grow >= NATION_FORMATION_POP).
        """
        is_modern = self.macro_always_active or self.history.year_bp < 200
        history_summary = self.history.get_summary()

        if is_modern:
            s = self.macro.state
            history_summary["co2_ppm"] = round(s.co2_ppm, 1)
            history_summary["temperature_anomaly"] = round(s.temperature_anomaly, 2)
            history_summary["sea_level_m"] = round(s.sea_level_rise_m, 3)
            year_ce = s.year
            history_summary["year_ce"] = round(year_ce, 1)
            history_summary["year_bp"] = round(1950.0 - year_ce, 1)
            if year_ce < 0:
                history_summary["year_display"] = f"{int(abs(year_ce)):,} BCE"
            else:
                history_summary["year_display"] = f"{int(year_ce):,} CE"
            macro_summary = self.macro.get_summary()
        else:
            climate = self.history.paleoclimate.get_climate(self.history.year_bp)
            n_techs = len(self.history.discovered_techs)
            # Use the real tech-tree size so this gauge stays correct if the
            # tree is edited (the old getattr(self.history, "tech_tree", [])
            # always hit the [] fallback -> hardcoded 32, silently desyncing).
            tech_tree_size = max(1, len(TECH_TREE))
            # Early-Anthropocene land-use signal (Ruddiman 2003): the simulated
            # civilisation's settled footprint nudges CO2 slightly above the
            # natural paleoclimate baseline, so a growing pre-industrial civ has a
            # small but visible climate effect (the macro ODE is dormant in the
            # paleo era). Capped at a few ppm — this is land use, not fossil fuel.
            settled_footprint = min(1.0, len(self.settlements) / 40.0)
            land_use_co2_ppm = round(12.0 * settled_footprint, 2)
            paleo_co2 = round(climate["co2_ppm"] + land_use_co2_ppm, 1)
            macro_summary = {
                "year": round(self.history.get_current_year_ce(), 1),
                "co2_ppm": paleo_co2,
                "co2_natural_ppm": round(climate["co2_ppm"], 1),
                "anthropogenic_co2_ppm": land_use_co2_ppm,
                "temperature": round(climate["temperature_anomaly"], 2),
                "sea_level_m": round(climate["sea_level_m"], 3),
                "population_B": round(
                    _paleo_population_billions(self.history.year_bp), 4
                ),
                "fossil_fuels": 1.0,        # Untapped before industrial era
                "renewable_frac": 0.0,      # No industrial energy infrastructure
                "pollution": 0.0,           # Pre-industrial atmosphere
                "technology": round(min(1.0, n_techs / tech_tree_size), 3),
            }
            # Reflect the civ's land-use effect in the history panel too.
            history_summary["co2_ppm"] = paleo_co2

        geopolitics_summary = self.geopolitics.get_summary()
        geopolitics_summary["settlements"] = len(self.settlements)
        return history_summary, macro_summary, geopolitics_summary

    def get_full_state(self) -> dict:
        """Get complete world state for UI rendering."""
        alive_agents = [a for a in self.agents if a.alive]
        history_summary, macro_summary, geopolitics_summary = (
            self._build_era_aware_summaries()
        )
        return {
            "tick": self.tick,
            "agents": [a.to_dict() for a in alive_agents],
            "businesses": [b.to_dict() for b in self.businesses if b.active],
            "settlements": [s.to_dict() for s in self.settlements],
            "stats": self.stats_history[-1] if self.stats_history else {},
            "stats_history": self.stats_history[-200:],
            "history": history_summary,
            "macro": macro_summary,
            "geopolitics": geopolitics_summary,
            "nations": self.geopolitics.get_nations_list(),
            "conflicts": self.geopolitics.active_conflicts,
            "scenario": {"id": self.scenario.id, "name": self.scenario.name},
            "dialogues": self.recent_dialogues[-20:],
            "llm_status": self.llm.get_status(),
            "god_mode_status": self.god_mode.get_status(),
            "logger": self.logger.get_status(),
            "jepa_status": self.get_jepa_status(),
        }

    def get_jepa_status(self) -> dict:
        """Current JEPA world-model configuration for the dashboard."""
        m = self.shared_world_model
        j = m._jepa
        device = getattr(j, "device", None)
        return {
            "backend": getattr(m, "backend", "numpy"),
            "preset": getattr(self, "jepa_preset", "default"),
            "latent_dim": m.latent_dim,
            "sigreg_mode": getattr(j, "sigreg_mode", "moments"),
            "sigreg_projections": getattr(j, "sigreg_projections", None),
            "lambda_reg": round(float(m.lambda_reg), 4),
            "train_steps": m.train_steps,
            "buffer_size": len(m.experience_buffer),
            "device": str(device) if device is not None else "cpu",
            "torch_available": _torch_available(),
        }
