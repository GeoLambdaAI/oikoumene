"""
Views mixin for World (world.py): era-aware summaries, the full UI state, and the
JEPA backend status and runtime swap (JEPA_PRESETS); plus _JEPA_SEED_OFFSET and
_torch_available.

Expects on the host class: the attributes agents, businesses, geopolitics, god_mode,
history, jepa_preset, llm, logger, macro, macro_always_active, recent_dialogues,
scenario, seed, settlements, shared_world_model, stats_history and tick.
"""

from oikoumene.core.history import TECH_TREE
from oikoumene.core.world_paleo import _paleo_population_billions
from oikoumene.worldmodel.shared_world_model import SharedWorldModel

# Offset for the shared JEPA seed. Derived from the world seed so weight init,
# CEM and SIGReg draws vary with it; the offset keeps the JEPA streams (which
# use s, s+7, s+13) from replaying the main world RNG sequence RandomState(seed).
_JEPA_SEED_OFFSET = 7_919


def _torch_available() -> bool:
    """True if the optional PyTorch backend can be imported (no import cost)."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


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

    # ------------------------------------------------------------------
    # JEPA world-model backend control (numpy default / torch, swappable
    # at runtime from the dashboard). See shared_world_model.SharedWorldModel
    # and world_model_torch.TorchJEPAWorldModel.
    # ------------------------------------------------------------------

    # Presets exposed in the UI. "default" reproduces the repo's deployed
    # config exactly; "paper" enables the LeWorldModel (Maes et al. 2026,
    # arXiv:2603.19312) Epps-Pulley SIGReg with the paper's lambda and a
    # large projection count, plus predictor dropout. "paper" requires torch.
    JEPA_PRESETS = {
        "default": {},
        "paper": {
            "sigreg_mode": "epps_pulley",
            "sigreg_projections": 1024,
            "lambda_reg": 0.1,
            "predictor_dropout": 0.1,
        },
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

    def set_jepa_backend(self, backend: str = "numpy", preset: str = "default",
                         device: str = "auto") -> dict:
        """
        Rebuild the shared JEPA model with a new backend/preset at runtime.

        The experience buffer is preserved across the swap (it holds backend-
        agnostic numpy tuples), so accumulated learning data is not lost. All
        agent references are repointed to the new model.

        Returns a status dict; on failure (e.g. torch not installed, invalid
        device) returns {"ok": False, "error": ...} with the unchanged status,
        leaving the live model intact.
        """
        backend = (backend or "numpy").lower()
        preset = (preset or "default").lower()

        if backend not in ("numpy", "torch"):
            return {"ok": False, "error": f"unknown backend: {backend}",
                    **self.get_jepa_status()}
        if preset not in self.JEPA_PRESETS:
            return {"ok": False, "error": f"unknown preset: {preset}",
                    **self.get_jepa_status()}
        if backend == "numpy" and preset == "paper":
            return {"ok": False,
                    "error": "paper preset requires the torch backend",
                    **self.get_jepa_status()}
        if backend == "torch" and not _torch_available():
            return {"ok": False,
                    "error": "PyTorch is not installed (pip install 'torch>=2.2')",
                    **self.get_jepa_status()}

        old = self.shared_world_model
        kwargs = {
            "obs_dim": old.obs_dim,
            "action_dim": old.action_dim,
            "latent_dim": old.latent_dim,
            "backend": backend,
            "seed": self.seed + _JEPA_SEED_OFFSET,
        }
        if backend == "torch":
            kwargs["device"] = device
            kwargs.update(self.JEPA_PRESETS[preset])

        try:
            new_model = SharedWorldModel(**kwargs)
        except Exception as exc:  # torch import/device/init failure
            return {"ok": False, "error": str(exc), **self.get_jepa_status()}

        # Preserve experience history (most-recent up to the new cap).
        old_buf = list(old.experience_buffer)
        if old_buf:
            new_model.experience_buffer.extend(old_buf[-new_model.max_buffer_size:])

        # Swap references everywhere the old model was held.
        self.shared_world_model = new_model
        self.jepa_preset = preset
        import oikoumene.agents.agents as _agents_module
        _agents_module._shared_world_model = new_model
        for agent in self.agents:
            agent.world_model = new_model

        return {"ok": True, **self.get_jepa_status()}
