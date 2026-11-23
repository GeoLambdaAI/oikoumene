"""
JEPA backend mixin for World (world.py): the runtime backend/preset swap
(set_jepa_backend, JEPA_PRESETS); plus _JEPA_SEED_OFFSET and _torch_available.

Expects on the host class: the attributes agents, seed and shared_world_model, and
the method get_jepa_status (ViewsMixin). Sets shared_world_model and jepa_preset.
"""

from oikoumene.worldmodel.shared_world_model import SharedWorldModel

# Offset for the shared JEPA seed. Derived from the world seed so weight init,
# CEM and SIGReg draws vary with it; the offset keeps the JEPA streams (which
# use s, s+7, s+13) from replaying the main world RNG sequence RandomState(seed).
_JEPA_SEED_OFFSET = 7_919


def _torch_available() -> bool:
    """True if the optional PyTorch backend can be imported (no import cost)."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


class JepaBackendMixin:
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
