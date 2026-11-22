"""Regression tests for the correctness pass (findings B1, B2, B4, B6-B9, B11, B12).

Each test guards one finding and states the invariant it protects. None of
these fixes may change simulation dynamics; the numeric tests pin that.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oikoumene.macro import macro
from oikoumene.agents.agents import Agent, SkillSet, generate_traits
from oikoumene.macro.macro import MacroModel, _crop_yield_factor
from oikoumene.core.world import Settlement, World
from oikoumene.worldmodel.world_model import heldout_probe_r2, sigreg_forward

PKG = Path(__file__).resolve().parents[1] / "src" / "oikoumene"


# --- B1: pollution decay constant ------------------------------------------

def test_pollution_decay_rate_constant_and_half_life() -> None:
    """Rate stays 0.02/yr: e-folding time 50 yr, half-life ~34.66 yr."""
    assert MacroModel.POLLUTION_DECAY_RATE == 0.02
    assert np.log(2) / MacroModel.POLLUTION_DECAY_RATE == pytest.approx(34.657, abs=1e-3)


def test_pollution_ode_uses_decay_constant() -> None:
    """The pollution derivative is generation minus POLLUTION_DECAY_RATE * stock."""
    m = MacroModel()
    y = m._state_to_vector()
    i = macro._IDX
    gen = 0.01 * y[i["gdp"]] * (y[i["population"]] / 8.1) * (1.0 - y[i["renewable"]] * 0.5) / y[i["tech"]]
    expected = gen - 0.02 * y[i["pollution"]]
    assert m._ode_system(0.0, y, {})[i["pollution"]] == pytest.approx(expected, rel=1e-12)


# --- B2: emissions baseline (documented, not recalibrated) -----------------

def test_default_state_co2_growth_rate_unchanged() -> None:
    """At the default state, emissions are ~38.10 GtCO2/yr and dCO2/dt ~2.482 ppm/yr."""
    m = MacroModel()
    dco2 = m._ode_system(0.0, m._state_to_vector(), {})[macro._IDX["co2"]]
    assert dco2 == pytest.approx(2.48215, abs=1e-4)
    emissions = dco2 / (MacroModel.PPM_PER_GTCO2 * (1.0 - 0.491))
    assert emissions == pytest.approx(38.098, abs=1e-2)


# --- B4: shared crop-yield factor is numerically identical -----------------

def _old_derived_food(temp: float, pollution: float, freshwater: float) -> float:
    temp_effect = max(0.3, 1.0 - 0.12 * max(0, temp - 1.5) ** 1.5)
    pollution_effect = max(0.5, 1.0 - 0.3 * pollution)
    water_effect = max(0.4, 1.0 - 0.5 * freshwater)
    return temp_effect * pollution_effect * water_effect


def _old_ode_carrying_capacity(temp: float, pollution: float, freshwater: float) -> float:
    food_index = max(0.3, 1.0 - 0.12 * max(0, temp - 1.5) ** 1.5) * \
                 max(0.5, 1.0 - 0.3 * pollution)
    return 12.0 * food_index * (1.0 - freshwater * 0.3)


def _grid() -> list[tuple[float, float, float]]:
    rng = np.random.RandomState(0)
    pts = [(t, p, f) for t in (-6.0, 0.0, 1.3, 1.5, 2.7, 6.0, 30.0)
           for p in (0.0, 0.3, 1.0) for f in (0.0, 0.25, 1.0)]
    pts += [tuple(v) for v in np.column_stack([rng.uniform(-10, 30, 500),
                                               rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)])]
    return pts


def test_derived_food_index_bit_identical() -> None:
    m = MacroModel()
    for temp, pol, fw in _grid():
        m.state.temperature_anomaly, m.state.persistent_pollution, m.state.freshwater_stress = temp, pol, fw
        m._compute_derived()
        assert m.state.food_production_index == _old_derived_food(temp, pol, fw)


def test_ode_carrying_capacity_bit_identical() -> None:
    for temp, pol, fw in _grid():
        assert 12.0 * _crop_yield_factor(temp, pol) * (1.0 - fw * 0.3) == \
            _old_ode_carrying_capacity(temp, pol, fw)


# --- B6: naming --------------------------------------------------------------

def test_depletion_constants_renamed() -> None:
    assert MacroModel.FOSSIL_DEPLETION_TIME_CONSTANT == 80.0
    assert MacroModel.MINERAL_DEPLETION_TIME_CONSTANT == 120.0
    assert not hasattr(MacroModel, "FOSSIL_HALF_DEPLETION")
    assert not hasattr(MacroModel, "MINERAL_HALF_DEPLETION")


# --- B7: Euler fallback is logged and counted --------------------------------

def test_euler_fallback_logged_and_counted(monkeypatch: pytest.MonkeyPatch,
                                           caplog: pytest.LogCaptureFixture) -> None:
    m = MacroModel()
    assert m.euler_fallback_count == 0
    failed = SimpleNamespace(success=False, message="forced failure", y=None)
    monkeypatch.setattr(macro, "solve_ivp", lambda **_: failed)
    with caplog.at_level(logging.WARNING, logger="macro"):
        m.step()
    assert m.euler_fallback_count == 1
    assert any("Euler fallback" in r.getMessage() for r in caplog.records)


def test_no_fallback_in_normal_step() -> None:
    m = MacroModel()
    for _ in range(12):
        m.step()
    assert m.euler_fallback_count == 0


# --- B8: govern age constant ------------------------------------------------

def test_govern_min_age_constant_and_threshold() -> None:
    assert Agent.GOVERN_MIN_AGE_YEARS == 8.0
    a = Agent(0.0, 0.0, rng=np.random.RandomState(0))
    a._era_time_scale_cache = 1.0 / 12.0
    threshold = a._yrs_to_ticks(Agent.GOVERN_MIN_AGE_YEARS)
    a.age = threshold
    assert a._evaluate_needs()["govern"] == 0
    a.age = threshold + 1
    assert a._evaluate_needs()["govern"] > 0


# --- B9: rng is required; runs never consume global np.random ---------------

@pytest.mark.parametrize("ctor", [
    lambda: Agent(0.0, 0.0),
    lambda: SkillSet(),
    lambda: generate_traits(),
    lambda: Settlement(0.0, 0.0, 1),
    lambda: sigreg_forward(np.zeros((4, 3))),
])
def test_rng_argument_required(ctor) -> None:
    with pytest.raises(TypeError):
        ctor()


def test_run_does_not_consume_global_numpy_random() -> None:
    np.random.seed(12345)
    before = np.random.get_state()
    world = World(seed=3)
    world.logger.config.enabled = False
    world.spawn_initial_agents(25)
    for _ in range(30):
        world.step()
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]


# --- B11: held-out probe R^2 --------------------------------------------------

def test_heldout_probe_split_is_deterministic_and_rng_free() -> None:
    data_rng = np.random.RandomState(1)
    Z = data_rng.randn(100, 24)
    Y = Z[:, :3] @ data_rng.randn(3, 3)
    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    a = heldout_probe_r2(Z, Y)
    b = heldout_probe_r2(Z, Y)
    assert np.array_equal(before, np.random.get_state()[1])
    assert a == b
    assert (a["n_train"], a["n_test"]) == (80, 20)
    assert a["physical_r2_heldout"] > 0.99


def test_heldout_probe_detects_overfitting() -> None:
    """Pure noise: in-sample R^2 > 0 from overfitting, held-out R^2 stays <= ~0."""
    data_rng = np.random.RandomState(2)
    Z = data_rng.randn(100, 24)
    Y = data_rng.randn(100, 3)
    Z_aug = np.hstack([Z, np.ones((100, 1))])
    W, *_ = np.linalg.lstsq(Z_aug, Y, rcond=None)
    in_sample = 1.0 - np.sum((Y - Z_aug @ W) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)
    assert in_sample > 0.1
    assert heldout_probe_r2(Z, Y)["physical_r2_heldout"] < 0.05


# --- B12: no print() in simulation modules ------------------------------------

@pytest.mark.parametrize("module", ["core/world.py", "core/scenarios.py", "macro/macro.py", "worldmodel/layers.py",
                                    "worldmodel/world_model.py", "core/history.py", "core/paleoclimate.py",
                                    "society/technology.py", "llm/llm_module.py", "llm/url_safety.py",
                                    "llm/generation.py", "society/geopolitics.py", "society/interactions.py",
                                    "agents/agents.py", "agents/traits.py", "agents/memory.py", "agents/movement.py",
                                    "agents/planning.py", "agents/actions.py"])
def test_no_print_calls_in_simulation_modules(module: str) -> None:
    tree = ast.parse((PKG / module).read_text())
    prints = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"]
    assert prints == []
