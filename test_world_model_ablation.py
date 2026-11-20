"""Tests for the world_model_mode ablation switch (World config) and its script."""
import sys
from pathlib import Path

import numpy as np
import pytest

from shared_world_model import SharedWorldModel
from world import World

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from ablation_world_model import run_once  # noqa: E402

MODES = ("jepa", "none", "random")


def _world(mode: str, seed: int = 5, agents: int = 25) -> World:
    w = World(seed=seed, config={"start_year_bp": 70000, "world_model_mode": mode})
    w.logger.config.enabled = False
    w.spawn_initial_agents(agents)
    return w


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        World(seed=1, config={"world_model_mode": "jepaa"})


@pytest.mark.parametrize("mode", MODES)
def test_same_seed_same_mode_identical_metrics(mode: str) -> None:
    a = run_once(seed=3, mode=mode, ticks=40, agents=25, random_scale=1.0)
    b = run_once(seed=3, mode=mode, ticks=40, agents=25, random_scale=1.0)
    a.pop("wall_clock_s")
    b.pop("wall_clock_s")
    assert a.keys() == b.keys()
    for k in a:  # NaN-aware: final_train_loss is NaN when the model never trains
        assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k


def test_none_mode_never_calls_world_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def forbid(name: str):
        def _raise(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"world model {name} called in 'none' mode")
        return _raise

    for name in ("encode", "predict_next", "store_experience", "train_step",
                 "encode_batch", "predict_batch", "plan_batch", "store_experience_batch"):
        monkeypatch.setattr(SharedWorldModel, name, forbid(name))
    monkeypatch.setattr(SharedWorldModel, "planner", property(forbid("planner")))

    w = _world("none")
    for _ in range(60):
        w.step()
    assert calls == []
    assert w.shared_world_model.train_steps == 0
    assert len(w.shared_world_model.experience_buffer) == 0


def test_main_rng_identical_across_modes_first_tick() -> None:
    states = []
    for mode in MODES:
        w = _world(mode)
        before = w.rng.get_state()
        w.step()
        states.append((before, w.rng.get_state()))
    for before, after in states[1:]:
        assert np.array_equal(before[1], states[0][0][1]) and before[2:] == states[0][0][2:]
        assert np.array_equal(after[1], states[0][1][1]) and after[2:] == states[0][1][2:]


def test_random_mode_scale_and_bounds() -> None:
    w = World(seed=2, config={"world_model_mode": "random", "random_steering_scale": 0.25})
    draws = np.array([w.ablation_action() for _ in range(500)])
    assert draws.shape == (500, 8)
    assert np.abs(draws).max() <= 0.25
    assert np.abs(draws).max() > 0.2


def test_none_mode_action_is_zero() -> None:
    w = World(seed=2, config={"world_model_mode": "none"})
    assert np.array_equal(w.ablation_action(), np.zeros(8))
