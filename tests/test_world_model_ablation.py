"""Tests for the world_model_mode ablation switch (World config) and its script."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oikoumene.worldmodel.shared_world_model import SharedWorldModel
from oikoumene.core.world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablation_world_model import (holm_adjust, load_reused, match_per_component,  # noqa: E402
                                  min_detectable_difference, parse_args, run_once,
                                  uniform_scale_for_intensity, write_raw)

MODES = ("jepa", "none", "random", "const")


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


# --- Stage 1b: JEPA seed derived from the world seed --------------------------

def _encoder_weights(w: World) -> dict:
    return {k: v.copy() for k, v in w.shared_world_model.encoder.params.items()}


def test_jepa_init_depends_on_world_seed() -> None:
    a, b, c = World(seed=1), World(seed=1), World(seed=2)
    wa, wb, wc = _encoder_weights(a), _encoder_weights(b), _encoder_weights(c)
    assert all(np.array_equal(wa[k], wb[k]) for k in wa)
    assert any(not np.array_equal(wa[k], wc[k]) for k in wa)


def test_jepa_rebuild_keeps_world_derived_seed() -> None:
    w = World(seed=4)
    before = _encoder_weights(w)
    w.set_jepa_backend("numpy")
    after = _encoder_weights(w)
    assert all(np.array_equal(before[k], after[k]) for k in before)


# --- Ablation v2: const mode, training interval, workers, statistics ----------

def test_const_mode_action() -> None:
    w = World(seed=2, config={"world_model_mode": "const", "const_intensity": 0.31})
    expected = np.zeros(8)
    expected[2] = 0.31
    assert np.array_equal(w.ablation_action(), expected)
    with pytest.raises(ValueError):
        World(seed=2, config={"world_model_mode": "const", "const_intensity": 0.05})


def test_const_mode_fixes_intensity_and_never_trains() -> None:
    row = run_once(seed=1, mode="const", ticks=40, agents=25, const_intensity=0.4)
    assert row["mean_intensity"] == pytest.approx(0.4)
    assert row["mean_abs_a0"] == 0.0 and row["mean_abs_a1"] == 0.0
    assert row["train_steps"] == 0


def test_train_interval_default_and_override() -> None:
    assert World(seed=1).train_interval_ticks == 20
    default = run_once(seed=1, mode="jepa", ticks=120, agents=25)
    fast = run_once(seed=1, mode="jepa", ticks=120, agents=25, train_interval=2)
    assert fast["train_steps"] > default["train_steps"] > 0
    with pytest.raises(ValueError):
        World(seed=1, config={"train_interval_ticks": 0})


_WORKERS_CHECK = """
import json, sys
sys.path.insert(0, sys.argv[1])
from ablation_world_model import run_many
jobs = [dict(seed=s, mode=m, ticks=25, agents=25) for s in (0, 1) for m in ("jepa", "random")]
out = {}
for w in (1, 2):
    rows, losses = run_many(jobs, workers=w)
    for r in rows:
        r.pop("wall_clock_s")
    out[w] = {"rows": rows, "losses": losses}
print(json.dumps(out))
"""


def test_workers_do_not_change_results() -> None:
    """workers=1 and workers=2 give identical rows and loss logs.

    Runs in a fresh interpreter: test_simcore_fixes imports app, whose
    eventlet.monkey_patch() makes multiprocessing deadlock inside pytest.
    """
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    proc = subprocess.run([sys.executable, "-c", _WORKERS_CHECK, scripts], capture_output=True,
                          text=True, timeout=300, cwd=Path(__file__).resolve().parents[1], check=False)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])  # NaN round-trips as NaN
    one, two = out["1"], out["2"]
    assert len(one["rows"]) == 4
    assert json.dumps(one) == json.dumps(two)


def test_holm_adjust_known_values() -> None:
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotone: max(0.06, 1 * 0.04)
    assert adj["d"] != adj["d"]


def test_min_detectable_difference_textbook() -> None:
    """Paired t, n = 30, 80% power, alpha 0.05: d is about 0.53 SD."""
    assert min_detectable_difference(1.0, 30) == pytest.approx(0.53, abs=0.01)
    assert min_detectable_difference(2.0, 30) == pytest.approx(2 * min_detectable_difference(1.0, 30))


def test_min_detectable_difference_small_n_strict_alpha() -> None:
    """Small n at the Holm alpha must not crash (nct NaN at large noncentrality)."""
    value = min_detectable_difference(1.0, 2, alpha=0.05 / 6)
    assert np.isfinite(value) and value > min_detectable_difference(1.0, 30, alpha=0.05 / 6)


# --- (iii-b), post hoc: per-component matched controls, reused jepa rows ------

def test_per_component_random_scale() -> None:
    scales = [0.5, 0.25, 0.9, 0, 0, 0, 0, 0]
    w = World(seed=2, config={"world_model_mode": "random", "random_steering_scale": scales})
    draws = np.array([w.ablation_action() for _ in range(2000)])
    assert np.all(np.abs(draws) <= np.array(scales))
    assert np.all(draws[:, 3:] == 0) and np.abs(draws[:, 2]).max() > 0.85
    with pytest.raises(ValueError):
        World(seed=2, config={"random_steering_scale": [0.5, 0.5]})
    with pytest.raises(ValueError):
        World(seed=2, config={"random_steering_scale": -0.1})


def test_vector_scale_equals_scalar_scale() -> None:
    a = World(seed=3, config={"world_model_mode": "random", "random_steering_scale": 0.6})
    b = World(seed=3, config={"world_model_mode": "random", "random_steering_scale": [0.6] * 8})
    assert isinstance(a.random_steering_scale, float)
    for _ in range(50):
        assert np.array_equal(a.ablation_action(), b.ablation_action())


def test_uniform_scale_for_intensity_monte_carlo() -> None:
    s = uniform_scale_for_intensity(0.478)
    u = np.random.RandomState(0).uniform(-s, s, 400_000)
    assert np.clip(np.abs(u), 0.1, 1.0).mean() == pytest.approx(0.478, abs=2e-3)
    with pytest.raises(ValueError):
        uniform_scale_for_intensity(0.6)  # would need s > 1


def test_match_per_component() -> None:
    rows = [{"seed": i, "mean_abs_a0": 0.2 + 0.02 * i, "mean_abs_a1": 0.3, "mean_intensity": 0.4} for i in range(3)]
    m = match_per_component(rows)
    assert m["random_scale"][:2] == pytest.approx([0.44, 0.6])
    assert m["random_scale"][2] == pytest.approx(uniform_scale_for_intensity(0.4))
    assert m["random_scale"][3:] == [0.0] * 5 and m["const_intensity"] == pytest.approx(0.4)


def test_load_reused_round_trip(tmp_path: Path) -> None:
    losses: list = []
    rows = [run_once(seed=s, mode="jepa", ticks=25, agents=25, train_interval=2, loss_log=losses) for s in (0, 1)]
    write_raw(rows, losses, tmp_path)
    args = parse_args(["--seeds", "2", "--ticks", "25", "--train-interval", "2", "--workers", "1"])
    got, got_losses = load_reused(tmp_path, args)
    assert [r["seed"] for r in got] == [0, 1] and len(got_losses) == len(losses)
    for k in ("food_gathered", "mean_abs_a0", "mean_intensity", "final_train_loss"):
        assert [r[k] for r in got] == [r[k] for r in rows], k
    with pytest.raises(ValueError):
        load_reused(tmp_path, parse_args(["--seeds", "3", "--ticks", "25", "--train-interval", "2"]))
