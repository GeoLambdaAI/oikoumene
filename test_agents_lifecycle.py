"""
Tests für die era-aware Lebenszyklus-Logik in agents_v2.py.

Verifiziert:
1. _yrs_to_ticks gibt korrekte Werte für verschiedene Eras
2. Modern-Verhalten ist nahe am Original (max 10% Drift in den meisten Schwellen)
3. Reproduktionsalter wird in Modern auf 15 Jahre gefixt (war buggy 3.3 Jahre)
4. Paleo-Verhalten ist konsistent (alles auf min 1 Tick komprimiert)
5. Default-era_time_scale_cache funktioniert vor der ersten update()
"""
import sys, os, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Try the real world_model first; only stub if its dependency chain isn't
# available. Unconditionally stubbing here leaked into sibling test files
# (test_shared_world_model) and replaced JEPAWorldModel with a dummy.
if 'world_model' not in sys.modules:
    try:
        import world_model  # noqa: F401
    except Exception:
        stub = types.ModuleType('world_model')
        class _DummyJEPA:
            def __init__(self, *args, **kwargs): pass
            def encode(self, x): import numpy as np; return np.zeros(24)
            def get_world_understanding(self): return {}
        stub.JEPAWorldModel = _DummyJEPA
        sys.modules['world_model'] = stub

import numpy as np
from agents import Agent


def test_yrs_to_ticks_modern():
    print("Test 1: _yrs_to_ticks in Modern (1 month/tick)")
    a = Agent(0, 0, rng=np.random.RandomState(0))
    a._era_time_scale_cache = 1.0 / 12.0  # Modern
    cases = [
        (15.0, 180),    # 15 yr * 12 = 180 ticks
        (6.0,  72),     # 6 yr * 12 = 72 ticks
        (60.0, 720),    # 60 yr * 12 = 720 ticks
        (80.0, 960),    # 80 yr * 12 = 960 ticks
        (666.0, 7992),  # 666 yr * 12 = 7992 ticks
        (8.0,  96),     # 8 yr * 12 = 96 ticks
    ]
    for years, expected in cases:
        got = a._yrs_to_ticks(years)
        ok = "OK" if got == expected else "FAIL"
        print(f"  [{ok}] _yrs_to_ticks({years:6.1f}) = {got:5d}  (expected {expected})")
        assert got == expected
    print("  PASS\n")


def test_yrs_to_ticks_paleolithic():
    print("Test 2: _yrs_to_ticks in Paleolithic (200 yr/tick)")
    a = Agent(0, 0, rng=np.random.RandomState(0))
    a._era_time_scale_cache = 200.0
    cases = [
        # All sub-200-year thresholds compress to 1 tick floor
        (15.0,  1),
        (6.0,   1),
        (60.0,  1),
        (80.0,  1),
        # Longer thresholds give plausible tick counts
        (666.0, 3),    # round(666/200) = 3
        (4000.0, 20),  # 4000/200 = 20
    ]
    for years, expected in cases:
        got = a._yrs_to_ticks(years)
        ok = "OK" if got == expected else "FAIL"
        print(f"  [{ok}] _yrs_to_ticks({years:6.1f}) = {got:5d}  (expected {expected})")
        assert got == expected, f"Expected {expected}, got {got}"
    print("  PASS\n")


def test_yrs_to_ticks_intermediate_eras():
    print("Test 3: _yrs_to_ticks in intermediate eras")
    a = Agent(0, 0, rng=np.random.RandomState(0))
    # Holocene (~50 yr/tick): 15-yr threshold should round to 0 -> floored to 1
    a._era_time_scale_cache = 50.0
    assert a._yrs_to_ticks(15.0) == 1
    assert a._yrs_to_ticks(60.0) == 1
    assert a._yrs_to_ticks(666.0) == 13   # round(666/50) = 13
    print(f"  Holocene 50yr/tick: 15yr->1, 60yr->1, 666yr->13 OK")

    # Agricultural (~10 yr/tick)
    a._era_time_scale_cache = 10.0
    assert a._yrs_to_ticks(15.0) == 2  # round(15/10) = 2 (note: round uses banker's rounding so 1.5->2)
    assert a._yrs_to_ticks(60.0) == 6
    assert a._yrs_to_ticks(666.0) == 67
    print(f"  Agricultural 10yr/tick: 15yr->2, 60yr->6, 666yr->67 OK")

    # Historical (1 yr/tick)
    a._era_time_scale_cache = 1.0
    assert a._yrs_to_ticks(15.0) == 15
    assert a._yrs_to_ticks(60.0) == 60
    print(f"  Historical 1yr/tick: 15yr->15, 60yr->60 OK")
    print("  PASS\n")


def test_default_cache_safe_before_update():
    print("Test 4: agent _yrs_to_ticks safe to call before first update()")
    a = Agent(0, 0, rng=np.random.RandomState(0))
    # Without ever calling update(), the cache should be Modern default
    assert a._era_time_scale_cache == 1.0 / 12.0
    assert a._yrs_to_ticks(15.0) == 180
    print(f"  Default cache: {a._era_time_scale_cache:.6f}, 15yr->{a._yrs_to_ticks(15.0)}")
    print("  PASS\n")


def test_modern_thresholds_match_intent():
    print("Test 5: Modern-era thresholds in real-world years")
    a = Agent(0, 0, rng=np.random.RandomState(0))
    a._era_time_scale_cache = 1.0 / 12.0  # Modern

    repro = a._yrs_to_ticks(a.REPRODUCE_MIN_AGE_YEARS)
    cooldown = a._yrs_to_ticks(a.REPRODUCE_COOLDOWN_YEARS)
    aging = a._yrs_to_ticks(a.AGING_THRESHOLD_YEARS)
    metab = a._yrs_to_ticks(a.METABOLISM_DOUBLE_YEARS)
    lifespan = a._yrs_to_ticks(a.LIFESPAN_NORM_YEARS)
    socialize = a._yrs_to_ticks(a.SOCIALIZE_MIN_AGE_YEARS)

    print(f"  Reproduktionsalter:    {repro:>5} ticks  ({repro/12:.1f} yr)")
    print(f"  Reproduktions-Cooldown:{cooldown:>5} ticks  ({cooldown/12:.1f} yr)")
    print(f"  Aging-Schwelle:        {aging:>5} ticks  ({aging/12:.1f} yr)")
    print(f"  Metabolism-Doppelung:  {metab:>5} ticks  ({metab/12:.1f} yr)")
    print(f"  Lebensspanne (norm):   {lifespan:>5} ticks  ({lifespan/12:.1f} yr)")
    print(f"  Sozialisierungsalter:  {socialize:>5} ticks  ({socialize/12:.1f} yr)")

    # Reproduktionsalter MUST be > 40 (the buggy old value)
    assert repro > 40, f"Reproduktionsalter zu niedrig: {repro}"
    # Reproduktionsalter MUST be > 12 yr (180 ticks) — biologisch realistisch
    assert repro >= 180, f"Reproduktionsalter sollte >=180 ticks sein, ist {repro}"
    # Aging-Schwelle sollte in plausibler Range bleiben (50-80 Jahre)
    assert 600 <= aging <= 960, f"Aging out of range: {aging}"
    print("  PASS\n")


def test_paleo_thresholds_compressed():
    print("Test 6: Paleolithic thresholds collapse to 1 tick (generation-per-tick)")
    a = Agent(0, 0, rng=np.random.RandomState(0))
    a._era_time_scale_cache = 200.0

    repro = a._yrs_to_ticks(a.REPRODUCE_MIN_AGE_YEARS)
    cooldown = a._yrs_to_ticks(a.REPRODUCE_COOLDOWN_YEARS)
    aging = a._yrs_to_ticks(a.AGING_THRESHOLD_YEARS)

    print(f"  Reproduktionsalter:    {repro} tick(s)")
    print(f"  Reproduktions-Cooldown:{cooldown} tick(s)")
    print(f"  Aging-Schwelle:        {aging} tick(s)")
    # Alle sub-200-yr Schwellen sollten auf 1 Tick gefloort werden
    assert repro == 1
    assert cooldown == 1
    assert aging == 1
    print("  PASS\n")


def test_modern_behavior_drift_acceptable():
    print("Test 7: Modern-Verhalten weicht <=15% von altem Code ab (außer Repro)")
    # Vergleiche neue Schwellen gegen alte hardcoded Werte
    a = Agent(0, 0, rng=np.random.RandomState(0))
    a._era_time_scale_cache = 1.0 / 12.0  # Modern

    # Cooldown war 80 ticks -> jetzt 72 (-10%)
    new_cooldown = a._yrs_to_ticks(a.REPRODUCE_COOLDOWN_YEARS)
    drift_cooldown = abs(new_cooldown - 80) / 80
    # Aging war 800 -> jetzt 720 (-10%)
    new_aging = a._yrs_to_ticks(a.AGING_THRESHOLD_YEARS)
    drift_aging = abs(new_aging - 800) / 800
    # Metab war 8000 -> jetzt 7992 (~0%)
    new_metab = a._yrs_to_ticks(a.METABOLISM_DOUBLE_YEARS)
    drift_metab = abs(new_metab - 8000) / 8000
    # Lifespan-Norm war 1000 -> jetzt 960 (-4%)
    new_lifespan = a._yrs_to_ticks(a.LIFESPAN_NORM_YEARS)
    drift_lifespan = abs(new_lifespan - 1000) / 1000
    # Socialize war 100 -> jetzt 96 (-4%)
    new_soc = a._yrs_to_ticks(a.SOCIALIZE_MIN_AGE_YEARS)
    drift_soc = abs(new_soc - 100) / 100

    print(f"  Cooldown: 80 -> {new_cooldown}  (drift {drift_cooldown:.1%})")
    print(f"  Aging:    800 -> {new_aging}  (drift {drift_aging:.1%})")
    print(f"  Metab:    8000 -> {new_metab}  (drift {drift_metab:.1%})")
    print(f"  Lifespan: 1000 -> {new_lifespan}  (drift {drift_lifespan:.1%})")
    print(f"  Soc:      100 -> {new_soc}  (drift {drift_soc:.1%})")
    for d in [drift_cooldown, drift_aging, drift_metab, drift_lifespan, drift_soc]:
        assert d <= 0.15, f"Drift too large: {d:.1%}"
    print("  PASS\n")


if __name__ == "__main__":
    test_yrs_to_ticks_modern()
    test_yrs_to_ticks_paleolithic()
    test_yrs_to_ticks_intermediate_eras()
    test_default_cache_safe_before_update()
    test_modern_thresholds_match_intent()
    test_paleo_thresholds_compressed()
    test_modern_behavior_drift_acceptable()
    print("=" * 60)
    print("ALL LIFECYCLE-SCALING TESTS PASSED")
    print("=" * 60)
