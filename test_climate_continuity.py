"""Climate continuity across the paleo -> Industrial-era macro handoff.

In a historical run the macro ODE is dormant through the paleo era. When it first
activates (Industrial onset, ~1750 CE) it must carry the paleoclimate state
forward rather than snapping to the present-day MacroState defaults
(+1.3 deg C / 425 ppm).
"""
from world import World


def test_macro_handoff_is_continuous():
    w = World(seed=1, scenario_id="historical")
    paleo = w.history.paleoclimate.get_climate(201)  # just before transition (1749 CE)

    w.history.year_bp = 199  # cross into the Industrial era
    w._hand_off_macro_from_paleo()
    s = w.macro.state

    # Temperature and CO2 carry over continuously (no jump to present-day).
    assert abs(s.temperature_anomaly - paleo["temperature_anomaly"]) < 0.05, (
        f"temperature jumped at handoff: {s.temperature_anomaly} vs paleo "
        f"{paleo['temperature_anomaly']}"
    )
    assert abs(s.co2_ppm - paleo["co2_ppm"]) < 1.0
    # Explicitly NOT the present-day MacroState defaults.
    assert s.temperature_anomaly < 0.5, "handoff left temperature at present-day level"
    assert s.co2_ppm < 320, "handoff left CO2 at present-day level"
    # Year handed off to the historical calendar, not 2025.
    assert 1700 < s.year < 1800


def test_handoff_happens_once():
    """The handoff flag flips so seeding cannot run twice."""
    w = World(seed=1, scenario_id="historical")
    assert w._macro_handed_off is False
    w.history.year_bp = 199
    w._hand_off_macro_from_paleo()
    w._macro_handed_off = True
    # present-day scenario is pre-seeded (no paleo handoff needed)
    wp = World(seed=1, scenario_id="present_day")
    assert wp._macro_handed_off is True
