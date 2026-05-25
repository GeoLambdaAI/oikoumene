"""
Standalone validation tests for MacroModel.

Anchors validation against IPCC AR6 SSP scenarios and observed Mauna Loa
trends rather than the wide permissive ranges of v0.1.

Reference scenarios (IPCC AR6 WG1 Table 4.5, projected 2081-2100):
    SSP1-2.6  (low):     445 ppm,  +1.8 deg C,  ~0.44 m SLR
    SSP2-4.5  (mid):     603 ppm,  +2.7 deg C,  ~0.55 m SLR
    SSP3-7.0  (high):    867 ppm,  +3.6 deg C,  ~0.71 m SLR
    SSP5-8.5  (extreme): 1135 ppm, +4.4 deg C,  ~0.84 m SLR

The model's "no agent feedback" run represents BAU with technology and
renewable growth but no aggressive mitigation. We anchor to the
SSP2-4.5 to SSP3-7.0 range as a reasonable BAU envelope.

Mauna Loa observed CO2 growth rate (NOAA GML, decadal mean 2014-2024):
    2.4-3.0 ppm/yr  -- the most stringent test, since this anchors the
    carbon-cycle calibration.
"""

import sys
sys.path.insert(0, '.')
from macro import MacroModel


def run_bau_scenario(verbose: bool = True):
    """Run BAU scenario from 2025 to 2100, return validation success."""
    model = MacroModel(config={"dt_years": 1.0 / 12.0})
    bau_feedback = {}

    if verbose:
        print("=" * 90)
        print("MACRO MODEL BAU SCENARIO: 2025 -> 2100")
        print("=" * 90)
        header = (f"{'Year':>6} {'CO2':>7} {'dCO2/yr':>8} {'T':>5} {'SLR':>6} "
                  f"{'Foss':>6} {'Pop(B)':>7} {'GDP':>6} {'Tens':>6} {'Renew':>6}")
        print(header)
        print("-" * len(header))

    # Track 1-year CO2 growth rate near 2030 (baseline anchor)
    co2_2029 = None
    co2_2030 = None

    target_years = list(range(2025, 2101, 5))
    prev_co2 = model.state.co2_ppm
    prev_year = model.state.year
    step_count = 0

    while model.state.year < 2100.5:
        model.step(bau_feedback)
        step_count += 1

        # Sample around 2030 to compute precise 1-yr growth rate
        if abs(model.state.year - 2029.0) < 0.05 and co2_2029 is None:
            co2_2029 = model.state.co2_ppm
        if abs(model.state.year - 2030.0) < 0.05 and co2_2030 is None:
            co2_2030 = model.state.co2_ppm

        year_rounded = round(model.state.year)
        if year_rounded in target_years and abs(model.state.year - year_rounded) < 0.05:
            s = model.state
            years_elapsed = s.year - prev_year
            dco2_per_yr = (s.co2_ppm - prev_co2) / years_elapsed if years_elapsed > 0 else 0
            prev_co2 = s.co2_ppm
            prev_year = s.year
            if verbose:
                print(f"{s.year:6.1f} {s.co2_ppm:7.1f} {dco2_per_yr:8.2f} "
                      f"{s.temperature_anomaly:5.2f} {s.sea_level_rise_m:6.3f} "
                      f"{s.fossil_fuels:6.3f} {s.global_population_billions:7.2f} "
                      f"{s.global_gdp_index:6.3f} {s.social_tension:6.3f} "
                      f"{s.renewable_fraction:6.3f}")
            target_years.remove(year_rounded)

    if verbose:
        print(f"\nTotal ODE steps: {step_count}")

    # ----- VALIDATION -----
    s = model.state
    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION (IPCC AR6 SSP2-4.5 to SSP3-7.0 envelope)")
        print("=" * 70)

    checks = []

    # Anchor 1: CO2 growth rate near 2030 ~ Mauna Loa observation
    # NOAA GML decadal mean 2014-2024: 2.4-3.0 ppm/yr
    # This is the strictest test of the carbon cycle.
    if co2_2029 and co2_2030:
        dco2_anchor = co2_2030 - co2_2029
        ok = 2.0 <= dco2_anchor <= 3.5
        checks.append(("CO2 growth rate ~2030", ok, f"{dco2_anchor:.2f} ppm/yr",
                       "2.0-3.5 (Mauna Loa decadal: 2.4-3.0)"))

    # Anchor 2: CO2 in 2100 between SSP2-4.5 and SSP3-7.0
    ok = 600 <= s.co2_ppm <= 800
    checks.append(("CO2 2100", ok, f"{s.co2_ppm:.0f} ppm",
                   "600-800 (SSP2-4.5 to SSP3-7.0)"))

    # Anchor 3: Temperature anomaly 2100 between SSP2-4.5 and SSP3-7.0
    ok = 2.4 <= s.temperature_anomaly <= 3.8
    checks.append(("Temperature 2100", ok, f"+{s.temperature_anomaly:.2f} degC",
                   "+2.4 to +3.8 (SSP2-4.5 to SSP3-7.0)"))

    # Anchor 4: Sea level rise 2100 within SSP2-4.5 to SSP3-7.0 likely range
    # Note: simple thermal-expansion proxy; true ice-sheet dynamics not modelled
    ok = 0.40 <= s.sea_level_rise_m <= 0.85
    checks.append(("Sea level rise 2100", ok, f"{s.sea_level_rise_m:.2f} m",
                   "0.40-0.85 (SSP2-4.5 to SSP3-7.0 likely)"))

    # Anchor 5: Population 2100 — UN WPP medium ~10.4B, Vollset 2020 ~8.8B
    # The model's lower endogenous population reflects strong demographic
    # transition with rising tech. Allow either projection.
    ok = 7.5 <= s.global_population_billions <= 11.0
    checks.append(("Population 2100", ok,
                   f"{s.global_population_billions:.2f} B",
                   "7.5-11.0 (Vollset 2020 to UN WPP 2024)"))

    # Anchor 6: Fossil fuels declined significantly but not exhausted
    ok = 0.20 <= s.fossil_fuels <= 0.65
    checks.append(("Fossil fuels remaining 2100", ok, f"{s.fossil_fuels:.3f}",
                   "0.20-0.65 (significant depletion)"))

    # Anchor 7: Renewable fraction substantial
    ok = s.renewable_fraction > 0.50
    checks.append(("Renewable fraction 2100", ok,
                   f"{s.renewable_fraction:.3f}", ">0.50"))

    # Anchor 8: Social tension rises (consistent with Earth4All trajectory)
    ok = s.social_tension > 0.40
    checks.append(("Social tension 2100", ok, f"{s.social_tension:.3f}", ">0.40"))

    # Anchor 9: Technology grew substantially
    ok = s.technology_level > 1.8
    checks.append(("Technology level 2100", ok,
                   f"{s.technology_level:.3f}", ">1.8"))

    if verbose:
        for label, ok, value, expected in checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {label:<30s} {value:<20s} (expect {expected})")

    passed = sum(1 for _, ok, _, _ in checks if ok)
    total = len(checks)

    if verbose:
        print(f"\nResults: {passed}/{total} checks passed")
        if passed == total:
            print("ALL VALIDATIONS PASSED — model behavior consistent with IPCC AR6 BAU envelope")
        else:
            print(f"WARNING: {total - passed} checks failed")

    return passed == total


def test_climate_sensitivity_consistency():
    """Verify that emergent ECS matches the declared CLIMATE_SENSITIVITY."""
    import numpy as np
    print("\n" + "=" * 70)
    print("UNIT TEST: emergent ECS == declared CLIMATE_SENSITIVITY")
    print("=" * 70)
    F_2x = MacroModel.FORCING_COEFF * np.log(2.0)
    ECS_emergent = F_2x / MacroModel.CLIMATE_FEEDBACK
    ECS_declared = MacroModel.CLIMATE_SENSITIVITY
    print(f"  F_2xCO2 (Myhre 1998):   {F_2x:.3f} W/m^2")
    print(f"  CLIMATE_FEEDBACK:       {MacroModel.CLIMATE_FEEDBACK} W/m^2/degC")
    print(f"  Emergent ECS:           {ECS_emergent:.3f} degC")
    print(f"  Declared CLIMATE_SENS:  {ECS_declared} degC")
    drift = abs(ECS_emergent - ECS_declared) / ECS_declared
    print(f"  Drift:                  {drift:.1%}")
    ok = drift < 0.02  # within 2%
    print(f"  {'PASS' if ok else 'FAIL'}: emergent and declared ECS within 2%")
    assert ok, f"emergent ECS {ECS_emergent:.3f} vs declared {ECS_declared} drifts {drift:.1%}"


def test_carbon_cycle_unit_anchor():
    """Smoke test: at typical 2025 emissions, dCO2/dt should be near observed."""
    print("\n" + "=" * 70)
    print("UNIT TEST: carbon cycle produces observed Mauna Loa rate")
    print("=" * 70)
    model = MacroModel(config={"dt_years": 1.0 / 12.0})
    # Force one ODE step with default feedback at startwerten
    y0 = model._state_to_vector()
    dy = model._ode_system(0, y0, {})
    dco2_per_yr = dy[0]  # _IDX["co2"]
    print(f"  Modelled dCO2/dt at 2025 startwerten: {dco2_per_yr:.2f} ppm/yr")
    print(f"  Mauna Loa 2014-2024 decadal mean:    2.4-3.0 ppm/yr")
    ok = 2.0 <= dco2_per_yr <= 3.5
    print(f"  {'PASS' if ok else 'FAIL'}: within 2.0-3.5 envelope")
    assert ok, f"dCO2/dt {dco2_per_yr:.2f} ppm/yr outside 2.0-3.5 envelope"


if __name__ == "__main__":
    success = run_bau_scenario()
    unit_ok = True
    for _fn in (test_climate_sensitivity_consistency, test_carbon_cycle_unit_anchor):
        try:
            _fn()
        except AssertionError as exc:
            print(f"  FAILED: {exc}")
            unit_ok = False
    print("\n" + "=" * 70)
    overall = success and unit_ok
    print(f"OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    sys.exit(0 if overall else 1)
