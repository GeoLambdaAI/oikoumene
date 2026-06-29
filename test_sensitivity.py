"""
Smoke test for scripts/sensitivity.py — the SALib Sobol pipeline.

We run with a deliberately small N (N=2, ~36 model evaluations, ~10 s on
this hardware) and verify only the structural contract: the script returns
S1 and ST arrays of the right shape for each output, the JSON artefact is
written, and the MacroModel parameter overrides actually change the BAU
outputs (not a no-op).

This is NOT a numerical validation of the indices — at N=2 the sample is
nowhere near large enough for stable Sobol estimates. The "real" analysis
for `docs/validation.md` is run separately with N=64 or larger via
    python scripts/sensitivity.py --n 64

The test skips automatically when SALib is not installed (its install
requires the optional `[sensitivity]` extra).
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("SALib")

# Make the scripts/ folder importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from sensitivity import (  # noqa: E402  (after sys.path tweak)
    OUTPUT_NAMES,
    PARAMETERS,
    RESULTS_DIR,
    build_problem,
    main,
    run_bau_once,
)

EXPECTED_PARAM_NAMES = {
    "FORCING_COEFF", "CLIMATE_FEEDBACK", "OCEAN_HEAT_CAPACITY",
    "DEEP_OCEAN_COUPLING", "NATURAL_ABSORPTION_RATE",
    "ABSORPTION_TEMP_SENSITIVITY", "BASE_EMISSION_RATE", "POP_GROWTH_BASE",
}


def test_problem_spec_matches_expected_parameters():
    """The Sobol problem dict must enumerate the eight macro parameters
    we are committed to in docs/validation.md."""
    problem = build_problem()
    assert problem["num_vars"] == len(EXPECTED_PARAM_NAMES)
    assert set(problem["names"]) == EXPECTED_PARAM_NAMES
    # bounds are 2-element lists with lo < hi
    for name, (lo, hi) in zip(problem["names"], problem["bounds"]):
        assert lo < hi, f"degenerate bounds for {name}: [{lo}, {hi}]"


def test_macro_parameter_overrides_actually_change_2100_outputs():
    """Sanity check that instance-level overrides hit the integrator.

    If `setattr(model, 'FORCING_COEFF', X)` were silently ignored, the
    Sobol analysis would be meaningless. Run the BAU once at the low end
    and once at the high end of the CLIMATE_FEEDBACK prior and assert
    that 2100 temperature meaningfully differs.
    """
    # Lambda is the lever with the most direct effect on 2100 ΔT.
    lo = dict(CLIMATE_FEEDBACK=0.80)
    hi = dict(CLIMATE_FEEDBACK=1.60)
    T_lo, _, _ = run_bau_once(lo)
    T_hi, _, _ = run_bau_once(hi)
    assert T_lo > T_hi + 0.5, (
        f"CLIMATE_FEEDBACK override did not propagate: "
        f"T(lambda={lo['CLIMATE_FEEDBACK']:.2f})={T_lo:.2f} vs "
        f"T(lambda={hi['CLIMATE_FEEDBACK']:.2f})={T_hi:.2f} -- expected "
        "lower lambda to give noticeably warmer 2100 (lambda^-1 sensitivity)."
    )


def test_sensitivity_pipeline_runs_and_writes_json(tmp_path, capsys):
    """End-to-end smoke: tiny N=2 SALib run completes and writes the
    JSON artefact with the expected shape."""
    # Run the CLI with N=2 (36 model evaluations, ~10 s).
    rc = main(["--n", "2", "--quiet", "--seed", "7"])
    assert rc == 0, "sensitivity main() returned non-zero"

    out_path = RESULTS_DIR / "sensitivity_N2.json"
    assert out_path.exists(), f"expected results file not produced: {out_path}"

    payload = json.loads(out_path.read_text())
    n_params = len(PARAMETERS)
    assert payload["n_samples"] == 2
    assert payload["total_runs"] == 2 * (2 * n_params + 2)
    assert list(payload["outputs"]) == list(OUTPUT_NAMES)
    for out in OUTPUT_NAMES:
        idx = payload["indices"][out]
        for key in ("S1", "S1_conf", "ST", "ST_conf"):
            assert key in idx, f"missing {key} for {out}"
            assert len(idx[key]) == n_params, (
                f"{key} for {out} has length {len(idx[key])}, "
                f"expected {n_params}"
            )

    # Sanity check that the BAU outputs are not all identical across the
    # 36 evaluations (would indicate the parameters never actually varied
    # or every run produced the same numbers).
    means = {out: payload["output_means_and_stds"][out]["mean"]
             for out in OUTPUT_NAMES}
    stds = {out: payload["output_means_and_stds"][out]["std"]
            for out in OUTPUT_NAMES}
    assert all(s > 0.01 for s in stds.values()), (
        f"output stds suspiciously low across the Saltelli sample: {stds}"
    )
    print(f"  N=2 smoke: T_2100 ~ {means['T_2100_degC']:.2f}±{stds['T_2100_degC']:.2f}, "
          f"CO2 ~ {means['CO2_2100_ppm']:.0f}±{stds['CO2_2100_ppm']:.0f}")
