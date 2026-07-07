# Validation Report — Oikoumene Macro Model

**Version:** 0.3.0
**Date of validation run:** 2026-05-06
**Seed:** deterministic (no stochastic terms in BAU macro path)
**Reproduction:** `python test_macro.py`

---

## 1. Scope of validation

This document records the calibration of the 14-state macro ODE
([`macro.py`](../macro.py)) against published reference values for a
**Business-As-Usual (BAU) scenario** running 2025 → 2100 with monthly
integration steps (`dt_years = 1/12`, ~906 ODE steps).

Validation targets the equations ultimately derived from:

- IPCC AR6 WG1 (2021) — climate response and forcing.
- Friedlingstein et al. (2024) — Global Carbon Budget; airborne fraction.
- Meadows et al. (1972, 2004) — World3 population/resource feedbacks.
- Nordhaus DICE (2017) — damage function `D = a·T²` and GDP coupling.
- Earth4All (Dixson-Decleve et al. 2022) — social tension and renewable transition.
- Hubbert (1956) — peak-and-decline resource curves.

The numbers in §3 are produced by [`test_macro.py`](../test_macro.py) and
are bit-reproducible from a clean checkout. v0.2.0 corrected six bugs in
the v0.1.0 calibration that produced the previously-quoted CO₂ = 504.8 ppm
trajectory; v0.2.1 left the standalone `test_macro.py` calibration path
unchanged (its `MacroModel(dt_years = 1/12)` instantiation matches the
calibrated-per-step path), so the numbers below reflect the v0.2.0
calibration as it now lives.

## 2. Methodology

The macro module is run **without agent feedback** (`bau_feedback = {}`),
isolating the closed-form ODE behaviour from the agent layer. State variables
are recorded every simulated year. The nine target metrics are evaluated at
year 2100 against tolerance bands chosen to span the IPCC AR6
SSP2-4.5 → SSP3-7.0 plausibility envelope. The BAU path falls inside this
envelope because renewable transition is endogenous and Hubbert depletion
limits late-century fossil burn.

## 3. Results — 2100 state vs. target bands

| # | Metric | Simulated 2100 | Target band | IPCC reference (SSP) | Status |
|---|---|---:|---:|---|:---:|
| 1 | CO₂ concentration | **679.4 ppm** | 600 – 800 ppm | SSP1-2.6: 445 · SSP2-4.5: 603 · SSP3-7.0: 867 · SSP5-8.5: 1135 | PASS |
| 2 | Temperature anomaly (vs. pre-industrial) | **+2.74 °C** | +2.4 – +3.8 °C | SSP1-2.6: +1.8 · SSP2-4.5: +2.7 · SSP3-7.0: +3.6 · SSP5-8.5: +4.4 | PASS |
| 3 | Global population | **8.37 B** | 7.5 – 11.0 B | Vollset (2020) low: 8.8 B; UN WPP 2024 median: 10.4 B | PASS |
| 4 | Fossil fuels remaining (fraction) | **0.327** | 0.20 – 0.65 | Hubbert post-peak; IEA Net-Zero 2050 ~0.2 | PASS |
| 5 | Social tension index | **0.624** | > 0.40 | Earth4All "Too Little Too Late" range | PASS |
| 6 | Sea level rise (above 2000) | **0.611 m** | 0.40 – 0.85 m | IPCC AR6 SSP2-4.5: 0.32–0.62 m · SSP3-7.0 likely: 0.46–0.71 m | PASS |
| 7 | Renewable energy fraction | **0.744** | > 0.50 | IEA STEPS 2050 ~0.4; APS 2050 ~0.7 | PASS |
| 8 | Technology multiplier | **2.184** | > 1.8 | Romer (1990) endogenous growth; ceiling 5.0 | PASS |
| — | dCO₂/dt around 2030 (anchor) | **2.58 ppm/yr** | 2.0 – 3.5 ppm/yr | NOAA GML Mauna Loa decadal mean 2014–2024: 2.4–3.0 ppm/yr | PASS |
| — | Emergent ECS vs. declared | **3.00 °C** | drift < 2 % | F<sub>2x</sub>/λ = 5.35·ln 2 / 1.236 = 3.00 °C exactly | PASS |

**Overall: 9 / 9 SSP-envelope checks plus 2 unit anchors pass.**

> **What this does and does not show.** These are *calibration-consistency*
> checks: the macro coefficients were tuned so the BAU path lands inside the
> SSP2-4.5 → SSP3-7.0 plausibility envelope, and the bands are deliberately wide
> (they span multiple SSPs). Passing therefore demonstrates that the model is
> internally consistent and qualitatively faithful to the IPCC AR6 envelope —
> **not** independent out-of-sample predictive skill. The genuinely constraining
> anchors are the dCO₂/dt ≈ 2030 rate and the emergent-ECS consistency check.
> See §5 for the explicit calibration caveats.

## 4. Trajectory excerpts (5-year increments)

| Year |  CO₂  | Temp |  SLR  | Fossil | Pop (B) |  GDP | Renew | Tech |
|---:  |---:   |---:  |---:   |---:    |---:     |---:  |---:   |---:  |
| 2065 | 544.0 | 1.98 | 0.396 | 0.524  | 8.40    | 2.84 | 0.44  | 1.56 |
| 2070 | 561.7 | 2.09 | 0.424 | 0.490  | 8.41    | 3.28 | 0.48  | 1.65 |
| 2075 | 579.9 | 2.19 | 0.452 | 0.458  | 8.41    | 3.80 | 0.53  | 1.73 |
| 2080 | 598.6 | 2.30 | 0.481 | 0.428  | 8.40    | 4.42 | 0.57  | 1.82 |
| 2085 | 617.8 | 2.41 | 0.511 | 0.400  | 8.40    | 5.14 | 0.62  | 1.90 |
| 2090 | 637.4 | 2.51 | 0.543 | 0.374  | 8.39    | 6.00 | 0.66  | 1.99 |
| 2095 | 657.3 | 2.62 | 0.575 | 0.350  | 8.38    | 7.01 | 0.70  | 2.08 |
| 2100 | 677.4 | 2.73 | 0.608 | 0.329  | 8.37    | 8.21 | 0.74  | 2.18 |

Full per-year trace is reproducible via `python test_macro.py | tee logs/validation.log`.

## 5. Known limitations

1. **Single deterministic path.** The BAU run has no stochastic terms; results
   are bit-exact across runs. Sensitivity analysis (Sobol indices on
   climate sensitivity, tech ceiling, ECS) is planned for v0.2.
2. **No regional disaggregation.** The macro layer is global; regional
   heterogeneity (e.g., per-country emissions) emerges only via the agent
   and geopolitics layers.
3. **Damage function.** DICE-style quadratic damage (`D = a·T²`) — known to
   under-weight tail risks above +3 °C (Stern 2022 critique). Future
   versions should compare with Burke-Hsiang-Miguel 2015 specifications.
4. **Tech transition single-scalar.** The `technology_level` ∈ [0.5, 5.0] does
   not distinguish between clean and dirty innovation paths. Acemoglu et al.
   (2012) directed-technical-change extension is a v0.2 candidate.
5. **Equation form vs. calibration constants.** Every equation in the codebase
   reproduces the *published functional form* of its source paper. Calibration
   constants, however, are tuned to the simulator's tick rate (months to
   centuries depending on era), latent population scale (hundreds to thousands
   of agents representing billions of humans), and grid resolution (0.5°).
   Concrete examples: the DICE damage coefficient `a = 0.01` in
   [`macro.py:407`](../macro.py#L407) (vs. published DICE-2016R `a ≈ 0.00236`)
   reflects the simulator's accelerated time step; Hubbert depletion timescales
   in [`macro.py:324`](../macro.py#L324) are scaled to scenario
   duration rather than transcribed from Hubbert (1956). Reviewers should
   treat the simulator as a *qualitatively faithful* implementation rather
   than a coefficient-by-coefficient replica.
6. **JEPA scale.** The implementation in
   [`world_model.py`](../world_model.py) faithfully realizes the
   encoder + AdaLN predictor + SIGReg + CEM-planner architecture from
   LeCun (2022) and Maes et al. (2026), but at small scale
   (`latent_dim = 24` in `SharedWorldModel` defaults; `hidden_dim = 48`),
   versus millions of parameters in the published papers. v0.2.0 replaced
   v0.1's directional finite-difference gradient estimator with hand-written
   analytic back-propagation in pure NumPy, gradient-checked against central
   finite differences to <1e-8 relative error
   ([`test_world_model_gradcheck.py`](../test_world_model_gradcheck.py));
   v0.2.1 made the training-batch sampling deterministic via a per-instance
   `RandomState`, removing global-RNG coupling. v0.3.0 adds an opt-in PyTorch
   backend ([`world_model_torch.py`](../world_model_torch.py)) with autograd
   and optional CUDA for scaling `latent_dim` up; it reproduces the NumPy
   model at default settings (cross-backend encode/predict agree to <1e-4)
   and exposes the paper's Epps–Pulley SIGReg as an opt-in toggle.
7. **Conceptual references vs. quantitative ones.** Diamond (1997),
   Dawkins (2009), Stringer (2012), and Marshak (2019) are cited at the
   *structural* level — the simulator implements the continental-axis
   multiplier, trait inheritance with mutation, Out-of-Africa migration
   timing, and geological resource provinces — but these works do not publish
   closed-form equations to transcribe. Earth4All / Dixson-Decleve et al.
   (2022) is similarly partial: the social tension model takes the structural
   form `f(inequality, food insecurity, env stress, expectation gap)` but the
   exact published coefficients are not fully available, so the calibration
   in [`macro.py:192-200`](../macro.py#L192-L200) is bespoke.

## 6. Parameter sensitivity (Sobol decomposition)

A global, variance-based sensitivity analysis ([Sobol 1993](
https://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804); Saltelli
et al. 2010) was run on the 2100 BAU outputs to identify which model
parameters dominate the uncertainty of the headline projections. The
analysis varies **eight parameters** within IPCC AR6 / Friedlingstein 2024 /
UN WPP literature ranges (see priors below) and computes first-order (S1)
and total (ST) Sobol indices via SALib's Saltelli sampler.

Run via [`scripts/sensitivity.py`](../scripts/sensitivity.py)
(requires `pip install -e ".[sensitivity]"`).

### 6.1 Parameter priors (flat U(lo, hi))

| Parameter | Bounds | Source |
|---|---|---|
| `FORCING_COEFF` (F<sub>2x</sub>) | 5.00–5.70 W m<sup>-2</sup> | Myhre 1998 1-σ |
| `CLIMATE_FEEDBACK` (λ) | 0.80–1.60 W m<sup>-2</sup> K<sup>-1</sup> | IPCC AR6 likely (gives ECS ≈ 2.3–4.6 °C) |
| `OCEAN_HEAT_CAPACITY` | 5.0–10.0 W·yr m<sup>-2</sup> K<sup>-1</sup> | Held et al. 2010 |
| `DEEP_OCEAN_COUPLING` | 0.50–1.00 W m<sup>-2</sup> K<sup>-1</sup> | Gregory 2000 |
| `NATURAL_ABSORPTION_RATE` | 0.40–0.60 | Friedlingstein 2024 decadal |
| `ABSORPTION_TEMP_SENSITIVITY` | 0.03–0.10 K<sup>-1</sup> | sink-weakening uncertainty |
| `BASE_EMISSION_RATE` | 38.0–46.0 GtCO<sub>2</sub>/yr | GCP 2024 ± 10 % |
| `POP_GROWTH_BASE` | 0.005–0.013 yr<sup>-1</sup> | UN WPP envelope |

### 6.2 Sobol indices (N = 64, 1 152 BAU evaluations)

| Parameter | T<sub>2100</sub> S1 | T<sub>2100</sub> ST | CO₂<sub>2100</sub> S1 | CO₂<sub>2100</sub> ST | Pop<sub>2100</sub> S1 | Pop<sub>2100</sub> ST |
|---|---|---|---|---|---|---|
| `FORCING_COEFF` | +0.04 | +0.06 | -0.00 | +0.00 | +0.02 | +0.06 |
| **`CLIMATE_FEEDBACK`** | **+0.82** | **+0.79** | +0.01 | +0.01 | **+0.58** | **+0.67** |
| `OCEAN_HEAT_CAPACITY` | +0.00 | +0.00 | -0.00 | +0.00 | +0.00 | +0.00 |
| `DEEP_OCEAN_COUPLING` | +0.05 | +0.07 | +0.00 | +0.00 | +0.07 | +0.10 |
| **`NATURAL_ABSORPTION_RATE`** | +0.05 | +0.05 | **+0.62** | **+0.60** | +0.04 | +0.03 |
| `ABSORPTION_TEMP_SENSITIVITY` | +0.01 | +0.00 | +0.04 | +0.03 | +0.00 | +0.00 |
| **`BASE_EMISSION_RATE`** | +0.01 | +0.01 | **+0.19** | **+0.17** | -0.01 | +0.01 |
| `POP_GROWTH_BASE` | +0.00 | +0.00 | +0.00 | +0.00 | +0.06 | **+0.20** |

S1 = first-order index (variance share explained by the parameter alone).
ST = total index (S1 + interaction effects). Bold entries indicate parameters
that explain ≥10 % of the variance in the corresponding output. Negative-
but-near-zero values reflect Monte-Carlo noise at this N; treat |S1| < 0.05
as not significant. Full results with 95 % confidence intervals in
[`scripts/results/sensitivity_N64.json`](../scripts/results/sensitivity_N64.json).

### 6.3 What this tells us

- **2100 temperature is overwhelmingly driven by the climate feedback
  parameter λ** (ST ≈ 0.79). Forcing-coefficient and deep-ocean-coupling
  uncertainty add only ~6 % each; everything else is in the noise. This is
  consistent with IAM literature where ECS dominates end-of-century ΔT
  uncertainty.
- **2100 CO₂ is driven mainly by the natural sink fraction** (ST ≈ 0.60)
  with baseline emissions a clear secondary (ST ≈ 0.17). Together these
  explain ~80 % of the projected CO₂ variance; sink weakening with warming
  contributes a further ~3 %. λ does *not* directly drive 2100 CO₂ — the
  carbon cycle and the energy balance are largely decoupled at this scale
  in the macro layer.
- **2100 population variance is dominated by climate feedback** (ST ≈ 0.67)
  with the population-growth rate a smaller direct contributor (S1 ≈ 0.06)
  but a substantial interaction contributor (ST ≈ 0.20, i.e. its effect is
  conditioned on the realised climate). This is the DICE-damage channel:
  warmer worlds reduce GDP and demography downstream.
- **`OCEAN_HEAT_CAPACITY` and `ABSORPTION_TEMP_SENSITIVITY` are essentially
  inactive at the 2100 horizon.** They could likely be fixed at their
  central values without measurable loss of fidelity, simplifying any
  future Bayesian-calibration / SBI sweep.

### 6.4 Caveats

- N = 64 (1 152 evaluations) gives stable rank ordering but the second-
  decimal of each index is still noisy (typical 95 % CI half-width ~0.05
  for the dominant terms; see the JSON). For publication-grade indices
  rerun with `--n 256` (≈ 18 min).
- Priors are flat over the published 1-σ / likely ranges. A more
  defensible prior set would use the IPCC AR6 ECS posterior shape (skewed,
  long upper tail) for `CLIMATE_FEEDBACK`; this is the natural next step
  via the SBI pipeline discussed in §5.
- The parameter ranges treat F<sub>2x</sub> and λ as independent, while in
  practice they are correlated through ECS observations. A Sobol design
  with the constraint `ECS = F_2x ln 2 / λ ∈ [2.5, 4.5]` would tighten the
  joint range.

### 6.5 How to reproduce the sensitivity run

```bash
pip install -e ".[sensitivity]"
python scripts/sensitivity.py --n 64      # ~5 min on commodity CPU
# results: scripts/results/sensitivity_N64.json + stdout markdown table
```

## 7. How to reproduce

```bash
# Clean checkout
git clone https://github.com/GeoLambdaAI/oikoumene.git
cd oikoumene
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run validation (≈ 2 seconds)
python test_macro.py
```

Expected exit code: `0`. Expected stdout final line: `OVERALL: ALL PASS`
(preceded by `ALL VALIDATIONS PASSED`). The same checks also run under
`pytest` via `test_bau_scenario_ipcc_envelope`.

## 8. References

See [`paper/paper.bib`](../paper/paper.bib) for the bibliographic entries
backing each calibration target.
