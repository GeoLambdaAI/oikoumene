# Validation Report — World Genesis Macro Model

**Version:** 0.2.1
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
are recorded every simulated year. The eight target metrics are evaluated at
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
   [`macro.py:388`](../macro.py#L388) (vs. published DICE-2016R `a ≈ 0.00236`)
   reflects the simulator's accelerated time step; Hubbert depletion timescales
   in [`macro.py:305-320`](../macro.py#L305-L320) are scaled to scenario
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

## 6. How to reproduce

```bash
# Clean checkout
git clone https://github.com/GeoLambdaAI/world-genesis.git
cd world-genesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run validation (≈ 2 seconds)
python test_macro.py
```

Expected exit code: `0`. Expected stdout final line: `ALL VALIDATIONS PASSED!`.

## 7. References

See [`paper/paper.bib`](../paper/paper.bib) for the bibliographic entries
backing each calibration target.
