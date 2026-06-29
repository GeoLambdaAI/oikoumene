# Development Log

A running, sectioned record of design decisions, verified facts, and
follow-ups across substantive development sessions. Use this to pick up
where the previous session left off without re-deriving rationale or
re-verifying dataset licences.

For the user-facing release history see [`CHANGELOG.md`](../CHANGELOG.md).
For the empirical-data sources and licences see
[`data_attributions.md`](data_attributions.md). For macro-model validation
see [`validation.md`](validation.md). This file complements those.

---

## Arc: v0.3.0 → v0.4 scaffolding (sessions of 2026-05-23, 2026-05-25, 2026-06-21, 2026-06-22)

Topic span: PyTorch JEPA backend; dashboard backend-swap knob; full
upload-readiness audit and v0.3.0 release; empirical-data ingestion
scaffolding (CHELSA, SoilGrids, HYDE, HadCRUT5, ETOPO, UCDP-GED, NE 10 m
rivers); Sobol sensitivity analysis. Roughly two weeks of focused work.

### What shipped as v0.3.0 (2026-05-25)

- **PyTorch JEPA backend** ([`world_model_torch.py`](../world_model_torch.py)) —
  autograd implementation paper-faithful to LeWorldModel (Maes et al. 2026,
  arXiv:2603.19312). Repo-default settings reproduce the NumPy backend so
  weight-copy cross-check passes (<1 e-4). Opt-in toggles: Epps–Pulley
  SIGReg (paper-aligned), predictor dropout, `device=auto` for CUDA.
- **Dashboard JEPA backend knob** ([`templates/index.html`](../templates/index.html),
  [`app.py`](../app.py), [`world.py`](../world.py)) — runtime backend swap
  (NumPy / PyTorch × Repo-default / Paper × device). Swap preserves the
  shared experience buffer and repoints every agent. Graceful degradation
  when torch is absent.
- **Restored / created the tests the README claimed to have**:
  - `test_world_model.py` — JEPA learning behaviour (prediction-loss
    reduction, learned action conditioning, anti-collapse, linear probe,
    CEM validity).
  - `test_world_model_gradcheck.py` — finite-difference gradient checks
    on every analytic backward pass; measured relative error <1e-8.
  - `test_world_model_torch.py` — torch parity, NumPy↔torch weight
    cross-check, Epps–Pulley toggle, device handling.
- **Doc reconciliation**: README test table, line counts, gradcheck wording
  ("<1e-10" → realistic "<1e-8"), file table; `validation.md` PyTorch claim
  rewritten from "future work" to "implemented".
- **Repo hygiene**: `.gitignore`, `.github/workflows/test.yml` (CI on
  3.11/3.12/3.13), version aligned to 0.3.0 across `pyproject.toml` +
  `CITATION.cff` + `CHANGELOG.md`.
- **Upload**: pushed to `https://github.com/GeoLambdaAI/world-genesis` after
  some PAT-vs-organisation-policy debugging (classic token with `repo` scope
  worked; SSH would also work; fine-grained PAT needed org Resource-Owner).

### Empirical data scaffolding (in `[Unreleased]`, targeting v0.4)

[`generate_empirical_inputs.py`](../generate_empirical_inputs.py) is a new
top-level downloader that fetches authoritative open datasets and
resamples / processes them onto the simulator's 0.5° grid. **Outputs land
under `data/empirical/` and DO NOT yet flow into the runtime simulator** —
wiring `world.py` to read these instead of the synthesized rasters is the
explicit v0.4 milestone.

Seven datasets are scaffolded, all licence-verified:

| # | Dataset | Licence | Verified URL / archive | Output |
|---|---|---|---|---|
| 1 | CHELSA V2.1 | **CC0 1.0** (EnviDat DOI 10.16904/envidat.228) | `https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/climatologies/<var>/1981-2010/CHELSA_<var>_<MM>_1981-2010_V.2.1.tif` | `data/empirical/chelsa_<var>_annual.npy` |
| 2 | SoilGrids 2.0 | **CC-BY 4.0** (ISRIC) | `https://files.isric.org/soilgrids/latest/data/<prop>/<prop>_<depth>_<stat>.vrt` | `data/empirical/soilgrids_fertility.npy` (composite) |
| 3 | HYDE 3.2.1 | **CC0 1.0** (DANS DOI 10.17026/DANS-25G-GEZ3) | Dataverse API; `baseline.zip` fileId `5490328` (≈ 5.3 GB) | `data/empirical/hyde_{popc,cropland,grazing}/<year>.npy` |
| 4 | HadCRUT.5.1.0.0 | **OGL v3.0** (UK Crown) | `https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/analysis/diagnostics/HadCRUT.5.1.0.0.analysis.component_series.global.annual.csv` | `data/empirical/hadcrut5_global_annual.npy` (structured) |
| 5 | ETOPO 2022 60s | **US Gov public domain** (verified in User Guide §4) | `https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc` (≈ 395 MB) | `data/empirical/etopo_elevation.npy` |
| 6 | UCDP-GED v25.1 | **CC-BY 4.0** (Uppsala) | `https://ucdp.uu.se/downloads/ged/ged251-csv.zip` (29 MB) → `GEDEvent_v25_1.csv` | `data/empirical/ucdp_ged_events.npy` (structured) |
| 7 | Natural Earth 10 m rivers | **Public domain** | `https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip` (2 MB) | `data/empirical/ne_10m_rivers_lake_centerlines.geojson` |

All seven were verified end-to-end against their canonical archives; the
"receipts" (PDF citations, S3 bucket listings, Dataverse JSON API
responses) are summarised in [`data_attributions.md`](data_attributions.md)
and (for HadCRUT5 specifically) in the test
[`test_macro.py::test_hadcrut5_consistency_with_gistemp`](../test_macro.py)
which already passes live (HadCRUT5 last-3-yr mean baseline-corrected to
+1.21 °C vs. MacroState 1.30 °C; agreement 0.09 °C, well under the
0.20 °C tolerance).

Two new optional extras were added to keep dependencies opt-in:

- **`[data]`** = `rasterio>=1.3`, `pyproj>=3.6`, `pyshp>=2.3` —
  needed for the CHELSA / SoilGrids / HYDE / ETOPO / NE-rivers downloaders.
- **`[torch]`** = `torch>=2.2` — needed only for the PyTorch JEPA backend.

HadCRUT5 and UCDP-GED ingestion need only `numpy + requests + stdlib`.

### Sobol sensitivity analysis (2026-06-22)

[`scripts/sensitivity.py`](../scripts/sensitivity.py) runs a global
variance-based Sobol decomposition of the macro layer's 2100 BAU outputs
against eight key parameters with literature-grounded priors. At N=64
(1 152 BAU evaluations, ~5 min on commodity CPU), real findings (see
[`validation.md` §6](validation.md)):

- `CLIMATE_FEEDBACK` (λ) explains **79 % of 2100 ΔT variance** (ST = 0.79).
- `NATURAL_ABSORPTION_RATE` explains **60 % of 2100 CO₂ variance**;
  `BASE_EMISSION_RATE` adds 17 %.
- `CLIMATE_FEEDBACK` dominates 2100 population variance too (ST = 0.67)
  via the DICE-damage channel.
- `OCEAN_HEAT_CAPACITY` and `ABSORPTION_TEMP_SENSITIVITY` are essentially
  inactive at the 2100 horizon — candidates for "fix at central value" in
  any future Bayesian-calibration sweep.

This was prompted by, and forms the highest-ROI single response to, an
external review proposing SALib among other suggestions (most of which were
already implemented in the repo — see *Reviewer critique* below).

---

## Decisions we made — and decisions we explicitly DECLINED

These exist primarily as "do not re-litigate" entries; the rationale is
preserved so future sessions don't re-open settled questions.

| Decision | Outcome | Why |
|---|---|---|
| PyTorch JEPA backend default settings | **Reproduce NumPy backend exactly** (paper-aligned options gated as opt-in toggles) | A "backend" must give the *same model*. Paper-faithful values land via `sigreg_mode="epps_pulley"` / `lambda_reg=0.1` / etc. |
| Torch as a hard dependency | **No — keep as `[torch]` extra** | User confirmed: heavy/optional deps stay opt-in. `pip install -r requirements.txt` must remain lightweight. |
| Seshat as a bulk-import data source | **No — adopt the *schema* (complexity characteristics), not the data dump** | Your world is emergent on real geography with a fictional accelerated history; can't validate trajectory-by-trajectory against real polities. Statistical regularities (Turchin et al. 2018 PC1) can be validated, the bulk databank cannot. |
| Bayesian SBI on the current macro | **Postpone to Phase 2 — data inputs first** | SBI on partially-synthetic substrate would put rigour on the wrong target. CHELSA / SoilGrids / HYDE first; SBI on real-input model after. |
| Berkeley Earth temperature | **Drop** | CC-BY-NC — the NC clause propagates to the combined AGPL+data distribution. GISTEMP + HadCRUT5 cover the same scientific role. |
| HydroSHEDS as a Tier-1 recommendation | **Drop, with apology — was wrong in earlier recommendation** | Custom WWF licence (Appendix A): "in no event shall Licensee license or distribute the Licensed Materials as a stand-alone product"; EULA + audit + recordkeeping requirements. Incompatible with an AGPL public repo. |
| MERIT Hydro as the HydroSHEDS alternative | **Also drop** | Dual-licence CC-BY-NC 4.0 OR ODbL 1.0 *plus* Google-form registration. Neither branch is clean for our situation. |
| High-res rivers replacement | **Natural Earth 10 m** | Same publisher as the shipped 110 m; public domain; clean drop-in (113× more features than the shipped 110 m vector). |
| Computable General Equilibrium (CGE) economics | **Skip** | Requires national-accounts calibration data; appropriate for policy IAMs, not an agent simulator with hundreds of agents on real geography. |
| Leontief input-output for sectoral resource conservation | **Genuinely valid future addition** (medium priority for v0.5+) | Real omission flagged by reviewer; current macro has aggregated resource stocks but no sectoral I/O closure. |
| Formal expected-utility conflict initiation | **Genuinely valid future addition** (lower priority) | Current `geopolitics.py` logistic is structural, not decision-theoretic. Bueno-de-Mesquita-style EU model would close that gap. |
| The "scientifically unrigorous" framing of the external review | **Largely uninformed about the actual code** | Asserted random-noise climate, dice-roll conflict, buzzword JEPA — all wrong (real 14-state ODE, IFs gravity/logistic, gradient-checked paper-faithful JEPA). Three of the bullets (SALib, Leontief I/O, expected utility) are real signal; the rest is generic noise. SALib has been implemented; the other two are queued for v0.5+. |

---

## Verified facts (don't re-verify in future sessions)

### Dataset licences — checked against official archives 2026-05-25 / 2026-06-21

- **CHELSA V2.1** is **CC0 1.0** (verified on EnviDat record).
- **HYDE 3.2.1** is **CC0 1.0** (verified on DANS data-station).
- **SoilGrids 2.0** is **CC-BY 4.0** (verified on ISRIC docs).
- **HadCRUT5** is **Open Government Licence v3.0** (Met Office Hadley Centre).
- **ETOPO 2022** is **US Government public domain** (User Guide §4 verbatim:
  *"freely available to use for all private, academic, or commercial purposes"*).
- **UCDP-GED** datasets are **CC-BY 4.0** (verbatim on downloads page:
  *"All datasets are free of charge and licensed under CC BY 4.0"*).
- **Natural Earth** (all resolutions) is **public domain**.
- **ERA5 (Copernicus)** is the Copernicus "Licence to Use Copernicus
  Products" v1.2 — free, commercial OK, derivatives OK, attribution
  required; AGPL-compatible. Requires CDS-API registration to *download*.
- **Berkeley Earth** is **CC-BY-NC 4.0** — *not* CC-BY 4.0 as I originally
  claimed. Avoid for AGPL projects.
- **HydroSHEDS** uses a **custom WWF licence** (see HydroSHEDS Technical
  Documentation v1.4 Appendix A). Not AGPL-compatible.
- **MERIT Hydro** is **dual-licensed CC-BY-NC 4.0 OR ODbL 1.0** with Google-
  form registration. Neither branch is clean for an AGPL repo.

### LeWorldModel (Maes et al. 2026, arXiv:2603.19312) — fetched from arXiv HTML

- ~15 M params, single-GPU training in hours.
- Encoder: ViT-tiny (~5 M), patch 14, 12 layers, 3 heads, dim 192.
- Predictor: Transformer, 6 layers, 16 heads (~10 M), 10 % dropout, **AdaLN
  per layer with zero-init** (DiT trick), autoregressive over history N.
- SIGReg: Epps–Pulley univariate stat + Cramér–Wold theorem; **M=1024**
  random unit-norm projections; **λ=0.1**; trapezoid integration, T nodes
  in [0.2, 4]; weighting w(t)=e^(−t²/2λ²).
- Loss: L = L_pred + λ·SIGReg(Z).
- **No stop-gradient, no EMA, no target encoder — fully end-to-end.** This
  is the paper's signature contribution.
- Optimiser / base LR / batch size / weight decay: **not stated** in
  abstract or rendered HTML (Appendix D not surfaced).
- *Input modality is pixel sequences — patchify / ViT / autoregressive
  frames only make sense for images. Our 40-dim vector observation does
  not need the ViT; the MLP adaptation is the correct engineering call.*

### File specifications already extracted

- CHELSA V2.1 file naming: `CHELSA_<short_name>_<MM>_<period>_V.2.1.tif`;
  monthly bioclimatology variables include `tas`, `tasmin`, `tasmax`,
  `pr`, plus derived `bio1`…`bio19` etc.
- SoilGrids WebDAV layout: `https://files.isric.org/soilgrids/latest/data/<prop>/<prop>_<depth>_<stat>.vrt`
  with depths `0-5cm`, `5-15cm`, `15-30cm`, `30-60cm`, `60-100cm`,
  `100-200cm` and stats `mean`, `Q0.05`, `Q0.5`, `Q0.95`, `uncertainty`.
- HYDE Dataverse fileIds (verified): `5490328` baseline (5.3 GB), `5490329`
  anthromes, `5490327` supplementary, `5396388` readme, `5398615`
  easy-migration.
- HadCRUT5 CSV columns: `Time, Anomaly (deg C), Ensemble standard
  deviation (1 sigma), Coverage uncertainty (1 sigma), Total uncertainty
  (1 sigma)` (1850–2026, annual).
- UCDP-GED inner CSV: `GEDEvent_v25_1.csv`; relevant columns for us are
  `year, latitude, longitude, best, conflict_new_id`.

---

## What's still to do (priority order)

### High-priority for v0.4

1. **Wire `data/empirical/*` into `world.py` runtime.** All seven empirical
   inputs are scaffolded but the simulator still reads the synthesised
   rasters from `generate_earth_data.py`. The wiring should:
   - Add a `world.py` option (or scenario flag) to prefer `data/empirical/*`
     when it exists, with graceful fallback to the synthesised raster.
   - Document the choice in `validation.md` and add a small switch test
     verifying both paths run.
2. **Demographic validation track using HYDE 3.2.1.** Mirror the IPCC AR6
   pattern: a `test_macro_population_vs_hyde` that compares simulator
   emergent population for a few past century windows to HYDE's
   reconstruction with an honest tolerance.
3. **Conflict validation track using UCDP-GED.** A spatial summary —
   conflict-event density by 5°×5° cell over the simulator's analogous
   period — and a check that the model's emergent conflict spatial
   pattern correlates positively with the empirical density.

### Medium-priority for v0.5+

4. **Leontief input-output for sectoral resource conservation.** Small I/O
   matrix `(food / energy / materials → goods / services)` with technical
   coefficients calibrated from World Bank or FAO. Closes the resource-
   accounting loop in the macro layer.
5. **Bayesian SBI on the macro** (the Phase-2 we discussed). With CHELSA /
   SoilGrids / HYDE / HadCRUT5 / ETOPO in place, SBI no longer points at
   synthetic substrate. Use `sbi` library (PyTorch-native, composes with
   the JEPA backend). Start with macro-only SNPE-C / NPE; add JEPA encoder
   as a learned embedding net once that proves itself.
6. **Formal expected-utility conflict initiation.** Replace the structural
   logistic in `geopolitics.py` with a Bueno-de-Mesquita-style EU model.

### Low-priority / nice-to-have

7. **Larger Sobol N (N=256 or 512)** to tighten the second decimal of each
   index; current N=64 has CI half-width ~0.05 for the dominant terms.
8. **Sobol with ECS-constrained joint prior** on `F_2x` × `λ` (currently
   they're treated as independent; in practice they are jointly
   constrained by ECS observations).
9. **GLOFAS / GRDC runoff data** if river-flow realism becomes important
   (Copernicus licence, AGPL-compatible).
10. **Maddison Project historical GDP** for a long-period macroeconomic
    validation track. (Licence: verify before committing.)
11. **Spatial paleo climate** (TraCE-21k or PMIP) to ground the 70 ka
    simulation timescale beyond scalar EPICA forcing.

### Explicitly out of scope

- CGE economics (handled above).
- D-PLACE cultural data (CC-BY-NC-SA — same licence trap as Berkeley
  Earth and HydroSHEDS).
- Hansen Global Forest Change (CC-BY-NC).
- The bulk Seshat databank as a data source (handled above).

---

## Memory entries already saved (in Claude's persistent memory)

Already persisted across conversations so I can recall the cross-cutting
facts without re-fetching:

- [`reference_leworldmodel_paper.md`](../../../.claude/projects/-home-parallels-Documents-Python-claude-code-world-genesis/memory/reference_leworldmodel_paper.md)
  — verified LeWorldModel hyperparameters.
- [`project_jepa_paper_deviations.md`](../../../.claude/projects/-home-parallels-Documents-Python-claude-code-world-genesis/memory/project_jepa_paper_deviations.md)
  — how `world_model.py` adapts the paper (MLP not ViT, etc.).
- [`feedback_optional_deps.md`](../../../.claude/projects/-home-parallels-Documents-Python-claude-code-world-genesis/memory/feedback_optional_deps.md)
  — keep torch / rasterio / SALib as opt-in extras, not core deps.

(Paths from this file are illustrative — memory lives outside the repo.)

---

## Code artefacts shipped or scaffolded across these sessions

| File | Purpose |
|---|---|
| `world_model_torch.py` | PyTorch JEPA backend (paper-faithful + opt-in paper toggles) |
| `test_world_model.py` | JEPA learning behaviour tests (5) |
| `test_world_model_gradcheck.py` | Finite-diff gradient checks (5, <1e-8 rel error) |
| `test_world_model_torch.py` | Torch parity + weight cross-check tests |
| `generate_empirical_inputs.py` | CHELSA, SoilGrids, HYDE, HadCRUT5, ETOPO, UCDP-GED, NE rivers downloaders |
| `test_empirical_inputs.py` | ETOPO / UCDP-GED / NE rivers structural validation (skip-clean) |
| `test_sensitivity.py` | SALib pipeline smoke + parameter-override propagation |
| `scripts/sensitivity.py` | Sobol N=64 over 8 macro parameters, 3 outputs |
| `scripts/results/sensitivity_N64.json` | Real Sobol indices + 95 % CIs |
| `.github/workflows/test.yml` | CI on Python 3.11/3.12/3.13 |
| `.gitignore` | Standard + project-specific (logs/, etc.) |
| `docs/data_attributions.md` | Extended with all empirical sources + verified licences |
| `docs/validation.md` | New §6 Sobol decomposition; PyTorch claim modernised |
| `docs/dev_log.md` | This file |
