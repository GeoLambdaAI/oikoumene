# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — empirical ingestion wired (v0.4, first increment)

- **`empirical_ingest.py`** — runtime-safe (pure numpy + stdlib, no GDAL)
  ingestion layer that turns the downloaded empirical datasets into simulator
  inputs at init, kept separate from the rasterio/GDAL downloaders so the
  runtime never depends on them.
- **UCDP-GED conflict anchoring.** The `present_day` scenario now seeds its
  initial active conflicts from real Uppsala Conflict Data Program georeferenced
  events (`derive_conflicts_from_ucdp`): recent events are grouped by
  `conflict_id` into zones with real centroid, fatality-log-scaled intensity,
  and event-spread radius; the 15 most lethal are seeded — replacing the 10
  hand-authored entries in `present_day_conflicts.json`. Correctly surfaces
  Ukraine, Tigray, Gaza, Sudan, the Sahel, etc. Degrades gracefully to the
  shipped JSON when the (gitignored, optional) dataset is absent.
- **HadCRUT5 temperature anchoring.** The `present_day` scenario now initializes
  its starting temperature from the latest observed HadCRUT5 global-mean anomaly
  (`observed_temperature_anomaly`), **converting from HadCRUT5's 1961–1990
  baseline to the 1850–1900 pre-industrial baseline the macro model uses** — the
  offset is derived from the data's own 1850–1900 window (not hardcoded) and
  reproduces IPCC AR6 (converted 2011–2020 mean = 1.11 °C vs AR6's ~1.09 °C).
  The full observed record is exposed via `observed_temperature_series` for
  validating the macro trajectory. Graceful fallback to the JSON value when the
  dataset is absent; derived macro fields are recomputed so the anchored state
  is consistent from tick 0.
- `test_empirical_ingest.py` — UCDP aggregation/ranking/schema/`top_k`/fallback,
  HadCRUT5 baseline conversion + fallback, and present-day integration tests.
- **Sobol sensitivity pipeline verified end-to-end** (`scripts/sensitivity.py`,
  SALib). Confirmed physically sensible variance decomposition of the BAU-2100
  outputs: `T_2100` dominated by the climate-feedback (ECS) parameter (ST≈0.79),
  `CO2_2100` by the natural sink fraction + baseline emissions, `Pop_2100` by
  climate feedback (via damage) then population growth. Documented in
  `docs/validation.md` §6.
- Remaining v0.4 empirical wiring (ETOPO elevation, CHELSA climate, SoilGrids
  fertility, HYDE population) is deferred to a v0.5 "high-fidelity Earth"
  milestone — those need rasterio/GDAL and large downloads (HYDE alone ~5.3 GB);
  the downloaders exist but the rasters are not yet ingested.

### Added — empirical input scaffolding (for v0.4)

- **`generate_empirical_inputs.py`** — downloads and resamples three
  authoritative open datasets to the simulator's 0.5° grid:
  - **CHELSA V2.1** climatology (CC0 1.0) — verified at
    https://doi.org/10.16904/envidat.228; streamed from
    `https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/climatologies/`
    via GDAL `/vsicurl/` so the network footprint stays small.
  - **SoilGrids 2.0** soil properties (CC-BY 4.0) — verified at
    `https://files.isric.org/soilgrids/latest/data/`; root-zone
    soc + nitrogen + clay combined into a composite fertility index.
  - **HYDE 3.2.1** historical land-use & population (CC0 1.0) —
    verified at https://doi.org/10.17026/DANS-25G-GEZ3 via the standard
    Dataverse access API (`baseline.zip` fileId 5490328, ~5.3 GB,
    cached on disk).
  - **HadCRUT.5.1.0.0** Met Office + CRU global mean annual temperature
    anomaly 1850-present (Open Government Licence v3.0) — verified at
    https://www.metoffice.gov.uk/hadobs/hadcrut5/, downloads the
    `component_series.global.annual.csv` (~10 KB) and saves as a structured
    NumPy array `data/empirical/hadcrut5_global_annual.npy` with named
    fields `(year, anom, sigma_ens, sigma_cov, sigma_total)`.
  - **ETOPO 2022** NOAA NCEI 60-arc-sec global relief
    (**US Government public domain**; *"freely available to use for all
    private, academic, or commercial purposes"* per User Guide §4) — single
    NetCDF on the NOAA THREDDS server (~395 MB, one-time download),
    resampled to the simulator's 0.5° grid and saved as
    `data/empirical/etopo_elevation.npy` (float32 metres, EGM2008-referenced;
    ocean values negative). Replaces the synthetic `earth_elevation.npy`.
  - **UCDP-GED v25.1** Uppsala Conflict Data Program Georeferenced Event
    Dataset 1989-present (**CC-BY 4.0**) — verified at
    https://ucdp.uu.se/downloads/, downloads `ged251-csv.zip` (~29 MB)
    and saves a structured NumPy array
    `data/empirical/ucdp_ged_events.npy` with named fields
    `(year, lat, lng, best, conflict_id)`. First empirical anchor for the
    emergent geopolitics layer (replaces the role of the illustrative
    `present_day_conflicts.json`).
  - **Natural Earth 10 m rivers** (public domain) — drop-in resolution
    upgrade over the shipped 110 m vector. Fetched from the Natural Earth
    CDN, shapefile converted to GeoJSON via `pyshp`, saved as
    `data/empirical/ne_10m_rivers_lake_centerlines.geojson`.
  - Outputs go under `data/empirical/` so they coexist with the synthesized
    `data/earth_*.npy` rasters; `world.py` will choose which to load when
    the actual ingestion is wired (separate change for v0.4).
- New optional dependency extra **`[sensitivity]`** (`SALib>=1.5`) in
  `pyproject.toml`; install with `pip install -e ".[sensitivity]"`.
  Runtime simulator does not depend on it.
- New optional dependency extra **`[data]`** (`rasterio>=1.3`,
  `pyproj>=3.6`, `pyshp>=2.3`) in `pyproject.toml`; install with
  `pip install -e ".[data]"`. The simulator runtime does not depend on it.
  (HadCRUT5 and UCDP-GED ingestion only need numpy + requests + stdlib, so
  they do not require the `[data]` extra. CHELSA / SoilGrids / HYDE / ETOPO
  need rasterio + GDAL; the Natural Earth 10 m rivers downloader needs
  pyshp for shapefile → GeoJSON conversion.)

### Added — validation

- **`scripts/sensitivity.py`** — Sobol global sensitivity analysis on the
  macro layer's 2100 BAU outputs (`T_2100`, `CO2_2100`, `Pop_2100`).
  Decomposes variance into the contributions of eight key parameters
  (forcing, climate feedback, ocean coupling, sink fraction + sink
  weakening, baseline emissions, population growth) with priors drawn
  from IPCC AR6 / Friedlingstein 2024 / UN WPP literature ranges. Uses
  SALib's Saltelli sampler; default N=64 takes ~5 min on commodity CPU.
  Results land in `scripts/results/sensitivity_N<N>.json` and are
  summarised in [`docs/validation.md`](docs/validation.md).
- **`test_sensitivity.py`** — three smoke tests for the SALib pipeline:
  problem spec matches the documented 8-parameter set, instance-level
  parameter overrides actually propagate into the integrated trajectory
  (lambda=0.8 vs 1.6 must yield T_2100 spread > 0.5 deg C), and a tiny
  N=2 end-to-end run produces a well-shaped JSON artefact. Skips when
  SALib is not installed.
- **`test_macro.py::test_hadcrut5_consistency_with_gistemp`** —
  new validation test that cross-checks the macro layer's NASA GISTEMP-
  derived present-day temperature anchor against an *independent* second
  observational reconstruction (HadCRUT5). Skips automatically if the user
  has not run `--hadcrut5` (no forced network download in CI).
- **`test_empirical_inputs.py`** — new test module dedicated to the
  empirical-input loaders. Each test skips cleanly when its dataset has
  not been fetched and asserts strong realistic properties when present:
  - `test_etopo_elevation_loads_and_is_realistic`: shape, ocean-dominated
    mean, Himalayan-scale max (>5 km), trench-scale min (<-5 km), land
    fraction in [0.20, 0.45].
  - `test_ucdp_ged_events_load_and_have_realistic_spread`: ≥100k events,
    1989-start year range, globally-spread lat/lng, plausible total
    best-estimate deaths.
  - `test_ne_10m_rivers_richer_than_shipped_110m`: strictly more features
    than the shipped 110m vector; only LineString / MultiLineString
    geometries.
- **`docs/data_attributions.md`** — new sections for CHELSA, SoilGrids,
  HYDE with verified licences, archives and citations; updated licence-
  compatibility footer.

### Added — dynamic anthropogenic climate

- **Industrialization-coupled CO₂ emissions.** Anthropogenic CO₂ now emerges
  from the civilization's development instead of a fixed present-day rate: the
  macro layer's fossil emissions are gated by an `industrialization` factor in
  `[0,1]` (derived from the civ's discovered industrial technologies via
  `HistoricalSimulation.industrialization_level()`), plus a small pre-industrial
  land-use term (Ruddiman 2003 "early Anthropocene"). A low/pre-industrial
  civilization now produces little CO₂; a fully industrial one reproduces the
  calibrated 2025 rate. `industrialization` defaults to `1.0` with no agent
  feedback, so the present-day scenario and the IPCC BAU validation are preserved
  exactly.
- **Continuous paleo → Industrial climate handoff.** When a historical run first
  enters the Industrial era, the macro state is seeded from the paleoclimate
  trajectory (temperature, CO₂, sea level, year, pre-industrial socioeconomics)
  instead of snapping to the present-day `MacroState` defaults — the climate now
  carries the lower paleo values forward and rises as the civ industrialises.
- **`test_climate_continuity.py`**, **`test_god_mode.py`**,
  **`test_reproducibility.py`** — new regression suites; the BAU IPCC envelope is
  now a collected `pytest` test (`test_bau_scenario_ipcc_envelope`), so the
  headline 9/9 validation runs in CI, not only via `python test_macro.py`.

### Changed

- **Reproducibility.** Agent-level randomness (traits, mutation, movement,
  partner selection, reproduction, research, names) now draws from the world's
  seeded `RandomState` instead of the unseeded global `np.random`; per-world
  entity ID counters reset on construction. A fixed `World(seed=…)` is now
  bit-reproducible across runs.
- **Earth-system grids regenerated.** The latitude–temperature law was recalibrated
  to its documented ERA5 zonal-mean anchors (it was ~7 °C too warm in the polar
  tail), so Antarctica now classifies as ice sheet; continentality is cos(lat)-
  corrected. The committed `data/earth_*.npy` were regenerated accordingly.
- **HadCRUT5 series** drops trailing provisional (in-progress) calendar years via
  their inflated coverage uncertainty, so the saved record ends at the last
  complete annual mean.

### Fixed

- **Engine: unbounded growth.** Dead agents and dead settlements (population 0)
  are now pruned each tick — previously both lists grew without bound over a long
  historical run, leaking memory and making per-tick loops O(total-ever-created).
- **Engine: economy.** Trade is now conservative (a transfer, not wealth created
  from nothing); wealth/energy/happiness/health are clamped to their declared
  ranges; reproduction debits both parents symmetrically; geopolitics aggregates
  are hardened against negative/NaN propagation.
- **Diamond geographic determinism** now diffuses agriculture/technology faster
  east–west than north–south (anisotropic, per Diamond 1997 Ch. 10) instead of an
  isotropic circular radius.
- **Paleoclimate anchor labels** corrected to the 1950-BP convention (industrial
  CO₂ ramp 290→310→420 ppm); `god_mode` drought now applies once and restores the
  exact baseline (was compounding per tick); plus dateline-wrap, geopolitics
  parity/relation-edge, and a dead scenario placeholder loop.

### Performance

- **Vectorised the two hot per-cell loops** that ran every macro/ice tick over the
  ~12k-cell grid: `bridge.apply_macro_to_world` (8.5 ms → 0.15 ms) and
  `world._apply_ice_age_effects` (289 ms → 0.10 ms) — both verified numerically
  identical to the original loops. These removed periodic multi-100 ms stalls.
- **Geopolitics** no longer stalls abruptly at the Industrial-era transition:
  settlement pruning bounds the count, and nation-stat aggregation is now
  O(agents + memberships) instead of O(settlements × agents).

## [0.3.1] - 2026-07-06

Security & correctness patch. No breaking API changes; shipped ahead of the v0.4
empirical-data work (which remains in `[Unreleased]`).

### Security — web/SocketIO surface hardening (`app.py`, `llm_module.py`, `templates/index.html`)

- **API-key exfiltration closed.** `LLMModule.update_config` now binds the API
  key to `base_url`: changing `base_url` to a new host without re-supplying the
  key clears it, so an attacker can no longer redirect the endpoint and have the
  operator's Bearer token forwarded to their server.
- **SSRF guard.** `base_url` is validated (http/https only; cloud-metadata and
  link-local ranges blocked; optional `OIKOUMENE_LLM_ALLOWED_HOSTS`
  allowlist) at config time and again before every outbound LLM request.
- **Stored/DOM XSS fixed.** Added an `escapeHtml()` helper and applied it to
  every dynamic value interpolated into `innerHTML` (dialogue feed, chat
  bubbles, god/LLM responses, nation/settlement names, LLM model/error strings).
- **Input validation** on unauthenticated SocketIO events: `reset` clamps agent
  count/seed/scenario, `set_speed` rejects NaN/inf, `set_llm_config` type-checks
  and clamps every field.

### Fixed — simulation-core correctness

- **Concurrency (`app.py`).** The background tick loop now uses a generation
  token plus a world-mutation lock, so a reset/JEPA-swap/restart can no longer
  leave a stale loop double-stepping the world or interleaving mid-tick.
- **present_day nations no longer self-delete.** Scenario-seeded real countries
  are flagged `seeded` and are exempt from the empty-nation prune and stat
  zeroing (previously all ~140 nations plus the NATO/EU/BRICS graph were wiped
  ~10 ticks in).
- **Reproducibility.** Tech-discovery and god-mode compliance draws now use the
  world's seeded RNG (not the global `np.random`); `Business`/nation id counters
  reset per `World`; `get_spawn_locations` honours the world seed.
- **Scientific logger** starts in the `present_day` scenario (was skipped by an
  early return) and `end_run()` is idempotent and called on reset (was never
  called — leaked the CSV handle and truncated runs).
- **Climate handoff continuity.** The macro `_vector_to_state` clamps were
  widened so the paleo→macro handoff's negative Little-Ice-Age anomaly (−0.3 °C,
  283 ppm) is no longer snapped to 0 on the first ODE step; the handoff now also
  recomputes derived fields immediately.
- **Resource-regen ownership.** Paleo ice-age effects are gated to the paleo era
  and god-mode drought is modelled as a persistent multiplicative factor that
  the macro bridge and ice-age both compose with — a drought is no longer erased
  at the next macro/ice tick, and per-cell baselines are captured pristine at
  init rather than ~50 ticks late.
- **Bounded growth.** Settlement membership sets are pruned to living members,
  `negotiation_history` is capped, and a removed nation's id is purged from every
  other nation's alliance/rival sets.
- **Agent fixes.** Velocity observations are normalised by the era-scaled max
  speed (were reaching ±20 and dominating the JEPA encoder); death cause
  distinguishes starvation / old_age / illness; `_action_migrate` clamps latitude
  and wraps longitude; historical spawns top up to the requested count instead of
  silently under-populating.

## [0.3.0] - 2026-05-25

Adds the PyTorch JEPA backend that earlier releases listed as a v0.3
candidate. The backend is opt-in; the default NumPy path and all prior
behaviour are unchanged. This release also closes documentation/test gaps
found in an upload-readiness audit.

### Added — PyTorch JEPA backend

- **`world_model_torch.py`** — a PyTorch re-implementation of the JEPA
  world model (encoder, AdaLN predictor, SIGReg, CEM planner) using autograd
  instead of the hand-written NumPy backprop. At its default settings it
  reproduces `world_model.py` exactly: weights copied across backends via
  `load_numpy_params` / `export_numpy_params` produce encode/predict outputs
  matching to < 1e-4 (verified in `test_world_model_torch.py`). CPU by
  default with optional CUDA (`device="auto"`); `torch.set_num_threads(1)`
  by default to stay friendly to the eventlet server loop.
- **Paper-aligned toggles (opt-in, off by default):** `sigreg_mode=
  "epps_pulley"` implements the LeWorldModel (Maes et al. 2026,
  arXiv:2603.19312) characteristic-function SIGReg with the paper's
  λ = 0.1 and a large projection count, plus `predictor_dropout`. The
  default `moments` SIGReg (M = 15, λ = 0.01) is unchanged.
- **`SharedWorldModel(backend="numpy"|"torch", ...)`** dispatch; the torch
  import is lazy so PyTorch remains an optional dependency
  (`pip install -e ".[torch]"`).
- **Dashboard JEPA tab** — select backend (NumPy / PyTorch), settings preset
  (Repo default / Paper), and device at runtime via `set_jepa_backend`
  (`app.py`) / `World.set_jepa_backend` (`world.py`). The swap preserves the
  experience buffer and repoints all agents; the sim loop is paused during
  the swap and resumed. Falls back gracefully with a clear message when
  PyTorch is not installed.

### Added — tests

- **`test_world_model_gradcheck.py`** — central finite-difference gradient
  checks for every analytic backward pass (linear, GELU, RMSNorm, AdaLN,
  SIGReg). Measured relative error < 1e-8. This is the test prior READMEs
  referred to but did not ship.
- **`test_world_model.py`** — JEPA learning/inference behaviour: prediction
  loss reduction, learned action conditioning (zero-init AdaLN identity →
  action-sensitive after training), anti-collapse, linear probe R², CEM
  planner output validity.
- **`test_world_model_torch.py`** — torch backend parity, weight cross-check,
  Epps–Pulley toggle, and device handling (skips when torch is absent).

### Fixed — documentation & packaging

- README test table referenced two test files that did not exist
  (`test_world_model.py`, `test_world_model_gradcheck.py`); both now ship.
- `world_model.py` docstring pointed at a non-existent `test_layers_gradcheck.py`.
- Project version was out of sync (`pyproject.toml` 0.1.0 vs CITATION/CHANGELOG
  0.2.1); all now read 0.3.0.
- `docs/validation.md` described the PyTorch port as future work; it is now
  documented as implemented.
- Added `.gitignore` and a GitHub Actions test workflow
  (`.github/workflows/test.yml`, Python 3.11/3.12/3.13).
- `test_macro.py` unit tests now `assert` instead of `return`-ing a bool
  (removes pytest `PytestReturnNotNoneWarning`).

## [0.2.1] - 2026-05-06

A same-day follow-up review pass on v0.2.0 surfaced five additional bugs
in the integration glue between the (now correctly calibrated) scientific
modules and the simulation loop, plus several UI-payload issues that
caused the right-sidebar dashboard to misrepresent paleo-era state. The
eight scientific modules' equations and constants are unchanged; v0.2.1
is purely a coupling, timing, and presentation correctness release. All
v0.2.0 calibration tests pass unchanged.

### Fixed — Macro/agent coupling (`bridge.py`)

- **Water and minerals regen ratchet.** `apply_macro_to_world` was
  multiplying `water_regen` and `minerals_regen` cell-wise by macro
  factors on every call, with no baseline reset. Over hundreds of macro
  ticks at typical factor < 1 the regen rates underflowed to zero
  independently of the actual macro state. v0.2.1 snapshots the
  resource-map's per-cell baselines on first call and rebuilds
  `regen[r,c] = base[r,c] × factor` each tick (idempotent) — the same
  pattern `food_regen` already followed.
- **`food_regen` terrain-factor inflation.** The v0.2.0 formula
  `(2.0 if plains else 1.0) × fertility × combined` collapsed five
  terrain-specific regen factors from `ResourceMap.initialize_from_terrain`
  (plains 2.0, forest 1.0, mountain 0.2, desert 0.1, tundra 0.3) into a
  two-factor approximation, silently inflating mountain food regen by
  5×, desert by 10×, and tundra by 3.3×. v0.2.1 derives the per-cell
  baseline from `terrain × fertility` via a constants table
  (`_FOOD_REGEN_TERRAIN_FACTOR`) that mirrors `world.py:78-113` exactly.
  The baseline is *derived* from the terrain map rather than snapshotted
  from the live `food_regen` array, because `world._apply_ice_age_effects`
  mutates `food_regen` multiplicatively in paleo era and a live snapshot
  would be contaminated on paleo→modern transitions.

### Fixed — Geopolitics (`geopolitics.py`)

- **Tech diffusion double-credited.** `_resolve_trade` writes both
  `(a, b)` and `(b, a)` edges with the same weight, and
  `_diffuse_technology` iterated `trade_graph.edges()` which yields both
  directions; on each visit the lower-tech nation was credited.
  Diffusion therefore ran at twice the calibrated `0.001 × volume × |Δtech|`
  rate. Fixed by `if na_id >= nb_id: continue` inside the loop so each
  unordered dyad is processed exactly once.
- **Stale phantom trade edges.** `_resolve_trade` only called `add_edge`
  when `volume > 0.01` and never `remove_edge`. When a dyad fell below
  threshold (e.g. autarky+rivalry), the previous tick's edge persisted
  with its old weight, feeding phantom values into
  `conflict_probability`'s `trade_interdep` term (Russett-Oneal liberal
  peace), `_update_relations`'s `trade_bonus`, and `_diffuse_technology`.
  Fixed by explicit edge removal in the `else` branch.

### Fixed — World engine (`world.py`)

- **Paleo `_apply_ice_age_effects` ratchet + ice-retreat recovery.**
  `food_regen *= cold_factor` compounded across thousands of paleo ticks
  (`tick % 20 == 0 and year_bp > 5000`). With cold_factor sustained at
  ~0.7 across ~40 000 paleo applications, `food_regen` underflowed to
  numerical zero long before LGM. Cells that became ice-covered at any
  point during the run had `food`, `food_regen`, `wood`, `wood_regen`,
  and `water` zeroed and never recovered, even after ice retreated and
  the cell became habitable — inconsistent with the post-LGM
  recolonisation record. v0.2.1 lazy-snapshots per-cell baselines on
  the first paleo call and uses set-from-baseline semantics; a per-cell
  `_was_iced` boolean flag triggers post-glacial recovery seeding on
  the iced→non-iced transition. The `cold_factor` formula is
  unchanged — at LGM (-8 °C anomaly per EPICA/Vostok ice cores) it
  yields ~36 % productivity, within the paleo-NPP envelope of Adams
  & Faure (1998) and Crowley & Baum (1997).
- **Macro `dt_years` 10× rate fix.** `MacroModel` was instantiated with
  `dt_years = 1/12` (one month per call) but `macro.step()` is invoked
  once per `macro_update_interval = 10` world ticks, and Modern era
  advances time at `era.time_scale = 1/12 yr/tick`. The macro ODE
  therefore integrated only one month per ten sim months, running at
  one-tenth of the calibrated rate; the macro clock fell ~10× behind
  the historical clock and CO₂/temperature/sea-level evolved
  proportionally slowly. v0.2.1 sets
  `dt_years = macro_update_interval / 12 = 10/12`, keeping both clocks
  synchronous. The IPCC AR6 SSP envelope outputs at 2100 are preserved
  (681 vs 679 ppm CO₂, 0.3 % drift; same temperature and sea level)
  because `solve_ivp(method='RK23')` adapts internally; the standalone
  `test_macro.py` calibration path is unaffected (it instantiates
  `MacroModel` with its own `dt_years = 1/12` and calls `step()` per
  iteration, the originally calibrated path).
- **Era-aware UI payload (`_build_era_aware_summaries`).** In paleo era
  the `MacroModel` ODE does not step (industrial economy emerges only
  post-1750), so `macro.get_summary()` returned `MacroState` defaults
  frozen at year 2025; the right-sidebar Global State panel never
  reflected the paleoclimate trajectory across the 70 000-yr history
  view. v0.2.1 introduces a single helper called from both
  `World.step()` (for the websocket "tick" emit) and
  `World.get_full_state()` (for "full_state"), so both paths deliver
  identical payloads. In paleo era it populates climate fields from
  `PaleoclimateModel` (calibrated to EPICA Dome C, Lüthi et al. 2008,
  and Vostok, Petit et al. 1999) and population from the new
  paleodemographic helper `_paleo_population_billions` (sourced from
  McEvedy & Jones 1978; Biraben 2003; HYDE 3.1, Klein Goldewijk et al.
  2010). Industrial-era fields carry their pre-industrial physical
  values: `fossil_fuels = 1.0`, `renewable_frac = 0.0`,
  `pollution = 0.0`. Technology is normalised by tech-tree size so it
  rises smoothly toward `MacroState.technology_level = 1.0` at the
  Industrial-era handoff. In modern era the helper continues to
  override `history.{co2_ppm, temperature_anomaly, sea_level_m,
  year_*}` with `MacroModel.state` values so the top-header and sidebar
  surfaces stay coherent (a fix originally introduced in v0.2.0 for the
  `step()` path; v0.2.1 extends it to the `get_full_state()` path that
  the websocket `'full_state'` event actually consumes).
- **Settlement count in geopolitics summary.** Nations form only when
  settlements grow ≥ `NATION_FORMATION_POP` ([geopolitics.py:125](geopolitics.py#L125)),
  so in paleo era `nations` / `active_conflicts` / `trade_volume` are
  zero by design. v0.2.1 injects `settlements = len(self.settlements)`
  into the geopolitics summary so the Nations tab reflects pre-nation
  tribal/clustered activity that does evolve through paleo time.

### Fixed — JEPA agent cognition (`world_model.py`)

- **Deterministic training-batch sampling.** `train_step` used global
  `np.random.choice` for minibatch index selection, coupling training
  reproducibility to whatever else in the simulation last consumed
  `np.random` state. v0.2.1 introduces
  `self._train_rng = np.random.RandomState(seed + 13)` for dedicated,
  deterministic batch sampling. Distributional properties
  (uniform-without-replacement) are unchanged; gradient expectation,
  variance, and SGD convergence are unchanged. The fix removes a
  reproducibility anti-pattern (Pineau et al. 2019) without altering
  scientific behaviour.

### Fixed — Test isolation (`test_agents_lifecycle.py`)

- **Stub leak across test files.** The test file unconditionally
  installed a `_DummyJEPA` stub into `sys.modules['world_model']` at
  import time and never cleaned up. Pytest collects test files in
  alphabetical order, so `test_agents_lifecycle.py` permanently
  replaced `sys.modules['world_model']`; `test_shared_world_model.py`
  then imported the dummy instead of the real `JEPAWorldModel`,
  causing six false failures in the full-suite run that did not
  reproduce when the file was tested in isolation. Fixed by trying
  the real import first and only stubbing on `ImportError`.

### Fixed — Frontend (`templates/index.html`)

- **Tick-handler dashboard updates.** The Macro and Nations panels
  only updated on the websocket `'full_state'` event (every 10 ticks),
  so panels lagged the simulation and could show stale values across
  server restarts. v0.2.1 also calls `updateMacroDashboard` and
  refreshes the geopolitics summary on every `'tick'` event for
  low-latency UI feedback.
- **Sign and unit formatting.** Temperature was rendered as `+${value}`
  unconditionally, producing `+-5.13 °C` for paleo negative
  temperatures. Sea level was always rendered in cm, producing the
  unreadable `-13 000 cm` for LGM Vostok-record sea-level (-130 m).
  v0.2.1 introduces `fmtSigned(v, dp)` (no double-sign on negatives,
  uses `??` for null/undefined detection so a legitimate `0 °C` no
  longer falls back to the default `+1.30`) and `fmtSeaLevel(m)`
  (adaptive cm / m units based on `|m| ≥ 1`).
- **Chart line padding.** `drawLineChart` had only 5 px of top
  padding while labels are drawn at y = 12 with a 10 px font (glyphs
  occupy y ≈ 2..14), so the line crossed through `Max:` and label
  readouts at peak values. Padding extended to 20 px. Applies to all
  five charts that share the helper (climate, tension, population,
  economy, happiness).

### Added — Documentation references

- McEvedy, C. & Jones, R. (1978). *Atlas of World Population History*.
  Penguin.
- Biraben, J.-N. (2003). An essay concerning mankind's evolution.
  *Population & Societies* 394, 1-4.
- Klein Goldewijk, K., Beusen, A., van Drecht, G., & de Vos, M. (2010).
  HYDE 3.1: Long-term dynamic modelling of global population and built-up
  area in a spatially explicit way. *The Holocene* 20, 565-573.
- Adams, J. M. & Faure, H. (1998). A new estimate of changing carbon
  storage on land since the last glacial maximum. *Global and Planetary
  Change* 16-17, 3-24.
- Crowley, T. J. & Baum, S. K. (1997). Effect of vegetation on an
  ice-age climate model simulation. *Journal of Geophysical Research*
  102, 16463-16480.
- Lüthi, D., et al. (2008). High-resolution carbon dioxide concentration
  record 650 000-800 000 years before present. *Nature* 453, 379-382.
- Pineau, J., et al. (2019). Improving Reproducibility in Machine
  Learning Research. *arXiv:1906.06337*.

### Migration notes

- All public APIs are unchanged. `World`, `Agent`, `MacroModel`,
  `GeopoliticalSystem`, `MacroAgentBridge`, `SharedWorldModel`,
  `JEPAWorldModel` retain identical method signatures.
- Internal additions: `World._build_era_aware_summaries()` (private),
  `world._paleo_population_billions()` and `_PALEO_POP_TABLE` (module
  level). `ResourceMap` gained `_baseline_food`, `_baseline_food_regen`,
  `_baseline_wood`, `_baseline_wood_regen`, `_baseline_water`,
  `_was_iced` lazy-initialised fields. `MacroAgentBridge` gained
  `_base_water_regen`, `_base_minerals_regen`, `_base_food_regen`,
  `_base_resource_map_id` lazy-initialised fields.
  `JEPAWorldModel` gained `_train_rng`. None of these are part of any
  public API.
- Saved `experience_buffer` data from v0.2.0 will load correctly; the
  trained weights are unchanged.

## [0.2.0] - 2026-05-06

This release is a **scientific calibration pass** prompted by domain review of
the v0.1.0 implementation. Six categories of bugs were identified — affecting
JEPA training, climate physics, conflict modelling, agent lifecycle scaling,
and the macro/agent coupling layer — and fixed with empirical verification.

The user-facing API surface is unchanged; this is a behavioural correctness
release, not a refactor.

### Fixed — JEPA agent cognition (`world_model.py`, `shared_world_model.py`)

- **Replaced finite-difference gradient estimation with analytic backpropagation.**
  v0.1 used 3 random search directions per weight matrix, which provides an
  unbiased but extremely high-variance gradient estimate; effective learning
  rate scaled as `n_directions / n_params`, meaning ~5500 steps per "real"
  weight update at the encoder's `W2` layer. v0.2 implements hand-written
  backward passes for all primitives (linear, RMSNorm, GELU, AdaLN, residual)
  in pure NumPy, each verified against finite-difference gradient checks
  (relative error < 1e-10).
- **Trained AdaLN parameters.** The action-conditioning weights
  (`W_ada1_scale`, `W_ada1_shift`, `W_ada2_scale`, `W_ada2_shift`) and RMSNorm
  gamma parameters were missing from the trainable set in v0.1. The predictor
  could therefore not learn how actions modify dynamics; CEM planning operated
  on effectively random rollouts. v0.2 trains all parameters and verifies
  empirically that opposite actions produce ~36% relative latent response.
- **SIGReg gradients now flow into the loss.** v0.1 computed the regularizer
  but recomputed only the prediction loss inside the gradient routine, so the
  anti-collapse signal had no training effect. v0.2 uses a moments-based
  variant (skewness² + kurtosis² + variance penalty along random projections)
  that is analytically differentiable, in the spirit of Cramer-Wold gaussianity
  testing but distinct from the original Epps-Pulley formulation.
- **Adam optimizer with gradient clipping** replaces the v0.1 plain SGD step.
- Added Adam state, residual connection (`z_next = z + delta` instead of the
  hardcoded `0.8*z_next + 0.2*z` mix), and DiT-style zero-initialisation of
  AdaLN scale/shift weights (Peebles & Xie 2022).
- Linear-probe `obs_indices` and `obs_scales` made configurable; previous
  hardcoded slot indices `[32, 34, 36]` would silently break if the agent
  observation layout changed.
- `SharedWorldModel` rewritten as a thin wrapper around `JEPAWorldModel`,
  removing duplicated training logic. The vectorised `plan_batch` was
  reimplemented and verified to produce bit-identical results to per-agent
  CEM at the equator (Test: max diff 1e-15).
- **Empirical validation** on a synthetic toy problem with hidden physical
  parameters: prediction loss reduced 103×, latent variance preserved
  (no collapse), linear-probe R² = 0.98 for hidden physics.

### Fixed — Macro climate physics (`macro.py`)

- **Carbon-cycle unit conversion bug.** v0.1 divided by 3.67 (GtCO₂ → GtC) and
  then multiplied by `CO2_PER_GT = 0.128`, but 0.128 is *already* ppm/GtCO₂,
  not ppm/GtC. The conversion was effectively applied twice, underestimating
  the CO₂ rise by a factor of 3.67. The model produced ~0.8 ppm/yr at typical
  2025 emissions, vs. the Mauna Loa observed ~2.5 ppm/yr. v0.2 verifies
  against Mauna Loa decadal mean (NOAA GML 2014–2024).
- **Climate-sensitivity inconsistency resolved.** v0.1 declared
  `CLIMATE_SENSITIVITY = 3.0 °C` but the emergent ECS from the two-layer
  energy balance was `F_2x / λ = 3.708 / 1.1 = 3.37 °C`, at the upper end
  of the IPCC AR6 likely range. v0.2 sets `CLIMATE_FEEDBACK = 1.236` so
  emergent ECS equals the declared 3.0 °C exactly.
- **Carbon sink rate updated to decadal mean.** `NATURAL_ABSORPTION_RATE`
  raised from 0.44 (interannual airborne fraction lower bound) to 0.50
  (decadal mean per Friedlingstein et al. 2024 *Global Carbon Budget*).
- Test ranges tightened: previous BAU validation accepted CO₂ between
  500–800 ppm (>50% bandwidth). v0.2 validates against the IPCC AR6
  SSP2-4.5 to SSP3-7.0 envelope (600–800 ppm at 2100), plus a strict
  carbon-cycle unit test against the Mauna Loa observation.
- Stress-tested for numerical stability over 200-year runs and dt
  sensitivity across 4 orders of magnitude (1 day to 1 year per step):
  identical results.

### Fixed — Geopolitics conflict model (`geopolitics.py`)

- **Conflict prevalence re-calibrated.** v0.1 produced active-conflict
  prevalence of ~99% in a 5-nation BAU run over 75 years. The dominant
  cause was a slow conflict decay (`intensity *= 0.95` with cutoff 0.05
  yields ~45-tick lifetime ≈ 38 years), far exceeding the UCDP/PRIO
  median active-conflict duration of ~3 years (Pettersson 2024). v0.2
  uses `decay = 0.80` (~2.6-year half-life at 10-month tick) and a
  `duration > 25` cap, combined with re-tuned coefficients
  (`CONFLICT_BASE_RATE: −4.5 → −7.5`, `CONFLICT_TENSION_COEFF: 3.0 → 1.5`)
  calibrated against UCDP-style prevalence targets for a 5-nation
  neighbour cluster: ~10–25% / 30–50% / 50–80% at low / mid / high social
  tension. Empirical 5-nation BAU run: 23% / 33% / 55%.
- **Climate-summit cadence bug fixed.** v0.1 used
  `sum(n.age for n in nations) % 24 == 0` to trigger summits, which fires
  every `24 / N` ticks for `N` nations all aging at +1/tick — i.e. every
  5 ticks with 5 nations, not the intended every 24 ticks. v0.2 uses a
  dedicated counter independent of nation count.
- **Haversine distance** replaces euclidean lat/lng for all five
  inter-nation distance calculations (formation merge threshold,
  resource-competition proximity, gravity-trade distance, conflict
  midpoint radius, territorial overlap). At 60°N, a "5-degree distance"
  along longitude was previously distorted by 50%; the new formulation
  preserves the existing degree-equivalent thresholds while correcting
  polar geometry.
- Misleading docstring `"~0.1-0.5% per dyad-year"` removed; new docstring
  references calibration target and theoretical anchors (Russett 1993,
  Bremer 1992, Homer-Dixon 1999, Leeds 2003 ATOP).

### Fixed — Agent lifecycle (`agents.py`)

- **Era-scaled lifecycle thresholds.** v0.1 had hardcoded thresholds in
  ticks (e.g. `age > 40` for reproduction, `reproduction_cooldown = 80`,
  `age > 800` for senescence). With Modern era at 1 month/tick, these
  produced reproduction at 3.3 years (biologically absurd) and a 67-year
  senescence onset; with Paleolithic era at 200 yr/tick, the same
  thresholds compressed to 8 000 / 16 000 / 160 000 years respectively
  (agents never reached reproductive age). v0.2 introduces six lifecycle
  constants in real-world years (`REPRODUCE_MIN_AGE_YEARS = 15.0`,
  `REPRODUCE_COOLDOWN_YEARS = 6.0`, `AGING_THRESHOLD_YEARS = 60.0`,
  `METABOLISM_DOUBLE_YEARS = 666.0`, `LIFESPAN_NORM_YEARS = 80.0`,
  `SOCIALIZE_MIN_AGE_YEARS = 8.0`) and a `_yrs_to_ticks()` helper that
  converts at runtime using the current era's time scale. Modern-era
  drift from previous behaviour is at most 10% (cooldown, aging),
  except reproduction which is corrected from 3.3 → 15 years; in
  Paleolithic, sub-200-year thresholds floor to 1 tick, giving
  generation-per-tick semantics.

### Fixed — World engine (`world.py`)

- **Iteration-mutation safety.** The agent update loop iterated
  `for agent in self.agents`, but `agent.update()` can call
  `world.add_agent(child)` via `_action_reproduce`, appending to
  `self.agents` during iteration. In CPython this is deterministic but
  semantically wrong: newborns received metabolism costs and could act
  in their birth tick. v0.2 iterates over a `list(self.agents)`
  snapshot, deferring newborns to the next tick.
- **Haversine distance** replaces euclidean in `_distance_deg`. This
  method is also used by `bridge.apply_geopolitics_to_agents` for
  conflict-zone proximity, which now benefits from the polar correction
  automatically. The static-method signature is unchanged.

### Fixed — Macro/agent coupling (`bridge.py`)

- **Settlement lookup performance.** The agent-nation lookup in
  `apply_geopolitics_to_agents` was a quartisch nested loop
  `O(N × M × S × K)` over (nations, settlements_per_nation,
  world_settlements, members). v0.2 builds a `settlement_id → Settlement`
  dict once per call and reduces complexity to `O(S + Σ K_s)`. Measured
  speedup: 3.4× at 150 settlements.
- **Hot-path lookup performance.** `get_macro_local_state` is called
  per-agent per-tick from `World.get_local_state`. v0.1 used a
  triple-nested loop (nations → settlement_ids → world.settlements) in
  this hot path; the misleading `break` exited only the innermost loop.
  v0.2 uses the same dict-cached lookup with proper outer-loop exit.
  Measured speedup at 300 agents / 80 settlements: **6.2×**
  (50 ms/tick → 8 ms/tick).
- **Distance consistency.** `get_macro_local_state` previously used raw
  euclidean `np.sqrt(dlat² + dlng²)` while `apply_geopolitics_to_agents`
  in the same module already used `world._distance_deg`. v0.2 routes
  both through `world._distance_deg` (haversine), restoring polar
  correction in conflict-nearby and nation-tech-level lookups.
- **Defensive zero-radius handling.** Conflicts with `radius == 0` no
  longer cause division-by-zero in the proximity damage calculation.
- Behaviour identity verified at agent level: 410-agent run produces
  bit-identical health / wealth / happiness / skill outputs as the v0.1
  algorithm on the same inputs.

### Added — Tests

- `tests/test_world_model_gradcheck.py` — finite-difference gradient
  verification of every backward implementation (linear, GELU, RMSNorm,
  AdaLN, SIGReg). Relative error < 1e-10 on all primitives.
- `tests/test_world_model.py` — end-to-end JEPA training on toy
  dynamics: prediction loss reduction, action-conditioning, anti-collapse,
  linear probe R², CEM planner output validity (5/5).
- `tests/test_shared_world_model.py` — single vs. batch equivalence
  (max diff 1e-15), per-agent vs. plan_batch behavioural identity,
  empty-input edge cases (6/6).
- `tests/test_agents_lifecycle.py` — `_yrs_to_ticks` correctness across
  4 eras, modern-era drift bounds, paleolithic 1-tick floor, default-cache
  safety (7/7).
- `tests/test_geopolitics.py` — haversine correctness, conflict
  monotonicity in tension and trade, 5-nation BAU calibration, summit
  cadence independence from nation count (5/5).
- `tests/test_world.py` — haversine threshold semantics, snapshot
  iteration safety, no-agent-loss invariant (4/4).
- `tests/test_bridge.py` — old-vs-new behavioural identity at 410-agent
  scale, equator parity, polar correction, lookup-build speedup,
  hot-path speedup, edge cases (6/6).
- Tightened `tests/test_macro.py` to validate against IPCC AR6
  SSP2-4.5 to SSP3-7.0 envelope and Mauna Loa decadal mean (9/9 +
  2 unit tests).

### Changed — Documentation

- README updated to reflect v0.2 implementation: replaced JEPA-training
  description (analytic backprop instead of finite differences),
  corrected macro-model parameter table (`λ = 1.236`, absorption = 0.50),
  corrected conflict-model description (UCDP-prevalence calibration
  instead of misleading per-dyad-year claim), expanded test inventory.
- Added a short *Implementation Notes* section documenting the v0.2
  domain-review pass and the calibration philosophy.

### Migration notes

- All public APIs are unchanged. `World`, `Agent`, `MacroModel`,
  `GeopoliticalSystem`, `MacroAgentBridge`, `SharedWorldModel`,
  `JEPAWorldModel` retain identical method signatures.
- Internal: `MacroModel.CO2_PER_GT` renamed to
  `MacroModel.PPM_PER_GTCO2` (the old name's units comment was wrong).
  No external callers found.
- Internal: `WorldEncoder.W1` etc. are now keys in `encoder.params`
  rather than direct attributes. No external callers found.
- Saved `experience_buffer` data from v0.1 will load but the trained
  weights are not portable (different parameter layout).

## [0.1.0] - 2026-05-03

### Added

- Initial public release under AGPL-3.0-or-later.
- **Agent cognition** — JEPA world model (LeCun 2022; Maes et al. 2026): encoder, AdaLN predictor, SIGReg regularization, CEM planner. Shared model for batch inference across all agents.
- **Macro layer** — 14-state ODE system: climate (IPCC AR6 calibrated, 9/9 validation checks), Hubbert resource depletion, DICE damage function, Earth4All social tension, endogenous Romer technology growth.
- **Geopolitics** — Emergent nation-states from settlement coalescence, gravity-model trade (Tinbergen 1962), International Futures conflict probability with liberal-peace coupling.
- **Scenarios** — (A) 70,000-year historical from Out-of-Africa with paleoclimate, Diamond geographic determinism, and Dawkins evolutionary adaptation; (B) present-day initialized from World Bank, NOAA, and NASA data.
- **Earth system** — Real geography from Natural Earth (110m), Whittaker biome diagram, USGS resource provinces, FAO GAEZ-inspired soil fertility.
- **LLM integration** — Optional System-2 cognition for trade negotiation, governance speech, and social dialogue (Ollama / OpenAI / Mistral compatible).
- **Reproducibility** — Seeded determinism via `world.seed`, structured run logging via `sim_logger.py` writing JSON metadata + per-tick CSV.
- **Web frontend** — Flask + SocketIO real-time visualization with Leaflet.js satellite/OSM tiles.

### Security

- **Path-traversal hardening** of `/api/logger/download/<run_id>`: `run_id` is regex-validated to the `YYYYMMDD_HHMMSS` format produced by `sim_logger.py`, and the resolved path is verified to stay under the repository's `logs/` directory.
- **Loopback-by-default**: the server now binds to `127.0.0.1`. Set `BIND_HOST=0.0.0.0` explicitly to expose the unauthenticated control surface (a warning is printed in that case). Place behind an authenticated reverse proxy before exposing publicly.
- **CORS restricted** to `http://localhost:5000` and `http://127.0.0.1:5000` by default, replacing the prior allow-all policy. Override via `CORS_ALLOWED_ORIGINS` (comma-separated).
- **Ephemeral secret key**: when `FLASK_SECRET_KEY` is unset, a per-process `secrets.token_hex(32)` is generated instead of falling back to a predictable shared default. A warning is printed at startup.
- **`SECURITY.md`** added — threat model, in-scope vs. out-of-scope mitigations, operator deployment checklist, and a vulnerability-disclosure process via GitHub's private vulnerability reporting.
- Verified that `LLMModule.api_key` is not leaked through `get_status()`, `get_full_state()`, snapshot dumps, or `metadata.json`.

[Unreleased]: https://github.com/GeoLambdaAI/oikoumene/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/GeoLambdaAI/oikoumene/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/GeoLambdaAI/oikoumene/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GeoLambdaAI/oikoumene/releases/tag/v0.1.0
