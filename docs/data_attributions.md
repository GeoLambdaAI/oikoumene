# Third-Party Data Attributions

World Genesis ships pre-computed Earth-system data derived from public sources.
This document lists every dataset, its license, and the attribution required
when redistributing or citing simulation outputs.

---

## Geographic base data

### Natural Earth (110 m cultural & physical)
- **Files:** `data/ne_110m_land.geojson`, `data/ne_110m_rivers.geojson`,
  `data/ne_110m_lakes.geojson`, `data/landmask.npy` (derived).
- **Source:** [naturalearthdata.com](https://www.naturalearthdata.com/)
- **Licence:** Public domain. No restriction on use.
- **Attribution suggested:** "Made with Natural Earth."

### Natural Earth 10 m rivers (drop-in resolution upgrade)
- **Files:** `data/empirical/ne_10m_rivers_lake_centerlines.geojson` —
  finer global river network than the shipped 110 m vector. Same data
  publisher and same licence as above. Fetched by
  `generate_empirical_inputs.py --ne-rivers` from the Natural Earth CDN
  (`https://naciscdn.org/naturalearth/10m/physical/`, verified 2026-06-21).
- **Licence:** Public domain (same as Natural Earth 110 m).
- **Attribution suggested:** "Made with Natural Earth."

### ETOPO 2022 (NOAA NCEI global relief at 60 arc-sec)
- **Files:** `data/empirical/etopo_elevation.npy` — global elevation
  resampled to the simulator's 0.5° grid (metres above the EGM2008 geoid;
  ocean values are negative). Replaces the synthetic `earth_elevation.npy`
  produced by `generate_earth_data.py` (currently derived from
  tectonic-plate-boundary heuristics). Fetched by
  `generate_empirical_inputs.py --etopo`.
- **Source:** NOAA National Centers for Environmental Information; THREDDS
  file server at
  `https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/`. Single
  global NetCDF file at 60-arc-sec resolution (~395 MB, one-time download).
- **Licence:** **US Government public domain** — verified verbatim in the
  ETOPO 2022 User Guide §4 (Dataset Usage and Citation): *"ETOPO tiles are
  freely available to use for all private, academic, or commercial
  purposes."* No restriction on use or redistribution.
- **Attribution required (citation):**
  NOAA National Centers for Environmental Information (2022). *ETOPO 2022
  15 Arc-Second Global Relief Model.* NOAA National Centers for
  Environmental Information. https://doi.org/10.25921/fd45-gt74.

### Synthetic Earth-system grids (heuristic fallback)
The `data/earth_*.npy` rasters (terrain, biome, elevation, temperature,
precipitation, fertility, minerals, freshwater, fossil fuels) produced by
`generate_earth_data.py` are **heuristic/synthetic**, not observations. They let
the world build offline and are replaced layer-by-layer by the empirical sources
above (ETOPO, CHELSA, SoilGrids, …) when fetched. Known residual limitations of
the synthetic layers:
- The latitude–temperature law is calibrated to ERA5 zonal means (v0.3 fixed a
  ~7 °C polar warm bias), so Antarctica now classifies as ice sheet. **Greenland’s
  interior may still read as tundra/grassland** because the *synthetic elevation*
  model does not represent its ice plateau — use the empirical ETOPO layer for a
  faithful Greenland.
- `data/present_day_*` population/resource grids use real per-country totals but a
  *synthetic spatial distribution* (Gaussian smear around centroids); the real
  gridded source is HYDE 3.2.1 (below), which is downloaded but not yet wired into
  the spatial grid.

## Climate & atmosphere

### NOAA Mauna Loa CO₂ record
- **Use:** Calibration of `MacroModel.CO2_PRE_INDUSTRIAL`, present-day initial
  condition (`co2_ppm = 427.0` for 2025).
- **Source:** [NOAA Global Monitoring Laboratory](https://gml.noaa.gov/ccgg/trends/)
- **Licence:** US Government public domain.
- **Attribution required:** Cite Friedlingstein et al. (2024) for the Global
  Carbon Budget figures used in emissions calibration.

### NASA GISTEMP v4 (surface temperature)
- **Use:** Present-day temperature anomaly initial condition (+1.19 °C).
- **Source:** [data.giss.nasa.gov/gistemp](https://data.giss.nasa.gov/gistemp/)
- **Licence:** US Government public domain.
- **Attribution suggested:** "GISS Surface Temperature Analysis (GISTEMP),
  version 4."

### HadCRUT.5.1.0.0 (Met Office Hadley Centre + CRU)
- **Use:** `data/empirical/hadcrut5_global_annual.npy` — global mean annual
  surface-temperature anomaly time series 1850 → present (anomaly relative
  to 1961-1990 baseline, with ensemble / coverage / total 1-σ uncertainties).
  Acts as a **second independent observational anchor** next to NASA GISTEMP
  for the macro temperature track; consumed by the
  `test_macro_hadcrut5_consistency` validation test. Fetched by
  `generate_empirical_inputs.py --hadcrut5`.
- **Source:** Met Office Hadley Centre + Climatic Research Unit (UEA).
  Download index:
  [metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/download.html](https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/download.html);
  CSV at `…/analysis/diagnostics/HadCRUT.5.1.0.0.analysis.component_series.global.annual.csv`
  (~10 KB, verified 2026-05-25).
- **Licence:** **Open Government Licence v3.0** (UK Crown copyright, open).
  Free of charge, redistribution and commercial use permitted with
  attribution; AGPL-compatible. Full text:
  https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
- **Attribution required:**
  Morice, C. P., Kennedy, J. J., Rayner, N. A., et al. (2021).
  *An updated assessment of near-surface temperature change from 1850: the
  HadCRUT5 data set.* *Journal of Geophysical Research: Atmospheres* 126,
  e2019JD032361. https://doi.org/10.1029/2019JD032361.
  Met Office Hadley Centre & Climatic Research Unit, University of East
  Anglia. Contains public sector information licensed under the Open
  Government Licence v3.0.
- **Baseline note:** HadCRUT5 anomalies are referenced to the **1961-1990**
  mean, while GISTEMP uses **1951-1980**. Tests comparing the two account
  for the ~0.10-0.15 °C baseline offset.
- **Completeness note:** the downloader (`_drop_incomplete_trailing_years`)
  drops any trailing *provisional* in-progress calendar year (its coverage
  uncertainty is sharply inflated vs complete years), so the saved series ends
  at the last complete annual mean. If you ship a stale copy, re-fetch with
  `python generate_empirical_inputs.py --hadcrut5` to refresh it.

### IPCC AR6 WG1 calibration values
- **Use:** Climate sensitivity (3.0 °C/2×CO₂), feedback parameter (1.1 W/m²/°C),
  CO₂ forcing coefficient (5.35 W/m²; Myhre et al. 1998).
- **Source:** [IPCC AR6 WG1 Table 7.SM.1](https://www.ipcc.ch/report/ar6/wg1/)
- **Licence:** IPCC reports may be reproduced for non-commercial purposes
  with attribution. Calibration values used here are common scientific
  knowledge and require only standard citation.

### CHELSA V2.1 (high-resolution global climatology)
- **Use:** `data/empirical/chelsa_<var>_annual.npy` — present-day annual
  temperature / precipitation rasters that replace the heuristic
  `earth_temperature.npy` / `earth_precipitation.npy` produced by
  `generate_earth_data.py`. Fetched and resampled by
  `generate_empirical_inputs.py --chelsa`.
- **Source:** [chelsa-climate.org](https://chelsa-climate.org/) — archived at
  [EnviDat](https://doi.org/10.16904/envidat.228); data bucket at
  `https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/climatologies/`.
- **Licence:** **Creative Commons Zero 1.0 (CC0 1.0, public domain)** —
  verified on the EnviDat record on 2026-05-25. No restriction on use or
  redistribution.
- **Attribution suggested (scholarly):**
  Karger, D. N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza,
  R. W., Zimmermann, N. E., Linder, H. P., & Kessler, M. (2021).
  *Climatologies at high resolution for the earth's land surface areas.*
  EnviDat. https://doi.org/10.16904/envidat.228.
  Methodology: Karger et al. (2017), *Scientific Data* 4, 170122.
  https://doi.org/10.1038/sdata.2017.122.

## Soils

### SoilGrids 2.0 (global soil properties at 250 m)
- **Use:** `data/empirical/soilgrids_fertility.npy` — a composite root-zone
  (0–5 cm) fertility index derived from soil organic carbon, nitrogen and
  clay fraction. Replaces the heuristic `earth_fertility.npy` produced by
  `generate_earth_data.py`. Fetched by
  `generate_empirical_inputs.py --soilgrids`.
- **Source:** [ISRIC SoilGrids](https://soilgrids.org/) — documented at
  [docs.isric.org/globaldata/soilgrids](https://docs.isric.org/globaldata/soilgrids/);
  WebDAV at `https://files.isric.org/soilgrids/latest/data/`.
- **Licence:** **Creative Commons Attribution 4.0 International (CC-BY 4.0)**
  — verified on the ISRIC SoilGrids page on 2026-05-25. Redistribution and
  commercial use allowed *with attribution*.
- **Attribution required:**
  Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen,
  B., Ribeiro, E., & Rossiter, D. (2021). *SoilGrids 2.0: producing soil
  information for the globe with quantified spatial uncertainty.* *SOIL*
  7(1), 217–240. https://doi.org/10.5194/soil-7-217-2021. © ISRIC — World
  Soil Information, licensed CC-BY 4.0.

## Paleoclimate

### EPICA / Vostok Antarctic ice core CO₂ (800 ka)
- Lüthi, D. et al. (2008). *High-resolution carbon dioxide concentration record
  650,000–800,000 years before present.* Nature 453, 379–382 (EPICA Dome C, the
  800-ka CO₂ record). Petit, J. R. et al. (1999, Vostok) and EPICA Community
  Members (2004) for the longer temperature/δD context.
- **Source:** PANGAEA / NOAA Paleoclimatology archives.
- **Licence:** Open scientific data; cite the original papers.
- **How it is used (important):** `history.py`'s `PALEOCLIMATE_DATA` is **not** a
  transcription of these ice-core series. It is a small, hand-curated set of ~25
  representative anchor points (year_bp, CO₂, ΔT, sea level) *informed by* the
  above records, linearly interpolated between anchors. Treat it as a stylised
  reconstruction for the simulation timeline, not the raw published data.

### Spratt & Lisiecki (2016) sea-level stack
- **Use:** Paleoclimate scenario sea level prior to 1850.
- **Citation:** Spratt, R. M. & Lisiecki, L. E. (2016). *Climate of the Past*
  12, 1079–1092. (CC-BY).

## Resources & geology

### USGS Mineral Commodity Summaries / Petroleum Assessment
- **Use:** `data/earth_minerals.npy`, `data/earth_fossil_fuels.npy` —
  province location and rough magnitudes.
- **Source:** [usgs.gov](https://www.usgs.gov/)
- **Licence:** US Government public domain.

### FAO GAEZ (Global Agro-Ecological Zones) — methodology only
- **Use:** Inspiration for `data/earth_fertility.npy` derivation; no FAO
  raster data is shipped, only the methodological approach (biome ×
  precipitation × temperature × known breadbaskets).
- **Citation:** Licker et al. (2010); Mueller et al. (2012).

## Socio-economic

### World Bank Open Data
- **Use:** `data/present_day_countries.json` — GDP, population, area
  per country circa 2024.
- **Source:** [data.worldbank.org](https://data.worldbank.org/)
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Attribution required:** Cite "World Bank Open Data" with retrieval date.
  Files are fetched fresh by `generate_present_day_data.py` — note the
  retrieval date in `present_day_metadata.json`.

### HYDE 3.2.1 (History Database of the Global Environment)
- **Use:** `data/empirical/hyde_{popc,cropland,grazing}/<year>.npy` — gridded
  population and land-use rasters for selected years, used as the *primary*
  historical validation track for emergent demographics and as a spatial
  initial condition for Scenario A. Fetched by
  `generate_empirical_inputs.py --hyde --years …`.
- **Source:** [HYDE 3.2 on DANS](https://doi.org/10.17026/DANS-25G-GEZ3)
  (the authoritative archive; the PBL homepage links here). Files served
  via the standard Dataverse API at
  `https://archaeology.datastations.nl/api/access/datafile/<fileId>`; the
  main `HYDE3_2_1-baseline.zip` (fileId 5490328) is ~5.3 GB.
- **Coverage:** **10 000 BCE → 2015 CE** at 5 arc-minute (~10 km) resolution,
  with time-step intervals of 1000 yr (paleo) → 100 yr (1–1700 CE) →
  10 yr (1700–2000 CE) → 1 yr (2000–2015 CE). This temporal scope matches
  Scenario A's Out-of-Africa starting window.
- **Licence:** **Creative Commons Zero 1.0 (CC0 1.0, public domain)** —
  verified on the DANS data-station record on 2026-05-25. No restriction
  on use or redistribution.
- **Attribution suggested (scholarly):**
  Klein Goldewijk, K. (2017). *Anthropogenic land-use estimates for the
  Holocene; HYDE 3.2.* DANS. https://doi.org/10.17026/DANS-25G-GEZ3.
  Methodology: Klein Goldewijk, K., Beusen, A., Doelman, J., & Stehfest, E.
  (2017). *Anthropogenic land use estimates for the Holocene – HYDE 3.2.*
  *Earth System Science Data* 9, 927–953.
  https://doi.org/10.5194/essd-9-927-2017.

### UN WPP / Conflict trackers
- Active-conflict geolocations in `data/present_day_conflicts.json` are
  hand-curated from publicly reported incidents (Ukraine, Gaza, Sudan,
  Myanmar, etc.). For research output, cite ACLED or UCDP for canonical
  conflict datasets — the shipped file is illustrative only.

## Conflict — empirical anchor

### UCDP-GED v25.1 (Uppsala Conflict Data Program — Georeferenced Event Dataset)
- **Files:** `data/empirical/ucdp_ged_events.npy` — a structured NumPy
  array with named fields `(year, lat, lng, best, conflict_id)` over the
  full UCDP-GED 1989-present event series. **Empirical validation anchor**
  for the macro/geopolitics emergent conflict layer; replaces the role of
  the illustrative `present_day_conflicts.json`. Fetched by
  `generate_empirical_inputs.py --ucdp-ged`.
- **Source:** Uppsala Conflict Data Program (Department of Peace and
  Conflict Research, Uppsala University). Downloads index:
  [ucdp.uu.se/downloads](https://ucdp.uu.se/downloads/); current ZIP at
  `https://ucdp.uu.se/downloads/ged/ged251-csv.zip` (~29 MB, verified
  2026-06-21).
- **Licence:** **Creative Commons Attribution 4.0 International (CC-BY 4.0)**
  — verified verbatim on the UCDP downloads page: *"All datasets are free
  of charge and licensed under CC BY 4.0."* Redistribution and commercial
  use permitted with attribution; AGPL-compatible. No registration required.
- **Attribution required (cite both):**
  Davies, S., Pettersson, T., & Öberg, M. (2026). *Organized violence
  1989-2025 and the prospects for peace.* *Journal of Peace Research*
  (annual update issue).
  Sundberg, R., & Melander, E. (2013). *Introducing the UCDP Georeferenced
  Event Dataset.* *Journal of Peace Research* 50(4), 523-532.
  https://doi.org/10.1177/0022343313484347.

---

## How to cite the simulation outputs

If you publish results derived from running World Genesis, please cite:

1. **The software** — see [`CITATION.cff`](../CITATION.cff) (auto-rendered
   on GitHub as a "Cite this repository" button).
2. **The relevant primary sources** — every equation in the codebase carries
   an inline citation; the master bibliography is in
   [`paper/paper.bib`](../paper/paper.bib).
3. **The pre-computed data sources above**, especially World Bank if you
   use Scenario B.

## License compatibility

The simulation code is released under **AGPL-3.0-or-later**. The shipped
and download-on-demand data files are either public domain (Natural Earth
110m/10m, US Government sources, **ETOPO 2022**, **CHELSA V2.1**,
**HYDE 3.2.1**), CC-BY 4.0 (World Bank, **SoilGrids 2.0**, **UCDP-GED**),
or Open Government Licence v3.0 (**HadCRUT.5.1.0.0**) — all compatible
with AGPL redistribution. Derived works must preserve attribution to the
original data sources even where the AGPL covers the code. The empirical
datasets ingested by `generate_empirical_inputs.py` are not shipped with
the repository — they are streamed/fetched from their official archives
so users always get the canonical, currently-licensed version.
