---
title: "Oikoumene: A Multi-Scale Agent-Based Simulator of Human Civilization with JEPA Cognition and Coupled Earth-System Dynamics"
tags:
  - Python
  - agent-based modelling
  - climate simulation
  - JEPA
  - world model
  - macroeconomics
  - geopolitics
  - paleoclimate
authors:
  - name: Gerrit Tombrink
    affiliation: 1
    corresponding: true
affiliations:
  - name: GeoLambda GmbH, Germany
    index: 1
date: 6 May 2026
bibliography: paper.bib
---

# Summary

`Oikoumene` is an open-source simulator that couples autonomous agents to
a coupled Earth-system, macroeconomic, and geopolitical model on real Earth
geography. Each agent perceives, plans, and acts through a Joint Embedding
Predictive Architecture (JEPA) world model [@lecun2022path; @maes2026leworldmodel],
while a 14-state ordinary-differential-equation (ODE) layer governs global
climate, resources, and socioeconomics. Nation-states emerge organically when
agent settlements coalesce, with bilateral trade and conflict probability
following established gravity-model and International Futures specifications.

The simulator ships with two scenarios. *Scenario A* runs from the
Out-of-Africa migration ~70,000 years before present through to climate
futures beyond 2100, integrating paleoclimate from EPICA ice cores
[@luthi2008high; @epica2004eight] and geographic determinism in the sense of
@diamond1997guns. *Scenario B*
initialises from present-day data (World Bank, NOAA, NASA) and projects
forward at monthly resolution. Every parameter, equation, and boundary
condition is grounded in published literature, and the macro layer passes
9/9 IPCC AR6-calibrated validation checks plus two unit anchors (Mauna
Loa decadal CO₂ growth rate; emergent equilibrium climate sensitivity)
at 2100.

# Statement of need

Existing tools occupy disjoint corners of the design space: integrated
assessment models such as DICE [@nordhaus2017revisiting] and World3
[@meadows1972limits] deliver coherent climate-economic dynamics but treat
populations as aggregate stocks; agent-based models such as Sugarscape and
Schelling-style frameworks deliver emergent behaviour but lack physical
grounding; large language model agents now match human conversational
fidelity but have no persistent world model. `World Genesis` bridges these
traditions: a single shared JEPA world model supplies vector cognition to
hundreds of agents simultaneously, while a calibrated macro ODE supplies
the physical and economic constraints those agents inhabit. The result is
a research instrument suitable for studying emergent civilisation
dynamics, climate-policy counterfactuals, and the coupling between
individual decisions and planetary boundaries.

The package targets three audiences. First, **climate and Earth-system
researchers** who want a tractable testbed for integrated assessment
beyond representative-agent assumptions. Second, **AI researchers** studying
JEPA-style architectures [@maes2026leworldmodel; @qu2026representation] in a
non-toy setting with reward-free exploration over thousands of episodes.
Third, **economists and political scientists** investigating emergent
nation-formation, trade, and conflict from primitive agent interactions
[@hughes2019international; @russett1993grasping].

# Features

- **JEPA cognition** with adaptive layer normalisation, SIGReg
  regularisation [@maes2026leworldmodel], and a Cross-Entropy Method planner.
- **14-state macro ODE** spanning two-layer energy balance, Hubbert resource
  curves [@hubbert1956nuclear], DICE damages [@nordhaus2017revisiting], and
  Earth4All social tension [@dixsondecleve2022earth].
- **Emergent geopolitics**: nation formation by settlement coalescence;
  Tinbergen [@tinbergen1962shaping] gravity trade; logistic conflict
  probability per @hughes2019international.
- **Real Earth geography**: Natural Earth coastlines rasterised to 0.25°,
  Whittaker [@whittaker1975communities] biomes, USGS resource provinces.
- **Reproducibility**: deterministic seeded runs (per-instance random
  generators across encoder, predictor, planner, and minibatch sampling
  decouple training from global state), structured CSV+JSON logging, a
  9/9 IPCC validation suite plus two unit anchors, a figure-generation
  pipeline, and a published validation report.

# Validation

Running the BAU macro path 2025–2100 produces CO₂ = 679 ppm, ΔT = +2.74 °C,
sea-level rise = 0.61 m, and population = 8.37 B, all within bounds derived
from the IPCC AR6 SSP2-4.5 to SSP3-7.0 envelope and the Vollset (2020) /
UN WPP 2024 demographic range. The CO₂ growth rate around 2030 is 2.58 ppm/yr,
inside the NOAA GML Mauna Loa decadal mean envelope of 2.4–3.0 ppm/yr; the
emergent equilibrium climate sensitivity is 3.00 °C, matching the declared
IPCC AR6 best estimate exactly via the calibrated climate-feedback parameter
λ = F<sub>2x</sub>/ECS = 1.236 W m<sup>-2</sup> K<sup>-1</sup>. Full reference
values, tolerance bands, and the reproduction command appear in
[`docs/validation.md`](../docs/validation.md).

# Acknowledgements

This work was developed at GeoLambda GmbH. The author thanks the open
research communities behind Natural Earth, the World Bank Open Data
initiative, NOAA GML, NASA GISS, and the IPCC for the data infrastructure
that makes calibration possible.

# References
