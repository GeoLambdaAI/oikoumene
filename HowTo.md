# HowTo — Running Oikoumene

A practical, copy-pastable guide. The [`README.md`](README.md) explains
*what* the simulator does and *why*; this file explains *how* to drive it.

> **Audience:** anyone who just cloned this repo and wants to see agents
> migrating across Africa within ten minutes. No prior knowledge of the
> codebase is assumed.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [First-time setup](#2-first-time-setup)
3. [Generate the Earth data](#3-generate-the-earth-data)
4. [Run the simulator](#4-run-the-simulator)
5. [Configure the runtime via environment variables](#5-configure-the-runtime-via-environment-variables)
6. [Using the web UI](#6-using-the-web-ui)
7. [Optional: enable LLM social cognition](#7-optional-enable-llm-social-cognition)
8. [Logs and analysis](#8-logs-and-analysis)
9. [Run the test suites](#9-run-the-test-suites)
10. [Stopping the simulator](#10-stopping-the-simulator)
11. [Troubleshooting](#11-troubleshooting)
12. [Where to next](#12-where-to-next)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11, 3.12, or 3.13** | `python --version` must print one of these. macOS, recommended via Conda |
| **~50 MB free disk** | 17 MB pre-computed Earth data + ~25 MB Python deps + headroom for run logs |
| **Modern browser** | Chrome, Firefox, Safari, Edge — anything that runs Leaflet.js |
| **Internet access (one-off)** | Only required by `generate_present_day_data.py` to fetch World Bank / NOAA / NASA values. Everything afterwards is offline. |
| **Optional: Ollama / Mistral / OpenAI** | For LLM-driven agent dialogue. The simulator runs without it. |

---

## 2. First-time setup

```bash
# From a fresh clone:
cd oikoumene-main

# Always use a virtual environment — the simulator pulls in eventlet,
# which monkey-patches the standard library and will surprise you in a
# global interpreter.
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# Runtime dependencies
pip install -r requirements.txt
```

Two **optional** dependency groups, installed via the `pyproject.toml`:

```bash
pip install -e ".[viz]"   # adds matplotlib + pandas (for scripts/figures.py)
pip install -e ".[dev]"   # adds pytest, pytest-cov, ruff, mypy
```

You can install both: `pip install -e ".[viz,dev]"`.

## Installation (macOS, recommended via Conda)

```bash
conda create -n oikoumene python=3.11 -y
conda activate oikoumene
pip install -r requirements.txt
```
---

## 3. Generate the Earth data

A **one-time** step. Run all three scripts from the repository root:

```bash
python generate_landmask.py          # ~10s  — rasterises Natural Earth coastlines to 0.25°
python generate_earth_data.py        # ~1s   — climate / biomes / resources at 0.5°
python generate_present_day_data.py  # ~15s  — fetches World Bank / NOAA / NASA (needs internet)
```

Output appears in [`data/`](data/) — `landmask.npy`, `earth_*.npy`, `ne_110m_*.geojson`,
and the `present_day_*.json/.npz` files for Scenario B.

If `generate_present_day_data.py` cannot reach an upstream API (rate
limit, offline, etc.), it falls back to bundled defaults. **Scenario B
will still work** with 2025 baseline values: CO₂ = 427 ppm, ΔT = +1.19 °C.

---

## 4. Run the simulator

### Default (loopback only — safest)

```bash
python app.py
```

You should see:

```
============================================================
  Oikoumene — Earth
  Open http://localhost:5000 in your browser
============================================================
```

Open the URL, choose a scenario, click **Start**.

### Scenario quick-reference

| Scenario | Starts at | Time scale | What you'll see |
|---|---|---|---|
| **A — 70,000 Years of Human History** | ~68,000 BCE in East Africa | 200 yrs/tick → 1 month/tick | Out of Africa migration, agriculture in the Fertile Crescent, civilisations rising/falling, Industrial Revolution kicking the macro ODE in around tick ~10,000 |
| **B — Present Day → Future** | 2025 CE | 1 month/tick | 300 agents distributed by real population density, 140 nations, 10 active conflicts, climate evolving from CO₂=427 ppm |

Both scenarios use the same code path; the difference is initial conditions.

---

## 5. Configure the runtime via environment variables

| Variable | Default | What it controls |
|---|---|---|
| `PORT` | `5000` | HTTP and WebSocket port |
| `BIND_HOST` | `127.0.0.1` | Listen address. **Only set to `0.0.0.0` behind an authenticated reverse proxy** — the simulator endpoints are unauthenticated by design (see [SECURITY.md](SECURITY.md)). |
| `FLASK_SECRET_KEY` | random per process | Flask session signing key. If unset, an ephemeral key is generated at startup with a `[security]` warning. Set explicitly for reproducible sessions across restarts. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5000,http://127.0.0.1:5000` | Comma-separated origin allowlist for the SocketIO API |

### Common configurations

**Different port:**

```bash
PORT=8080 python app.py
```

**LAN access (behind an authenticated proxy only — read [SECURITY.md](SECURITY.md) first):**

```bash
BIND_HOST=0.0.0.0 python app.py
# Server prints WARNING: bound to 0.0.0.0 — reachable from the network.
```

**Stable production-ish setup:**

```bash
export FLASK_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export CORS_ALLOWED_ORIGINS=https://my-frontend.example.com
export PORT=5000
python app.py
```

Persist these in a `.envrc` (direnv) or your shell profile — never commit them.

---

## 6. Using the web UI

<p align="center">
  <a href="static/oikoumene.jpg">
    <img src="static/oikoumene.jpg" alt="Oikoumene web UI — global map with agents, top bar statistics, side panel with agent cognition and nation feed" width="800">
  </a>
  <br>
  <sub><em>The Present-Day scenario in motion. Top bar = live macro statistics; map = agents on real Earth; right panel = live JEPA cognition, press feed, and discussions.</em></sub>
</p>

| Element | What it does |
|---|---|
| **Top bar** | Scenario picker · Start / Stop / Reset · speed slider (ticks per second) |
| **Map** | Agents = dots, settlements = buildings, conflicts = highlighted cells. Pan = drag, zoom = scroll. Toggle satellite vs. OSM tiles in the corner. |
| **Side panel — macro** | CO₂, ΔT, sea level, population, GDP, social tension, technology level (live) |
| **Side panel — nations** | Emergent nations sorted by power; click for trade / conflict detail |
| **Side panel — dialogues** | Recent LLM-generated agent conversations (only populated when LLM is enabled) |
| **Click any agent** | Opens detail pane: traits (intelligence, cooperation, ambition, curiosity), memory, recent actions, ancestry |
| **God mode** (toggle bottom of side panel) | Operator interventions: whisper goals to an agent, drop droughts/plagues by lat/lng, grant technology, nudge climate. **Available to anyone with HTTP access** — see SECURITY.md |

---

## 7. Optional: enable LLM social cognition

Agents make trade negotiations, governance speeches, and social dialogue
via a Large Language Model (the "System 2" path in Kahneman's Dual-Process
Theory). With LLM disabled, agents fall back to rule-based behaviour and
the simulation runs faster.

### Option A — local Ollama (offline, private)

```bash
# 1. Install Ollama → https://ollama.com/
# 2. Pull a small model:
ollama pull llama3        # or qwen2.5, mistral, gemma2
# 3. Keep Ollama running:
ollama serve
```

In the UI side panel:

- Provider: **Ollama**
- Base URL: `http://localhost:11434`
- Model: `llama3` (whatever you pulled)
- Click **Test** — should return latency in milliseconds and a list of
  available models.

### Option B — Mistral AI / OpenAI

1. Obtain an API key:
   - Mistral → [console.mistral.ai](https://console.mistral.ai)
   - OpenAI → [platform.openai.com](https://platform.openai.com)
2. In the UI:
   - Provider: **Mistral** or **OpenAI**
   - Base URL: `https://api.mistral.ai` or `https://api.openai.com`
   - Model: e.g. `mistral-small-latest` or `gpt-4o-mini`
   - Paste API key.
3. Click **Test**.

The key is stored only in the running process — it is not logged, dumped
to `metadata.json`, sent over the WebSocket to the browser, or written
to snapshot JSON. (Verified: see SECURITY.md §"Verified-safe".)

### Tuning LLM cost / latency

In the UI:

- **`max_calls_per_tick`** — caps API calls per tick. Default is low; set
  to 0 to disable LLM calls without changing provider config.
- **`temperature`** — usual LLM creativity knob.

---

## 8. Logs and analysis

Every run writes to `logs/{run_id}/` where `run_id = YYYYMMDD_HHMMSS`:

| File | Contents |
|---|---|
| `metadata.json` | Seed, scenario config, start/end timestamps, column descriptions |
| `timeseries.csv` | One row per logged tick, ~60 columns spanning population, macro state, geopolitics, agent traits, performance |
| `snapshots/tick_NNNNNNNN.json` | Full world state dumps at checkpoint intervals (default every 100 ticks) |

### Generate paper-ready figures

```bash
pip install -e ".[viz]"
python scripts/figures.py                            # latest run
python scripts/figures.py --run-id 20260505_142536   # specific run
python scripts/figures.py --list                     # all available runs
```

Output appears in `logs/{run_id}/figures/`. Six standard plots:

- `macro_climate.png` — CO₂, ΔT, sea level
- `macro_resources.png` — fossil, minerals, freshwater, food
- `macro_society.png` — population, GDP, inequality, welfare
- `agents_population.png` — alive count, births, deaths
- `geopolitics.png` — nations, conflicts, trade volume
- `ipcc_validation.png` — simulation overlaid on IPCC SSP1-2.6 → SSP5-8.5

### Download a CSV via the API

```bash
curl http://localhost:5000/api/logger/download/20260505_142536 -o run.csv
```

The `run_id` is regex-validated against the timestamp format — arbitrary
paths return HTTP 400. (See [SECURITY.md](SECURITY.md) for the
path-traversal mitigation.)

### Reading the CSV elsewhere

```python
import pandas as pd
df = pd.read_csv("logs/20260505_142536/timeseries.csv")
df[["tick", "macro_co2_ppm", "macro_temperature", "population"]].plot(x="tick")
```

---

## 9. Run the test suites

```bash
python test_macro.py          # 9/9 — IPCC AR6 calibration on BAU 2025→2100
python test_agent_state.py    # 4/4 PASS + benchmarks across 500–2000 agents
python test_llm_module.py     # 9/9 — fallback mode, JSON parsing, rate limiting
```

Or use `pytest`:

```bash
pip install -e ".[dev]"
pytest -v
pytest --cov=. --cov-report=term-missing
```

CI runs the same suites across Python 3.11, 3.12, and 3.13 on every push
(see [`.github/workflows/test.yml`](.github/workflows/test.yml)).

---

## 10. Stopping the simulator

In the terminal where `app.py` is running: `Ctrl-C`.

If eventlet does not exit on the first interrupt (it sometimes holds a
green-thread waiting on a socket), press `Ctrl-C` a second time. As a
last resort:

```bash
lsof -i :5000          # find the PID
kill -9 <pid>          # only if Ctrl-C twice did not work
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Virtual env not activated | `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows) |
| `Address already in use` on port 5000 | Previous instance still bound | `lsof -i :5000` then `kill <pid>` — or run with `PORT=5001 python app.py` |
| Browser shows blank map | Cached old JS / CDN block | Hard reload (Cmd-Shift-R / Ctrl-Shift-R). Open browser console for the real error. |
| Agents do not appear after Start | Server not actually running, or scenario data missing | Check terminal for tracebacks; rerun the three `generate_*.py` scripts |
| `generate_present_day_data.py` exits with API errors | World Bank / NOAA / NASA rate-limited or down | Wait a few minutes and retry. The simulator runs fine without it; defaults kick in. |
| `eventlet` `ImportError` after Python upgrade | Compiled extensions invalidated | `pip install --upgrade --force-reinstall eventlet` |
| LLM Test → "Connection refused" | Ollama daemon not running | `ollama serve` in another terminal |
| LLM Test → 401 Unauthorized | Wrong / expired API key | Regenerate key at the provider console; paste again; **Test** |
| `FLASK_SECRET_KEY unset` warning at startup | Expected — informational | Set the env var if you want sessions to survive restarts |
| `WARNING: bound to 0.0.0.0` at startup | You set `BIND_HOST=0.0.0.0` | Expected only if you intentionally placed an authenticated proxy in front |
| Tests fail with `FileNotFoundError: data/landmask.npy` | Earth data not generated | Run the three scripts in §3 |
| Figures script fails with "Missing visualization dependencies" | viz deps not installed | `pip install -e ".[viz]"` |

---

## 12. Where to next

- [`README.md`](README.md) — scientific foundations, equations, references
- [`docs/validation.md`](docs/validation.md) — IPCC calibration evidence (9/9)
- [`docs/data_attributions.md`](docs/data_attributions.md) — third-party data licenses
- [`SECURITY.md`](SECURITY.md) — threat model, before exposing beyond localhost
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to submit changes
- [`paper/paper.md`](paper/paper.md) — JOSS paper draft for academic citation
- [`CHANGELOG.md`](CHANGELOG.md) — release notes
