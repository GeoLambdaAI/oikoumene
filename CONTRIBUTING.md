# Contributing to Oikoumene

Thank you for your interest in contributing. This is a scientific simulation
project where every equation cites a published source — contributions are
welcome but held to a research-grade standard.

## Code of Conduct

Be respectful and constructive. Critique ideas and code, not people.
Disagreements about scientific calibration are welcome and expected;
disrespect is not.

## Development Setup

```bash
# Clone and enter
git clone https://github.com/GeoLambdaAI/oikoumene.git
cd oikoumene

# Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Runtime dependencies
pip install -r requirements.txt
pip install -e .

# Dev tools (optional but recommended)
pip install -e ".[dev]"

# Earth data ships in data/ — do not regenerate it for normal development.
# The generators refuse to overwrite existing files; --force overwrites the
# reference, changes simulation results and fails tests/test_data_integrity.py
# (see README, "Generate Earth Data"). To compare, write elsewhere:
#   python -m oikoumene.datagen.generate_earth_data --output-dir /tmp/earth

# Run the simulation
export FLASK_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python app.py
# Open http://127.0.0.1:5000
```

## Running Tests

```bash
pytest                      # all tests
pytest tests/test_macro.py -v  # one file, verbose
pytest --cov=. --cov-report=term-missing
```

All tests must pass before submitting a PR; CI will run them automatically
across Python 3.11, 3.12, and 3.13.

## Code Style

- Python 3.11+ required.
- 4-space indentation, no tabs.
- Type hints encouraged for new code; not required for existing modules.
- Module-level docstrings required for new modules.
- **Inline citations required for new equations** — comment with author, year,
  and journal/source.

Example:

```python
# Climate sensitivity per IPCC AR6 WG1 Table 7.SM.1
ECS = 3.0  # °C per doubling of CO2
```

## Scientific Contributions

If you change a calibrated parameter or equation:

1. Cite the source paper inline.
2. Update the relevant table in `README.md` if it is a top-level parameter.
3. If the change affects the IPCC validation suite (`test_macro.py`), justify
   any new tolerance bands in your PR description.
4. If the change alters emergent macro behaviour (e.g., 2100 temperature),
   include a before/after run with seed `42` so reviewers can reproduce.

## Pull Request Process

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Make your changes; ensure `pytest` passes.
4. Update `CHANGELOG.md` under `## [Unreleased]`.
5. Open a PR using the template; link relevant issues.
6. Be ready to defend calibration changes with citations.

## Reporting Bugs

Use the **Bug report** issue template. Include:

- Scenario and seed (for reproducibility).
- The relevant `logs/{run_id}/metadata.json` if available.
- Python version and OS.

## Validation Questions

If you suspect an equation is miscalibrated rather than buggy, use the
**Validation question** issue template. Cite the paper you believe should
govern the behaviour, and indicate the discrepancy you observed.

## License

By contributing, you agree that your contributions will be licensed under the
project's [AGPL-3.0-or-later](LICENSE) licence — this means derived works
distributed over a network must also be open-sourced under AGPL.
