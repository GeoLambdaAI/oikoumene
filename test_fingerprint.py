"""Behaviour fingerprint: a fresh run must reproduce tests/fixtures/fingerprint.json bit for bit.

Safety net for pure refactors (see scripts/fingerprint.py). The hashes are
bit-exact, so the test skips on a platform or library version other than the
one recorded in the fixture (e.g. CI on Linux); there it cannot tell a refactor
from a floating-point difference.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from fingerprint import FIXTURE, environment, fingerprint, first_mismatches  # noqa: E402


def test_fingerprint_matches_fixture() -> None:
    reference = json.loads(FIXTURE.read_text())
    if reference["environment"] != environment():
        pytest.skip(f"fixture recorded on {reference['environment']}, running on {environment()}")
    fresh = fingerprint()
    assert fresh["runs"].keys() == reference["runs"].keys()
    assert first_mismatches(reference, fresh) == {}, "first differing tick per run"
