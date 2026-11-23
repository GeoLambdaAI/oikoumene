"""
Command-line guard of the data generators (``oikoumene.datagen.cli_guard``).

The committed ``data/`` files are the pinned reference, so ``--help`` must only
print usage and a run against existing outputs must refuse without ``--force``.
No test here runs a real generation: each generator's ``main`` is replaced by a
recorder, and every run points ``--output-dir`` at a temporary directory.
"""
import ast
import inspect
from pathlib import Path
from types import ModuleType

import pytest

from oikoumene.datagen import generate_earth_data, generate_landmask, generate_present_day_data
from oikoumene.paths import DATA_DIR

GENERATORS: list[ModuleType] = [generate_earth_data, generate_landmask, generate_present_day_data]
IDS: list[str] = [m.__name__.rsplit(".", 1)[1] for m in GENERATORS]


def _record_main(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> list[Path]:
    calls: list[Path] = []
    monkeypatch.setattr(module, "main", lambda output_dir: calls.append(output_dir))
    return calls


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


@pytest.mark.parametrize("module", GENERATORS, ids=IDS)
def test_help_exits_zero_and_writes_nothing(module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls = _record_main(monkeypatch, module)
    with pytest.raises(SystemExit) as exc:
        module.cli(["--help", "--output-dir", str(tmp_path)])
    assert exc.value.code == 0
    assert "--output-dir" in capsys.readouterr().out
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", GENERATORS, ids=IDS)
def test_existing_outputs_refused_without_force(module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    calls = _record_main(monkeypatch, module)
    existing = tmp_path / module.OUTPUT_FILES[0]
    existing.write_bytes(b"reference")
    before = _snapshot(tmp_path)
    assert module.cli(["--output-dir", str(tmp_path)]) != 0
    assert calls == []
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("module", GENERATORS, ids=IDS)
def test_force_and_empty_dir_run_generation_into_output_dir(module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                            tmp_path: Path) -> None:
    calls = _record_main(monkeypatch, module)
    assert module.cli(["--output-dir", str(tmp_path / "fresh")]) == 0
    (tmp_path / module.OUTPUT_FILES[0]).write_bytes(b"old")
    assert module.cli(["--output-dir", str(tmp_path), "--force"]) == 0
    assert calls == [tmp_path / "fresh", tmp_path]


@pytest.mark.parametrize("module", GENERATORS, ids=IDS)
def test_default_output_dir_is_data_dir(module: ModuleType) -> None:
    assert inspect.signature(module.main).parameters["output_dir"].default == DATA_DIR


@pytest.mark.parametrize("module", GENERATORS, ids=IDS)
def test_script_entry_goes_through_cli(module: ModuleType) -> None:
    """``python -m ...`` must call ``cli()``, never ``main()`` directly."""
    tree = ast.parse(Path(module.__file__).read_text())
    guards = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert len(guards) == 1
    called = {n.func.id for n in ast.walk(guards[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "cli" in called and "main" not in called


def test_earth_output_files_match_pinned_reference() -> None:
    from test_data_integrity import EXPECTED_SHA256

    assert set(generate_earth_data.OUTPUT_FILES) == set(EXPECTED_SHA256)
