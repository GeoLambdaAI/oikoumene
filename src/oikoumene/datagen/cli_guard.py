"""Shared command line for the data generators: ``--output-dir``, ``--force`` and the overwrite guard.

The committed files in ``data/`` are the pinned reference for all published
results (see ``tests/test_data_integrity.py``). A generator therefore refuses to
overwrite an existing output file unless ``--force`` is given, and ``--help``
only prints usage.
"""
import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from oikoumene.paths import DATA_DIR

log = logging.getLogger(__name__)

_EPILOG = ("The committed data/ files are the reference for all published results; "
           "regenerating them can change simulation results.")


def build_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the two options every generator shares."""
    parser = argparse.ArgumentParser(description=description, epilog=_EPILOG)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR,
                        help=f"directory for the output files (default: {DATA_DIR})")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    return parser


def existing_outputs(output_dir: Path, names: Sequence[str]) -> list[Path]:
    """Output files that already exist in ``output_dir``."""
    return [output_dir / name for name in names if (output_dir / name).exists()]


def run_guarded(description: str, output_names: Sequence[str],
                generate: Callable[[Path], object], argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, refuse to overwrite existing outputs without ``--force``, then call ``generate``.

    Returns the process exit code: 0 after generating, 1 when it refused and wrote nothing.
    """
    args = build_parser(description).parse_args(argv)
    existing = existing_outputs(args.output_dir, output_names)
    if existing and not args.force:
        log.error("Refusing to overwrite %d existing file(s); nothing was written. Use --force to overwrite:\n%s",
                  len(existing), "\n".join(f"  {path}" for path in existing))
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate(args.output_dir)
    return 0
