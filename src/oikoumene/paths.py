"""Repository locations: the one home of the data, logs and static directories."""
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
LOGS_DIR: Path = REPO_ROOT / "logs"
STATIC_DIR: Path = REPO_ROOT / "static"
