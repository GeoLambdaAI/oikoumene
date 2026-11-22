"""Thin start wrapper: keeps ``python app.py`` working. The app lives in ``oikoumene.web.app``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from oikoumene.web.app import main  # noqa: E402

if __name__ == "__main__":
    main()
