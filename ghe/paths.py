from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = REPO_ROOT / "images"
RUNS_DIR = REPO_ROOT / "runs"

BEST_POSITION_FILE = DATA_DIR / "bestPosition.txt"
BEST_POSITION_JSON_FILE = DATA_DIR / "bestPosition.json"

SOURCE_ARRAY_DISTRIBUTION_FILE = DATA_DIR / "source_array_distribution.csv"
SOURCE_ARRAY_NPZ_FILE = DATA_DIR / "source_array_distribution.npz"

FREQS_FILE = DATA_DIR / "freqs.npy"
MAGNITUDE_FILE = DATA_DIR / "magnitude.npy"
TOTAL_FREQS_FILE = DATA_DIR / "total_freqs.npy"
TOTAL_MAGNITUDE_FILE = DATA_DIR / "total_magnitude.npy"

SNR_YEAR_TABLE_FILE = DATA_DIR / "snr_year_table.csv"

YEAR_SECONDS = 365 * 24 * 3600


def ensure_project_dirs() -> None:
    """Create standard output directories if they do not exist."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_run_dir(name: str | None = None, root: Path = RUNS_DIR) -> Path:
    """
    Create and return a reproducible run directory.

    If ``name`` is omitted, a local timestamp is used.
    """

    run_name = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / run_name
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir
