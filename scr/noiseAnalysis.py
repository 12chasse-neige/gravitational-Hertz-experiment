from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghe.config import FREQS_FILE, MAGNITUDE_FILE
from ghe.snr import calculate_snr as _calculate_snr


def calculate_snr(magnitude_path=MAGNITUDE_FILE, freq_path=FREQS_FILE) -> float:
    snr_year = _calculate_snr(magnitude_path=magnitude_path, freq_path=freq_path)
    print(f"Calculated SNR (1 year) = {snr_year:.4e}")
    return snr_year


if __name__ == "__main__":
    calculate_snr()
