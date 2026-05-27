from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import (
    DetectorConfig,
    FREQS_FILE,
    MAGNITUDE_FILE,
    NoiseConfig,
    SamplingConfig,
)
from ghe.snr import calculate_snr as _calculate_snr
from ghe.snr import calculate_snr_from_arrays as _calculate_snr_from_arrays


def calculate_snr(
    magnitude_path=MAGNITUDE_FILE,
    freq_path=FREQS_FILE,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    verbose: bool = True,
) -> float:
    snr_year = _calculate_snr(
        magnitude_path=magnitude_path,
        freq_path=freq_path,
        noise_config=noise_config,
        detector_config=detector_config,
        sampling_config=sampling_config,
    )
    if verbose:
        print(f"Calculated SNR (1 year) = {snr_year:.4e}")
    return snr_year


def calculate_snr_from_arrays(
    signal_magnitude: np.ndarray,
    freq: np.ndarray,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    verbose: bool = True,
) -> float:
    snr_year = _calculate_snr_from_arrays(
        signal_magnitude,
        freq,
        noise_config=noise_config,
        detector_config=detector_config,
        sampling_config=sampling_config,
    )
    if verbose:
        print(f"Calculated SNR (1 year) = {snr_year:.4e}")
    return snr_year


if __name__ == "__main__":
    calculate_snr()
