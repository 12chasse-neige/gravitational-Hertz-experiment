"""
Signal-to-noise-ratio integration.

Inputs are FFT magnitudes and their corresponding positive frequencies.  The
integration follows the original project convention:

    SNR = sqrt(sum(4 * |h(f)|^2 / S_h(f)) * df)

The result is then scaled from the sampled integration time to one year by
``sqrt(YEAR_SECONDS / duration_s)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .config import DetectorConfig, NoiseConfig, SamplingConfig
from .noise import get_noise_psd
from .paths import FREQS_FILE, MAGNITUDE_FILE, YEAR_SECONDS


def calculate_snr_from_arrays(
    signal_magnitude: np.ndarray,
    freq: np.ndarray,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    noise_psd_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """
    Calculate 1-year SNR from in-memory spectrum arrays.

    ``noise_psd_func`` is injectable for tests and model comparisons.  When it is
    omitted, the package uses the model selected by ``NoiseConfig.model``.
    """

    active_noise = noise_config or NoiseConfig()
    active_detector = detector_config or DetectorConfig()
    active_sampling = sampling_config or SamplingConfig()

    signal_magnitude = np.asarray(signal_magnitude, dtype=float)
    freq = np.asarray(freq, dtype=float)
    valid_mask = (freq >= active_noise.min_frequency_hz) & (freq <= active_noise.max_frequency_hz)
    freq_valid = freq[valid_mask]
    signal_magnitude_valid = signal_magnitude[valid_mask]

    if freq_valid.size < 2:
        raise ValueError(
            "Frequency array has too few points in the configured SNR band. "
            "Check generated spectrum data."
        )

    if noise_psd_func is None:
        total_noise_psd = get_noise_psd(
            freq_valid,
            noise_config=active_noise,
            detector_config=active_detector,
        )
    else:
        total_noise_psd = noise_psd_func(freq_valid)

    # Frequencies are evenly spaced because they come from rfftfreq.  The
    # integral is therefore a rectangular sum over the selected frequency band.
    integrand = (4.0 * signal_magnitude_valid**2) / total_noise_psd
    df = freq_valid[1] - freq_valid[0]
    snr = np.sqrt(np.sum(integrand) * df)
    return float(snr * np.sqrt(YEAR_SECONDS / active_sampling.duration_s))


def calculate_snr(
    magnitude_path: str | Path = MAGNITUDE_FILE,
    freq_path: str | Path = FREQS_FILE,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
    sampling_config: SamplingConfig | None = None,
) -> float:
    """Load legacy spectrum arrays from disk and calculate 1-year SNR."""

    signal_magnitude = np.load(magnitude_path)
    freq = np.load(freq_path)
    return calculate_snr_from_arrays(
        signal_magnitude,
        freq,
        noise_config=noise_config,
        detector_config=detector_config,
        sampling_config=sampling_config,
    )


def save_snr_json(snr_year: float, output_path: str | Path) -> None:
    """Persist a small machine-readable SNR summary for run directories."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"snr_year": snr_year}, indent=2),
        encoding="utf-8",
    )
