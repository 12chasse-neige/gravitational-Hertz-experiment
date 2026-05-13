from __future__ import annotations

import numpy as np

from ghe.config import NoiseConfig, SamplingConfig
from ghe.snr import calculate_snr_from_arrays


def test_snr_with_synthetic_constant_noise() -> None:
    freqs = np.array([0.0, 1.0, 2.0, 3.0])
    magnitude = np.array([0.0, 2.0, 2.0, 0.0])
    noise = lambda f: np.ones_like(f) * 4.0

    snr = calculate_snr_from_arrays(
        magnitude,
        freqs,
        noise_config=NoiseConfig(min_frequency_hz=1.0, max_frequency_hz=2.0),
        sampling_config=SamplingConfig(duration_s=1.0, sample_rate_hz=4.0),
        noise_psd_func=noise,
    )

    expected = np.sqrt(8.0) * np.sqrt(365 * 24 * 3600)
    assert np.isclose(snr, expected)
