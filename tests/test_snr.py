from __future__ import annotations

import numpy as np

from ghe.config import NoiseConfig, SamplingConfig
from ghe.snr import calculate_snr_from_arrays
from ghe.spectrum import fourier


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


def test_snr_duration_comes_from_frequency_spacing_not_sampling_config() -> None:
    freqs = np.array([1.0, 2.0, 3.0])
    magnitude = np.array([2.0, 2.0, 0.0])
    noise = lambda f: np.ones_like(f) * 4.0

    inferred = calculate_snr_from_arrays(
        magnitude,
        freqs,
        noise_config=NoiseConfig(min_frequency_hz=1.0, max_frequency_hz=2.0),
        noise_psd_func=noise,
    )
    mismatched_config = calculate_snr_from_arrays(
        magnitude,
        freqs,
        noise_config=NoiseConfig(min_frequency_hz=1.0, max_frequency_hz=2.0),
        sampling_config=SamplingConfig(duration_s=100.0, sample_rate_hz=4.0),
        noise_psd_func=noise,
    )

    assert np.isclose(mismatched_config, inferred)


def test_year_snr_is_independent_of_fft_duration_for_same_periodic_signal() -> None:
    sample_rate = 1000.0
    signal_frequency = 50.0
    noise = lambda f: np.ones_like(f) * 4.0
    noise_config = NoiseConfig(min_frequency_hz=1.0, max_frequency_hz=100.0)

    def spectrum_for_duration(duration_s: float) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(int(sample_rate * duration_s)) / sample_rate
        signal = np.sin(2.0 * np.pi * signal_frequency * t)
        _, magnitude, freqs = fourier(signal, sampling_rate=sample_rate)
        return magnitude, freqs

    short_magnitude, short_freqs = spectrum_for_duration(1.0)
    long_magnitude, long_freqs = spectrum_for_duration(100.0)

    short_snr = calculate_snr_from_arrays(
        short_magnitude,
        short_freqs,
        noise_config=noise_config,
        noise_psd_func=noise,
    )
    long_snr = calculate_snr_from_arrays(
        long_magnitude,
        long_freqs,
        noise_config=noise_config,
        noise_psd_func=noise,
    )

    assert np.isclose(long_snr, short_snr, rtol=1e-10)
