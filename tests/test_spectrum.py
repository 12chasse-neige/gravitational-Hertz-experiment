from __future__ import annotations

import numpy as np

from ghe.spectrum import fourier


def test_fourier_peak_for_synthetic_sinusoid() -> None:
    sample_rate = 1000.0
    t = np.arange(1000) / sample_rate
    signal = np.sin(2.0 * np.pi * 50.0 * t)
    _, magnitude, freqs = fourier(signal, sampling_rate=sample_rate)
    assert freqs[np.argmax(magnitude)] == 50.0
