"""
Frequency-domain conversion for generated strain signals.

The project keeps the original FFT normalization: the real FFT coefficients are
multiplied by ``dt`` before taking magnitudes.  Downstream SNR calculations
expect this convention, so changes here should be treated as physics-visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import rfft, rfftfreq

from .config import SamplingConfig


@dataclass(frozen=True)
class Spectrum:
    """Container for a time-domain signal and its positive-frequency spectrum."""

    signal: np.ndarray
    magnitude: np.ndarray
    freqs: np.ndarray


def fourier(signal: np.ndarray, sampling_rate: float | None = None, sampling: SamplingConfig | None = None):
    """
    Compute the positive-frequency real FFT.

    Returns ``(signal, magnitude, freqs)`` to preserve the legacy script API.
    The zero-frequency bin is omitted because SNR integration only uses
    oscillatory signal power.
    """

    if sampling_rate is None:
        active_sampling = sampling or SamplingConfig()
        sampling_rate = active_sampling.num_samples / active_sampling.duration_s

    signal = np.asarray(signal, dtype=float)
    n_samples = len(signal)
    dt = 1.0 / sampling_rate
    fft_complex = rfft(signal)
    fft_phys = fft_complex * dt
    freqs = rfftfreq(n_samples, d=dt)
    mask = freqs > 0
    return signal, np.abs(fft_phys[mask]), freqs[mask]


def calculate_spectrum(
    signal: np.ndarray,
    sampling_rate: float | None = None,
    sampling: SamplingConfig | None = None,
) -> Spectrum:
    """Return a structured ``Spectrum`` object from a time-domain signal."""

    input_signal, magnitude, freqs = fourier(signal, sampling_rate=sampling_rate, sampling=sampling)
    return Spectrum(signal=input_signal, magnitude=magnitude, freqs=freqs)


def save_spectrum_arrays(
    spectrum: Spectrum,
    magnitude_path: str | Path,
    freq_path: str | Path,
) -> None:
    """Save legacy ``.npy`` arrays consumed by existing scripts."""

    magnitude_output = Path(magnitude_path)
    freq_output = Path(freq_path)
    magnitude_output.parent.mkdir(parents=True, exist_ok=True)
    freq_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(magnitude_output, spectrum.magnitude)
    np.save(freq_output, spectrum.freqs)


def save_spectrum_npz(spectrum: Spectrum, output_path: str | Path) -> None:
    """Save signal, magnitude, and frequency arrays in one reproducible artifact."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        signal=spectrum.signal,
        magnitude=spectrum.magnitude,
        freqs=spectrum.freqs,
    )
