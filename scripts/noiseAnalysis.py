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
from ghe.spectrum import Spectrum, calculate_spectrum, save_spectrum_arrays
from scr.fourier import build_default_signal, plot


def generate_current_spectrum(
    *,
    sampling_config: SamplingConfig | None = None,
    magnitude_path=MAGNITUDE_FILE,
    freq_path=FREQS_FILE,
    save: bool = True,
    make_plot: bool = True,
) -> Spectrum:
    """Regenerate the signal and its FFT from the current project parameters."""

    active_sampling = sampling_config or SamplingConfig()
    time_axis = active_sampling.time_axis()
    signal = build_default_signal(active_sampling)
    spectrum = calculate_spectrum(signal, sampling=active_sampling)

    if save:
        save_spectrum_arrays(
            spectrum,
            magnitude_path=magnitude_path,
            freq_path=freq_path,
        )
    if make_plot:
        plot(
            spectrum.signal,
            spectrum.magnitude,
            spectrum.freqs,
            time_axis=time_axis,
        )

    return spectrum


def calculate_snr(
    magnitude_path=None,
    freq_path=None,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    verbose: bool = True,
) -> float:
    """
    Calculate SNR from a freshly generated spectrum by default.

    Passing both ``magnitude_path`` and ``freq_path`` preserves the legacy
    behavior of integrating an already saved spectrum.
    """

    if (magnitude_path is None) != (freq_path is None):
        raise ValueError("magnitude_path and freq_path must be provided together.")

    if magnitude_path is None:
        if verbose:
            print("Regenerating the signal spectrum from current parameters ...")
        spectrum = generate_current_spectrum(sampling_config=sampling_config)
        snr_year = _calculate_snr_from_arrays(
            spectrum.magnitude,
            spectrum.freqs,
            noise_config=noise_config,
            detector_config=detector_config,
            sampling_config=sampling_config,
        )
    else:
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
