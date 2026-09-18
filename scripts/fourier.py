from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import (
    FREQS_FILE,
    INT_TIME,
    MAGNITUDE_FILE,
    NUM,
    PAPER_FIGURES_DIR,
    SamplingConfig,
    build_time_axis,
)
from ghe.spectrum import fourier


def build_default_signal(
    sampling_config: SamplingConfig | None = None,
    *,
    source_config=None,
) -> np.ndarray:
    """Generate the single-source signal using the active project parameters."""

    from ghe.config import SourceConfig
    from ghe.metric import calculate_response_phasor
    from ghe.optimization import solve_best_geometry
    from ghe.signal import synthesize_signal

    cfg = source_config or SourceConfig()
    time_axis = (sampling_config or SamplingConfig()).time_axis()
    H = calculate_response_phasor(*solve_best_geometry(config=cfg).angles, config=cfg)
    return synthesize_signal(H, time_axis, cfg)


def plot(inputSignal, fft_magnitude, freqs, time_axis=None):
    import matplotlib.pyplot as plt

    if time_axis is None:
        time_axis = build_time_axis()

    _, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].plot(time_axis, inputSignal)
    axes[0].set_title("Original Signal")
    axes[0].set_xlabel("Time [s]")

    axes[1].plot(freqs, fft_magnitude)
    axes[1].set_title("FFT Magnitude (Positive Frequencies)")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_xlim(1, 1000)

    plt.tight_layout()
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PAPER_FIGURES_DIR / "Fouriered Signal.png")


def main():
    time_axis = build_time_axis()
    h_values = build_default_signal()
    inputSignal, fft_magnitude, freqs = fourier(h_values, sampling_rate=NUM / INT_TIME)
    from ghe.spectrum import Spectrum, save_spectrum_arrays

    save_spectrum_arrays(
        Spectrum(inputSignal, fft_magnitude, freqs), MAGNITUDE_FILE, FREQS_FILE
    )
    plot(inputSignal, fft_magnitude, freqs, time_axis=time_axis)


if __name__ == "__main__":
    main()
