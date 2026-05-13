from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import FREQS_FILE, IMAGES_DIR, INT_TIME, MAGNITUDE_FILE, NUM, build_time_axis
from ghe.spectrum import fourier


def build_default_signal() -> np.ndarray:
    from scripts.metricCalculate import calculate_metric_response

    time_axis = build_time_axis()
    return np.array([calculate_metric_response(ti) for ti in time_axis], dtype=float)


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
    plt.savefig(IMAGES_DIR / "Fouriered Signal.png")


def main():
    time_axis = build_time_axis()
    h_values = build_default_signal()
    inputSignal, fft_magnitude, freqs = fourier(h_values, sampling_rate=NUM / INT_TIME)
    np.save(FREQS_FILE, freqs)
    np.save(MAGNITUDE_FILE, fft_magnitude)
    plot(inputSignal, fft_magnitude, freqs, time_axis=time_axis)


if __name__ == "__main__":
    main()
