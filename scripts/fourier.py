import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import tukey
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_IMAGE_DIR = Path(__file__).resolve().parent.parent / "images"
_FREQS_FILE = _DATA_DIR / "freqs.npy"
_MAGNITUDE_FILE = _DATA_DIR / "magnitude.npy"

INT_TIME = 0.01
NUM = int(120000 * INT_TIME)

t = np.linspace(0, INT_TIME, NUM, endpoint=False)
# demo = np.sin(2 * np.pi * 600 * t)

from scripts.metricCalculate import calculate_metric_response

h_values = [calculate_metric_response(ti) for ti in t]
h_values = np.array(h_values)

plt.figure(figsize=(10, 6))

plt.plot(t, h_values)
plt.xlabel('Time [s]')
plt.ylabel('Signal [1]')
plt.title('Input Signal Curve')
plt.savefig(_IMAGE_DIR / "Signal.png")

def fourier(signal, sampling_rate = NUM / INT_TIME):
    """
    doing fft to the signal with normalization.
    """
    N = len(signal)
    dt = 1.0 / sampling_rate

    # window = tukey(N, alpha=0.1) 
    # signal_windowed = signal * window
    
    fft_complex = rfft(signal)
    fft_phys = fft_complex * dt
    
    freqs = rfftfreq(N, d=dt)
    
    mask = freqs > 0
    positive_freqs = freqs[mask]

    positive_magnitude = np.abs(fft_phys[mask])

    return signal, positive_magnitude, positive_freqs

def plot(inputSignal, fft_magnitude, freqs):
    _, axes = plt.subplots(1, 2, figsize=(12, 8))

    axes[0].plot(t, inputSignal)
    axes[0].set_title('Original Signal')
    axes[0].set_xlabel('Time [s]')

    axes[1].plot(freqs, fft_magnitude)
    axes[1].set_title('FFT Magnitude (Positive Frequencies)')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_xlim(1,1000)

    plt.tight_layout()
    plt.savefig(_IMAGE_DIR / "Fouriered Signal.png")

def main():
    inputSignal, fft_magnitude, freqs = fourier(h_values)
    np.save(_FREQS_FILE, freqs)
    np.save(_MAGNITUDE_FILE, fft_magnitude)
    plot(inputSignal, fft_magnitude, freqs)

if __name__ == "__main__":
    main()