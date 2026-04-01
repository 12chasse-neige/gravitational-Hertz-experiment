import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import tukey

int_time = 0.01
num = int(120000 * int_time)

t = np.linspace(0, int_time, num, endpoint=False)
# demo = np.sin(2 * np.pi * 600 * t)

from metricCalculate import calculate_metric_response

h_values = [calculate_metric_response(ti) for ti in t]
h_values = np.array(h_values)

plt.figure(figsize=(10, 6))

plt.plot(t, h_values)
plt.xlabel('Time [s]')
plt.ylabel('Signal [1]')
plt.title('Input Signal Curve')
plt.savefig("./Figure/Signal.png")

def fourier(signal, sampling_rate = num / int_time):
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
    plt.savefig("./Figure/fourierSignal.png")

def main():
    inputSignal,fft_magnitude,freqs = fourier(h_values)
    np.save("Data/magnitude.npy", fft_magnitude)
    np.save("Data/freqs.npy", freqs)
    plot(inputSignal, fft_magnitude, freqs)

if __name__ == "__main__":
    main()