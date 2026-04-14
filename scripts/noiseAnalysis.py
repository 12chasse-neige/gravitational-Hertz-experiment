import numpy as np
import matplotlib.pyplot as plt
import sys
import gwinc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fourier import main, INT_TIME
from scripts.quantumNoise import squeeze_quantum_noise_with_varying_angle

YEAR = 365 * 24 * 3600

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FREQS_FILE = _DATA_DIR / "freqs.npy"
_MAGNITUDE_FILE = _DATA_DIR / "magnitude.npy"

def calculate_snr(magnitude_path: Path = _MAGNITUDE_FILE, freq_path: Path = _FREQS_FILE) -> float:
    """    
    This function calculates the SNR of the signal.
    The SNR is calculated by the following formula:
    SNR = sqrt(4 * signal_magnitude**2 / total_noise_psd)
    where signal_magnitude is the magnitude of the signal and total_noise_psd is the total noise power spectral density.
    """
    # main()

    signal_magnitude = np.load(magnitude_path)
    freq = np.load(freq_path)

    valid_mask = (freq >= 1.0) & (freq <= 5000.0)
    freq_valid = freq[valid_mask]
    signal_magnitude_valid = signal_magnitude[valid_mask]

    total_noise_psd = squeeze_quantum_noise_with_varying_angle(freq_valid)

    if freq_valid.size < 2:
        raise ValueError(
            "Frequency array has too few points in the 1-5000 Hz band. "
            "Check the generated data and ensure _FREQS_FILE contains the correct frequencies."
        )

    # if using the noise from gwinc
    # budget = gwinc.load_budget('aLIGO')
    # trace = budget.run(freq=freq_valid)
    # total_noise_psd = trace.psd 
    
    integrand = (4 * signal_magnitude_valid**2) / total_noise_psd
    
    df = freq_valid[1] - freq_valid[0]
    
    snr = np.sqrt(np.sum(integrand) * df)
    snr_year = snr * np.sqrt(YEAR / INT_TIME)
    
    print(f"Calculated SNR (1 year) = {snr_year:.4e}")
    
    return snr_year

# def plot(freq, trace):
#     plt.figure(figsize=(10, 6))
    
#     plot_mask = (freq >= 200) & (freq <= 1000)
#     freq_plot = freq[plot_mask]
#     budget = gwinc.load_budget('aLIGO')
#     trace = budget.run(freq=freq_plot)

#     plt.plot(freq_plot, trace, label='Total Noise', color='black', linewidth=2)

#     for name, sub_trace in trace.items[plot_mask]():    
#         if hasattr(sub_trace, 'asd'):
#             plt.plot(freq, sub_trace.asd, label=name, linestyle='--', alpha=0.7)

#     plt.grid(True, which='both', linestyle='--', alpha=0.4)
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Strain Noise [1/sqrt(Hz)]')
#     plt.title('Advanced LIGO Noise Budget')
#     plt.legend(loc='upper right', fontsize='small')
#     plt.xlim(200, 1000)
#     plt.savefig("./Figure/Noise.png")

if __name__ == "__main__":
    freq = calculate_snr()
