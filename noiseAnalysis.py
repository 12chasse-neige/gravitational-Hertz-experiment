import gwinc
import numpy as np
import matplotlib.pyplot as plt
from fourier import main

def calculate_snr() -> float:
    budget = gwinc.load_budget('aLIGO')
    
    # main()
    signal_magnitude = np.load("Data/magnitude.npy")
    freq = np.load("Data/freqs.npy")

    valid_mask = (freq >= 1.0) & (freq <= 5000.0)
    freq_valid = freq[valid_mask]
    signal_magnitude_valid = signal_magnitude[valid_mask]

    trace = budget.run(freq=freq_valid)
    total_noise_psd = trace.psd 
    
    integrand = (4 * signal_magnitude_valid**2) / total_noise_psd
    
    df = freq_valid[1] - freq_valid[0]
    
    snr = np.sqrt(np.sum(integrand) * df)
    
    print(f"Calculated SNR (1 second) = {snr:.4e}")
    
    return freq_valid, trace

def plot(freq, trace):
    plt.figure(figsize=(10, 6))
    
    plot_mask = (freq >= 200) & (freq <= 1000)
    freq_plot = freq[plot_mask]
    trace_plot = trace[plot_mask]

    plt.plot(freq_plot, trace_plot.asd, label='Total Noise', color='black', linewidth=2)

    for name, sub_trace in trace.items():    
        if hasattr(sub_trace, 'asd'):
            plt.plot(freq, sub_trace.asd, label=name, linestyle='--', alpha=0.7)

    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Strain Noise [1/sqrt(Hz)]')
    plt.title('Advanced LIGO Noise Budget')
    plt.legend(loc='upper right', fontsize='small')
    plt.xlim(200, 1000)
    plt.savefig("./Figure/Noise.png")

freq, trace = calculate_snr()
plot(freq, trace)