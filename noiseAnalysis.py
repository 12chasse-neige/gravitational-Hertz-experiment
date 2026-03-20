import gwinc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from fourier import main

def calculate_snr():
    budget = gwinc.load_budget('aLIGO')
    
    main()
    signal_magnitude = np.load("magnitude.npy")
    freq = np.load("freqs.npy")

    trace = budget.run(freq=freq)
    total_noise_psd = trace.psd # getting the amplitude spectral density of the noise
    
    integrand = 4 * signal_magnitude**2 / (total_noise_psd)
    total = simpson(signal_magnitude,freq)
    print(total)
    snr = np.sqrt(simpson(integrand, freq))
    print(snr)
    return freq, trace

def plot(freq, trace):
    plt.figure(figsize=(10, 6))

    plt.plot(freq, trace.asd, label='Total Noise', color='black', linewidth=2)

    for name, sub_trace in trace.items():    
        if hasattr(sub_trace, 'asd'):
            plt.plot(freq, sub_trace.asd, label=name, linestyle='--', alpha=0.7)

    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Strain Noise [1/sqrt(Hz)]')
    plt.title('Advanced LIGO Noise Budget')
    plt.legend(loc='upper right', fontsize='small')
    plt.xlim(1, 1000)
    plt.savefig("./Noise.png")

calculate_snr()