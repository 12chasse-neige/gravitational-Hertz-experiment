import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from metricCalculate import ExperimentConfig
import gwinc

@dataclass
class DetectorConfig:
    """
    Dataclass to store all detector parameters.
    """
    testmass: float = 39.6                  # mass of the end testmass (kg)
    length: float = ExperimentConfig.L      # length of the arm (meter)
    hbar: float = 1.05457e-34               # reduced Plank constant (J⋅s)
    wavelength: float = 1064e-9             # laser wavelength (m)
    c: float = ExperimentConfig.c           # light speed
    power: float = 125                      # power of the input laser (W)
    T_PRM: float = 0.03                     # transmittance of the power recycling mass
    T_ITM: float = 0.014                    # transmittance of the input test mass
    T_ETM: float = 5e-6                     # transmittance of the end test mass
    loss_mirror_ppm: float = 40             # loss of one reflection (ppm)
    loss_BS_ppm: float = 500                # loss of the BS system (ppm)
    

def get_laser_power_in_cavity(inputPower: float, config = DetectorConfig())-> float:
    """
    Calculate the circling power in the cavity from the given input laser power.
    """
    l_mirror = config.loss_mirror_ppm * 1e-6
    l_BS     = config.loss_BS_ppm * 1e-6

    l_roundtrip_arm = 2 * l_mirror

    total_arm_loss = config.T_ETM + l_roundtrip_arm
    R_arm = 1 - (4 * total_arm_loss / config.T_ITM)
    
    r_arm = np.sqrt(R_arm)
    r_comp = r_arm * (1 - l_BS) * (1 - l_BS)
    r_PRM = np.sqrt(1 - config.T_PRM)    

    t_PRM = np.sqrt(config.T_PRM)
    PRG = (t_PRM / (1 - r_PRM * r_comp))**2

    P_BS = inputPower * PRG   
    P_arm_input = P_BS / 2
    
    ACG = 4 / config.T_ITM
    P_circulating = P_arm_input * ACG
    
    return P_circulating

def get_standard_quantum_limit(gravitationalWaveOmega: float, config = DetectorConfig()) -> float:
    """
    Calculate standard quantum limit for the square root of the 
    single-sided spectral density.
    """
    h_SQL = np.sqrt(8 * config.hbar / (config.testmass * gravitationalWaveOmega**2 * config.length**2))
    return h_SQL

def get_coupling_constant(gravitationalWaveOmega: float, config = DetectorConfig()) -> float:
    """
    Calculate the coupling constant for the given frequency.
    """
    omega = 2 * np.pi * config.c / config.wavelength
    gamma = config.T_ITM * config.c / (4 * config.length)  # cavity's half band width

    P_0 = 2 * get_laser_power_in_cavity(config.power)

    # calculate the coupling constant for the given frequency
    kappa = (8 * gamma * omega * P_0) / (config.testmass * config.length * config.c * gravitationalWaveOmega**2 * (gamma**2 + gravitationalWaveOmega**2))
    return kappa

def get_quantum_noise_psd(freq: float, config = DetectorConfig()) -> float:
    """
    Returns the total quantum noise (shot and radiation) power spetral density.
    """
    gravitationalWaveOmega = 2 * np.pi * freq
    kappa = get_coupling_constant(gravitationalWaveOmega)

    S_SQL = (get_standard_quantum_limit(gravitationalWaveOmega)**2 / 2) * (kappa + 1 / kappa)
    return S_SQL

def plot():
    freq = np.linspace(100, 1000, 10000)
    plt.figure(figsize = (10, 6))
    noise_psd = get_quantum_noise_psd(freq)
    noise_asd = np.sqrt(noise_psd)
    plt.plot(freq, noise_asd, label = 'Quantum Noise', color = 'blue', linewidth = 2)

    buget = gwinc.load_budget('aLIGO')
    trace = buget.run(freq = freq)
    aLIGO_noise = trace['Quantum']
    plt.plot(freq, aLIGO_noise.asd, label='aLIGO Quantum Noise', color='red', linewidth=2)

    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Quantum Noise [1/sqrt(Hz)]')
    plt.legend(loc='upper right', fontsize='small')
    plt.title('Quantum Noise (ASD)')
    plt.savefig("./Figure/Quantum Noise.png")
    
plot()