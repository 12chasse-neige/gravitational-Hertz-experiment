import numpy as np
from dataclasses import dataclass, field
import os
import matplotlib.pyplot as plt
from metricCalculate import ExperimentConfig
import gwinc
from gwinc import Struct

@dataclass
class DetectorConfig:
    """
    Dataclass to store all detector parameters.
    """
    testmass: float = field(default_factory=lambda: float(os.getenv("LIGO_TEST_MASS", "39.6")))  # mass of the end testmass (kg)
    length: float = field(default_factory=lambda: float(os.getenv("LIGO_ARM_LENGTH", "4000")))   # length of the arm (meter)
    hbar: float = 1.05457e-34               # reduced Plank constant (J⋅s)
    wavelength: float = 1064e-9             # laser wavelength (m)
    c: float = ExperimentConfig.c           # light speed
    power: float = 125                      # power of the input laser (W)
    T_PRM: float = 0.03                     # transmittance of the power recycling mass
    T_ITM: float = 0.014                    # transmittance of the input test mass
    T_ETM: float = 5e-6                     # transmittance of the end test mass
    loss_mirror_ppm: float = 40             # loss of one reflection (ppm)
    loss_BS_ppm: float = 500                # loss of the BS system (ppm)
    

def get_laser_power_in_cavity(inputPower: float, config: DetectorConfig | None = None) -> float:
    if config is None:
        config = DetectorConfig()
    """
    Calculate the circling power in the cavity from the given input laser power.
    """
    # from ppm to transmittance
    l_mirror = config.loss_mirror_ppm * 1e-6
    l_BS     = config.loss_BS_ppm * 1e-6

    # loss per round trip
    l_roundtrip_arm = 2 * l_mirror

    # per round trip, the loss is composed by two mirror reflectings and the leaking at the ETM.
    total_arm_loss = config.T_ETM + l_roundtrip_arm

    # the equivalent reflection rate for the cavity and the beam splitter
    R_arm = 1 - (4 * total_arm_loss / config.T_ITM)
    r_arm = np.sqrt(R_arm)
    r_comp = r_arm * (1 - l_BS)
    r_PRM = np.sqrt(1 - config.T_PRM)    

    t_PRM = np.sqrt(config.T_PRM)
    PRG = (t_PRM / (1 - r_PRM * r_comp))**2

    # the power at the beam splitter and at each arm's cavity
    P_BS = inputPower * PRG   
    P_arm_input = P_BS / 2
    
    # magnifying of the arm cavity
    ACG = 4 / config.T_ITM
    P_circulating = P_arm_input * ACG
    
    return P_circulating

def get_standard_quantum_limit(gravitationalWaveOmega: float, config: DetectorConfig | None = None) -> float:
    if config is None:
        config = DetectorConfig()
    """
    Calculate standard quantum limit for the square root of the 
    single-sided spectral density.
    """
    h_SQL = np.sqrt(8 * config.hbar / (config.testmass * gravitationalWaveOmega**2 * config.length**2))
    return h_SQL

def get_coupling_constant(gravitationalWaveOmega: float, config: DetectorConfig | None = None) -> float:
    if config is None:
        config = DetectorConfig()
    """
    Calculate the coupling constant for the given frequency.
    """
    omega = 2 * np.pi * config.c / config.wavelength
    gamma = config.T_ITM * config.c / (4 * config.length)  # cavity's half band width
    # gamma = 2 * np.pi * 600

    P_0 = 2 * get_laser_power_in_cavity(config.power, config=config)

    # calculate the coupling constant for the given frequency
    kappa = (8 * gamma * omega * P_0) / (config.testmass * config.length * config.c * gravitationalWaveOmega**2 * (gamma**2 + gravitationalWaveOmega**2))

    return kappa

def get_quantum_noise_psd(freq: float, config: DetectorConfig | None = None) -> float:
    if config is None:
        config = DetectorConfig()
    """
    Returns the total quantum noise (shot and radiation) power spetral density.
    """
    gravitationalWaveOmega = 2 * np.pi * freq
    kappa = get_coupling_constant(gravitationalWaveOmega, config=config)
    h_sql_sq = get_standard_quantum_limit(gravitationalWaveOmega, config=config) ** 2
    return (h_sql_sq / 2.0) * (kappa + 1 / kappa)

def squeeze_db_to_r(squeeze_db: float) -> float:
    """
    Convert squeezing level in dB (power reduction in the squeezed quadrature)
    to the squeeze parameter r, with e^(-2r) = 10^(-squeeze_db/10).
    """
    return squeeze_db * np.log(10.0) / 20.0

def squeeze_quantum_noise_with_same_angle(
    freq,
    squeeze_db: float = 10.0,
    config: DetectorConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = DetectorConfig()
    """
    Total quantum-noise power spectral density with frequency-independent
    squeezed vacuum at the antisymmetric port. In the usual κ = (shot)/(RP)
    decomposition, squeezing the phase quadrature gives

        S_h = (h_SQL^2 / 2) * (e^(-2r) κ + e^(2r) / κ),

    with r fixed by squeeze_db. Unsqueezed noise is recovered at squeeze_db = 0.
    """
    freq = np.asarray(freq, dtype=float)
    gravitationalWaveOmega = 2 * np.pi * freq
    r = squeeze_db_to_r(squeeze_db)
    e_m2r = np.exp(-2.0 * r)
    e_p2r = np.exp(2.0 * r)
    h_sql_sq = get_standard_quantum_limit(gravitationalWaveOmega, config=config) ** 2
    kappa = get_coupling_constant(gravitationalWaveOmega, config=config)
    return (h_sql_sq / 2.0) * (e_m2r * kappa + e_p2r / kappa)

def squeeze_quantum_noise_with_varying_angle(
    freq,
    squeeze_db: float = 10.0,
    config: DetectorConfig | None = None,
) -> np.array:
    if config is None:
        config = DetectorConfig()
    """
    Total quantum-noise power spectral density with frequency-dependent
    squeezed vacuum at the antisymmetric port. In the usual κ = (shot)/(RP)
    decomposition, squeezing the phase quadrature gives

        S_h = (h_SQL^2 / 2) * (κ + 1 / κ) e^-2r
    
    with r decided by squeeze_db.
    """
    freq = np.asarray(freq, dtype=float)
    r = squeeze_db_to_r(squeeze_db)
    gravitationalWaveOmega = 2 * np.pi * freq
    h_sql_sq = get_standard_quantum_limit(gravitationalWaveOmega, config=config) ** 2
    kappa = get_coupling_constant(gravitationalWaveOmega, config=config)
    return (h_sql_sq / 2.0) * (kappa + 1 / kappa) * np.exp(-2.0 * r)


def plot():
    freq = np.linspace(100, 1000, 10000)
    squeeze_db = 10.0

    # Figure 1: unsqueezed model vs gwinc aLIGO (quantum trace only)
    plt.figure(figsize=(10, 6))
    noise_psd = get_quantum_noise_psd(freq)
    noise_asd = np.sqrt(noise_psd)
    plt.plot(
        freq,
        noise_asd,
        label="Quantum noise (model, unsqueezed)",
        color="blue",
        linewidth=2,
    )

    budget = gwinc.load_budget("aLIGO")
    budget.ifo.Optics.SRM.Transmittance = 1
    trace = budget.run(freq=freq)
    aLIGO_noise = trace["Quantum"]
    plt.plot(
        freq,
        aLIGO_noise.asd,
        label="aLIGO quantum noise (gwinc)",
        color="red",
        linewidth=2,
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Quantum noise [1/sqrt(Hz)]")
    plt.legend(loc="upper right", fontsize="small")
    plt.title("Quantum noise before squeezing (ASD)")
    plt.savefig("./Figure/Quantum Noise (Before Squeezing).png")
    plt.close()

    # Figure 2: ~10 dB squeezing — analytic model vs gwinc with Squeezer enabled
    plt.figure(figsize=(10, 6))
    psd_sqz = squeeze_quantum_noise_with_varying_angle(freq, squeeze_db=squeeze_db)
    noise_sqz_asd = np.sqrt(psd_sqz)
    plt.plot(
        freq,
        noise_sqz_asd,
        label=f"Quantum Noise (model, {squeeze_db:.0f} dB squeeze)",
        color="blue",
        linewidth=2,
    )

    budget_sqz = gwinc.load_budget("aLIGO")
    budget_sqz.ifo.Optics.SRM.Transmittance = 1.0
    budget_sqz.ifo.Squeezer = Struct(
        Type="Freq Independent",
        AmplitudedB=squeeze_db,
        AntiAmplitudedB=squeeze_db,
        SQZAngle=0.0,
        InjectionLoss=0.0,
    )
    trace_sqz = budget_sqz.run(freq=freq)
    plt.plot(
        freq,
        trace_sqz["Quantum"].asd,
        label=f"aLIGO Quantum Noise (gwinc, {squeeze_db:.0f} dB)",
        color="red",
        linewidth=2,
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Quantum noise [1/sqrt(Hz)]")
    plt.legend(loc="upper right", fontsize="small")
    plt.title("Quantum noise after squeezing (ASD)")
    plt.savefig("./Figure/Quantum Noise (After Squeezing).png")
    plt.close()

if __name__ == "__main__":
    plot()
