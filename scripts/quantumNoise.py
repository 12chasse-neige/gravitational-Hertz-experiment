from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import DetectorConfig, IMAGES_DIR
from ghe.noise import (
    get_coupling_constant,
    get_laser_power_in_cavity,
    get_quantum_noise_psd,
    get_standard_quantum_limit,
    squeeze_db_to_r,
    squeeze_quantum_noise_with_same_angle,
    squeeze_quantum_noise_with_varying_angle,
)


def plot():
    import matplotlib.pyplot as plt
    import gwinc
    from gwinc import Struct

    freq = np.linspace(100, 1000, 10000)
    squeeze_db = 10.0

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
    plt.savefig(IMAGES_DIR / "Quantum Noise (Before Squeezing).png")
    plt.close()

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
    plt.savefig(IMAGES_DIR / "Quantum Noise (After Squeezing).png")
    plt.close()


if __name__ == "__main__":
    plot()
