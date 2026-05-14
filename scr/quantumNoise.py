from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ghe.config import DetectorConfig, IMG_DIR
from ghe.noise import (
    get_detuned_signal_recycling_noise_psd,
    get_quantum_noise_psd,
    squeeze_quantum_noise_with_varying_angle,
)


class MissingOptionalDependency(RuntimeError):
    """Raised when an optional plotting dependency is not installed."""


def require_gwinc() -> tuple[Any, Any]:
    try:
        import gwinc
        from gwinc import Struct
    except ModuleNotFoundError as exc:
        if exc.name == "gwinc":
            raise MissingOptionalDependency(
                "The gwinc comparison curve requires gwinc. "
                "Install project dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        raise
    return gwinc, Struct


def get_gwinc_quantum_asd(freq: np.ndarray, squeeze_db: float, srm: float = 1.0) -> np.ndarray:
    gwinc, Struct = require_gwinc()

    budget = gwinc.load_budget("aLIGO")
    budget.ifo.Optics.SRM.Transmittance = srm
    budget.ifo.Squeezer = Struct(
        Type="Freq Independent",
        AmplitudedB=squeeze_db,
        AntiAmplitudedB=squeeze_db,
        SQZAngle=0.0,
        InjectionLoss=0.0,
    )
    trace = budget.run(freq=freq)
    return trace["Quantum"].asd


def plot_noise_curve_before_and_after_squeezing(squeeze_db: float = 10.0) -> None:
    gwinc, _ = require_gwinc()
    import matplotlib.pyplot as plt

    freq = np.linspace(100, 1000, 10000)

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
    plt.savefig(IMG_DIR / "Quantum Noise (Before Squeezing).png")
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

    gwinc_sqz_asd = get_gwinc_quantum_asd(freq, squeeze_db=squeeze_db)
    plt.plot(
        freq,
        gwinc_sqz_asd,
        label=f"aLIGO Quantum Noise (gwinc, {squeeze_db:.0f} dB)",
        color="red",
        linewidth=2,
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Quantum noise [1/sqrt(Hz)]")
    plt.legend(loc="upper right", fontsize="small")
    plt.title("Quantum noise after squeezing (ASD)")
    plt.savefig(IMG_DIR / "Quantum Noise (After Squeezing).png")
    plt.close()


def plot_noise_curve_with_detuned_interferometer(
    *,
    output_path: Path = IMG_DIR / "Quantum Noise (With Detuned Interferometer).png",
    freq_min_hz: float = 10.0,
    freq_max_hz: float = 1000.0,
    points: int = 10000,
    squeeze_db: float = 10.0,
    detector_config: DetectorConfig | None = None,
) -> None:
    if freq_max_hz <= freq_min_hz:
        raise ValueError("freq_max_hz must be greater than freq_min_hz.")
    if points < 2:
        raise ValueError("points must be at least 2.")

    active_detector = detector_config or DetectorConfig()
    # The quantum-noise formulas are singular at DC, so compute just above 0 Hz
    # while displaying the requested 0 Hz lower axis bound.
    calculation_min_hz = max(freq_min_hz, 1.0)
    freq = np.linspace(calculation_min_hz, freq_max_hz, points)
    previous_asd = np.sqrt(
        squeeze_quantum_noise_with_varying_angle(
            freq,
            squeeze_db=squeeze_db,
            config=active_detector,
        )
    )
    detuned_asd = np.sqrt(
        get_detuned_signal_recycling_noise_psd(
            freq,
            squeeze_db=squeeze_db,
            config=active_detector,
        )
    )
    gwinc_asd = get_gwinc_quantum_asd(freq, squeeze_db=squeeze_db, srm=active_detector.T_SRM)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freq, gwinc_asd, label=f"gwinc aLIGO ({squeeze_db:.0f} dB)", linewidth=2)
    ax.loglog(freq, previous_asd, label="previous model", linewidth=2)
    ax.loglog(
        freq,
        detuned_asd,
        label=(
            "detuned interferometer"
            f"(T_SRM={active_detector.T_SRM:g}, L_SR={active_detector.length_SR:g} m)"
        ),
        linewidth=2,
    )
    ax.set_xlim(freq_min_hz, freq_max_hz)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Quantum noise [1/sqrt(Hz)]")
    ax.set_title("Quantum noise curve comparison")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quantum-noise model comparisons.")
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only draw the gwinc/previous/detuned comparison figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / "Quantum Noise (With Detuned Interferometer).png",
        help="Output path for the comparison figure.",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=10.0,
        help="Minimum frequency in Hz.",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=1000.0,
        help="Maximum frequency in Hz.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=10000,
        help="Number of frequency samples.",
    )
    parser.add_argument(
        "--squeeze-db",
        type=float,
        default=10.0,
        help="Squeezing level in dB.",
    )
    parser.add_argument(
        "--length-sr",
        type=float,
        default=None,
        help="Override DetectorConfig.length_SR for the detuned SR curve.",
    )
    parser.add_argument(
        "--t-srm",
        type=float,
        default=None,
        help="Override DetectorConfig.T_SRM for the detuned SR curve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    detector_config = DetectorConfig()
    if args.length_sr is not None:
        detector_config = replace(detector_config, length_SR=args.length_sr)
    if args.t_srm is not None:
        detector_config = replace(detector_config, T_SRM=args.t_srm)

    if not args.comparison_only:
        plot_noise_curve_before_and_after_squeezing(squeeze_db=args.squeeze_db)
    plot_noise_curve_with_detuned_interferometer(
        output_path=args.output,
        freq_min_hz=args.freq_min,
        freq_max_hz=args.freq_max,
        points=args.points,
        squeeze_db=args.squeeze_db,
        detector_config=detector_config,
    )


if __name__ == "__main__":
    try:
        main()
    except MissingOptionalDependency as exc:
        raise SystemExit(str(exc))
