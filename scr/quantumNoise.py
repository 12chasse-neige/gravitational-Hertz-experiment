from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ghe.config import DetectorConfig, IMG_DIR
from ghe.noise import (
    get_detuned_signal_recycling_noise_psd,
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


def get_gwinc_quantum_asd(freq: np.ndarray, squeeze_db: float, srm: float = 1.0, l_sr: float = 55.0, phi_sr: float = 0.0) -> np.ndarray:
    gwinc, Struct = require_gwinc()

    budget = gwinc.load_budget("aLIGO")
    budget.ifo.Optics.SRM.Transmittance = srm
    budget.ifo.Optics.SRM.CavityLength = l_sr 
    budget.ifo.Optics.SRM.Tunephase = phi_sr
    budget.ifo.Squeezer = Struct(
        Type="Freq Dependent",
        AmplitudedB=squeeze_db,
        AntiAmplitudedB=squeeze_db,
        SQZAngle=0.0,
        InjectionLoss=0.0,
    )
    trace = budget.run(freq=freq)
    return trace["Quantum"].asd


def plot_noise_curve_with_detuned_interferometer(
    *,
    output_path: Path = IMG_DIR / "Quantum Noise (With Detuned Interferometer).png",
    freq_min_hz: float = 10.0,
    freq_max_hz: float = 1000.0,
    points: int = 10000,
    squeeze_db: float = 10.0,
    detector_config: DetectorConfig | None = None,
) -> float:
    if freq_max_hz <= freq_min_hz:
        raise ValueError("freq_max_hz must be greater than freq_min_hz.")
    if points < 2:
        raise ValueError("points must be at least 2.")

    active_detector = detector_config or DetectorConfig()
    # The quantum-noise formulas are singular at DC, so compute just above 0 Hz
    # while displaying the requested 0 Hz lower axis bound.
    calculation_min_hz = max(freq_min_hz, 1.0)
    freq = np.geomspace(calculation_min_hz, freq_max_hz, points)
    target_frequency_hz = active_detector.resonance_frequency_hz
    if freq_min_hz <= target_frequency_hz <= freq_max_hz:
        freq = np.unique(np.append(freq, target_frequency_hz))
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
    # Keep the GWINC curve as an aLIGO reference. The two analytic curves below
    # then isolate the effect of the project's model and its detuned SR extension.
    gwinc_asd = get_gwinc_quantum_asd(freq, squeeze_db=squeeze_db)

    target_detuned_asd = float(
        np.sqrt(
            get_detuned_signal_recycling_noise_psd(
                np.array([target_frequency_hz]),
                squeeze_db=squeeze_db,
                config=active_detector,
            )[0]
        )
    )
    target_asd_exponent = int(np.floor(np.log10(target_detuned_asd)))
    target_asd_mantissa = target_detuned_asd / 10.0**target_asd_exponent

    matplotlib_cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "ghe-matplotlib-cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    import matplotlib.pyplot as plt

    paper_style = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "legend.fontsize": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    blue = "#0072B2"
    vermillion = "#D55E00"
    bluish_green = "#009E73"

    with plt.rc_context(paper_style):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.loglog(
            freq,
            gwinc_asd,
            color=blue,
            linestyle="-",
            linewidth=1.4,
            label=rf"GWINC aLIGO quantum noise + {squeeze_db:g} dB squeezing",
        )
        ax.loglog(
            freq,
            previous_asd,
            color=vermillion,
            linestyle="--",
            linewidth=1.4,
            label=rf"Analytic model + {squeeze_db:g} dB FD squeezing",
        )
        ax.loglog(
            freq,
            detuned_asd,
            color=bluish_green,
            linestyle="-.",
            linewidth=1.45,
            label=rf"Detuned SR model + {squeeze_db:g} dB FD squeezing",
        )
        ax.axvline(target_frequency_hz, color="0.55", linestyle=":", linewidth=0.9)
        ax.plot(
            target_frequency_hz,
            target_detuned_asd,
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=bluish_green,
            markeredgewidth=1.0,
            zorder=5,
        )
        ax.annotate(
            rf"$f_0={target_frequency_hz:g}\,\mathrm{{Hz}}$"
            "\n"
            rf"$\sqrt{{S_h}}={target_asd_mantissa:.2f}\times10^{{{target_asd_exponent}}}"
            rf"\,\mathrm{{Hz}}^{{-1/2}}$",
            xy=(target_frequency_hz, target_detuned_asd),
            xytext=(-8, 31),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="0.2",
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.7},
        )
        ax.set_xlim(freq_min_hz, freq_max_hz)
        ax.grid(which="major", color="0.88", linewidth=0.65)
        ax.grid(which="minor", color="0.94", linewidth=0.4)
        ax.set_xlabel(r"Frequency $f$ [Hz]")
        ax.set_ylabel(r"Quantum-noise ASD $\sqrt{S_h(f)}$ [$\mathrm{Hz}^{-1/2}$]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.2)
        ax.legend(frameon=False, loc="lower left", handlelength=3.0)
        fig.subplots_adjust(left=0.12, right=0.99, bottom=0.16, top=0.98)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
        vector_output_path = output_path.with_suffix(".pdf")
        if vector_output_path != output_path:
            fig.savefig(vector_output_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print(f"Saved figure: {output_path}")
    print(
        f"Detuned ASD at {target_frequency_hz:g} Hz = "
        f"{target_detuned_asd:.17e} Hz^(-1/2)"
    )
    return target_detuned_asd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quantum-noise model comparisons.")
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
        detector_config = replace(detector_config, length_SR=args.length_sr, phi_SR=None)
    if args.t_srm is not None:
        detector_config = replace(detector_config, T_SRM=args.t_srm, phi_SR=None)

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
