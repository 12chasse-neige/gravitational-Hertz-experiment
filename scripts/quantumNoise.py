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
from scipy.optimize import brentq

from ghe.config import DETECTOR_CONFIG_FILE, DetectorConfig, PAPER_FIGURES_DIR
from ghe.noise import (
    get_detuned_signal_recycling_noise_psd,
    squeeze_quantum_noise_with_varying_angle,
)
from ghe.paths import CONFIG_DIR
from ghe.phase_convention import gwinc_tunephase_from_analytic_phi


CE2_SILICON_CONFIG_FILE = CONFIG_DIR / "CE2silicon.yaml"


class MissingOptionalDependency(RuntimeError):
    """Raised when an optional plotting dependency is not installed."""


def require_gwinc() -> Any:
    try:
        import gwinc
    except ModuleNotFoundError as exc:
        if exc.name == "gwinc":
            raise MissingOptionalDependency(
                "The gwinc comparison curve requires gwinc. "
                "Install project dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        raise
    return gwinc


def _gwinc_quantum_psd(freq: np.ndarray, ifo: Any) -> np.ndarray:
    """Evaluate GWINC quantum noise and its finite-arm strain calibration.

    GWINC 0.6.2's full Quantum budget requests a phase-noise subterm even when
    zero squeezing omits that subterm.  Its ``shotrad`` function still returns
    all physical vacuum terms in displacement PSD.  Sum those independent
    terms once and apply the same strain calibration as the full budget.
    """
    from gwinc.ifo.noises import dhdl, ifo_power
    from gwinc.noise.quantum import shotrad
    from gwinc.suspension import precomp_suspension

    suspension = precomp_suspension(freq, ifo)
    power = ifo_power(ifo)
    displacement_terms = shotrad(freq, ifo, suspension, power)
    displacement_psd = sum(
        displacement_terms.values(), np.zeros_like(freq, dtype=float)
    )
    strain_calibration, _ = dhdl(freq, ifo.Infrastructure.Length)
    return displacement_psd * strain_calibration


def _load_unsqueezed_gwinc_ifo(path: str | Path) -> Any:
    """Load GWINC optics with its physical squeezer disabled.

    Both GWINC curves use the same ideal ASD multiplier as the analytic
    curves.  Applying it after the unsqueezed quantum calculation makes the
    comparison independent of the YAML filter-cavity design.
    """
    ifo = require_gwinc().load_budget(str(path)).ifo
    ifo.Squeezer.AmplitudedB = 0.0
    ifo.Squeezer.Type = "None"
    return ifo


def tune_gwinc_srm_phase(ifo: Any, target_frequency_hz: float) -> float:
    """Place a local GWINC quantum-PSD minimum at the target frequency.

    ``Optics.SRM.Tunephase`` is GWINC's public detuning parameter.  Its
    signal-recycled optical model includes the finite SR-cavity propagation,
    so the analytic model's phase is not a reliable phase for this curve.
    The ideal squeezing multiplier is constant in frequency and cannot move
    the minimum; tune with the unsqueezed GWINC PSD instead.
    """
    if not np.isfinite(target_frequency_hz) or target_frequency_hz <= 0:
        raise ValueError("The GWINC tuning frequency must be finite and positive.")
    transmittance = ifo.Optics.SRM.Transmittance
    if not np.isfinite(transmittance) or not 0 < transmittance < 1:
        raise ValueError("A tunable GWINC SR dip requires 0 < SRM transmittance < 1.")

    # Differentiate log PSD rather than PSD to keep the root calculation well
    # scaled at strain-noise values near 1e-50 Hz^-1.  A short symmetric step
    # resolves the narrow resonance without changing its measured location.
    step_hz = min(0.01, 1e-4 * target_frequency_hz)
    probe_freq = target_frequency_hz + np.array([-step_hz, 0.0, step_hz])

    def phase_statistics(tunephase: float) -> tuple[float, float, float]:
        ifo.Optics.SRM.Tunephase = float(tunephase)
        # Some scan phases have zero signal in the selected quadrature.
        # Reject their nonfinite PSDs below without warning at each probe.
        with np.errstate(divide="ignore", invalid="ignore"):
            psd = _gwinc_quantum_psd(probe_freq, ifo)
        if not np.all(np.isfinite(psd)) or np.any(psd <= 0):
            return np.nan, np.nan, np.nan
        log_psd = np.log(psd)
        slope = (log_psd[2] - log_psd[0]) / (2 * step_hz)
        curvature = (log_psd[2] - 2 * log_psd[1] + log_psd[0]) / step_hz**2
        return float(slope), float(curvature), float(psd[1])

    # Search the positive-detuning branch.  A zero frequency slope can also
    # describe a maximum; keep only roots with positive frequency curvature.
    phase_grid = np.linspace(0.0, np.pi, 1001)
    slopes = [phase_statistics(phase)[0] for phase in phase_grid]
    candidates: list[tuple[float, float]] = []
    for left, right, left_slope, right_slope in zip(
        phase_grid[:-1], phase_grid[1:], slopes[:-1], slopes[1:]
    ):
        if not np.isfinite(left_slope) or not np.isfinite(right_slope):
            continue
        if left_slope == 0:
            root = float(left)
        elif left_slope * right_slope < 0:
            root = brentq(
                lambda phase: phase_statistics(phase)[0],
                left,
                right,
                xtol=1e-13,
            )
        else:
            continue
        _, curvature, target_psd = phase_statistics(root)
        if np.isfinite(curvature) and curvature > 0:
            candidates.append((target_psd, root))

    if not candidates:
        raise ValueError(
            f"GWINC has no positive-detuning quantum-noise dip at "
            f"{target_frequency_hz:g} Hz for the selected SRM settings."
        )
    tunephase = min(candidates)[1]
    ifo.Optics.SRM.Tunephase = tunephase
    return tunephase


def get_gwinc_quantum_asd(
    freq: np.ndarray,
    squeeze_db: float | None = None,
    ifo: str | Path = CE2_SILICON_CONFIG_FILE,
) -> np.ndarray:
    """CE2 quantum ASD without signal recycling, with ideal dB scaling.

    The unsqueezed GWINC curve includes CE2 optical and readout losses.  The
    final multiplier models the same ideal frequency-dependent squeezing rule
    used by the analytic curves; it does not simulate CE2's filter cavity.
    """
    freq = np.asarray(freq, dtype=float)
    if freq.ndim != 1 or not np.all(np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("freq must be a one-dimensional array of positive frequencies.")
    active_squeeze_db = (
        DetectorConfig.from_yaml(DETECTOR_CONFIG_FILE).squeeze_db
        if squeeze_db is None
        else squeeze_db
    )
    if not np.isfinite(active_squeeze_db) or active_squeeze_db < 0:
        raise ValueError("squeeze_db must be finite and nonnegative.")

    gwinc_ifo = _load_unsqueezed_gwinc_ifo(ifo)
    # Fully transmitting SRM removes recycling.  Zero squeezing bypasses the
    # YAML's physical filter cavity, injection loss, and LO angle jitter.
    gwinc_ifo.Optics.SRM.Transmittance = 1.0
    gwinc_ifo.Optics.SRM.Tunephase = 0.0
    unsqueezed_psd = _gwinc_quantum_psd(freq, gwinc_ifo)
    return np.sqrt(unsqueezed_psd) * 10.0 ** (-active_squeeze_db / 20.0)


def get_gwinc_detuned_quantum_asd(
    freq: np.ndarray,
    *,
    detector_config: DetectorConfig,
    squeeze_db: float | None = None,
    ifo: str | Path = CE2_SILICON_CONFIG_FILE,
    sr_tuning: str = "target-minimum",
) -> tuple[np.ndarray, float]:
    """Return ideal-scaled GWINC b2 quantum ASD with detuned SR.

    CE2 supplies the optical plant; the analytic detector supplies only the
    SRM transmittance, SR-cavity length, target frequency, and shared dB level.
    The readout selects the same Buonanno--Chen b2 quadrature as the analytic
    detuned model: GWINC's ``[sin(eta), cos(eta)]`` projection requires eta=0.
    ``target-minimum`` finds a GWINC dip at the source frequency.
    ``analytic-phase`` instead uses the analytic detector's physical detuning,
    including when the GWINC model has no such frequency minimum.
    """
    freq = np.asarray(freq, dtype=float)
    if freq.ndim != 1 or not np.all(np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("freq must be a one-dimensional array of positive frequencies.")
    active_squeeze_db = (
        detector_config.squeeze_db if squeeze_db is None else squeeze_db
    )
    if not np.isfinite(active_squeeze_db) or active_squeeze_db < 0:
        raise ValueError("squeeze_db must be finite and nonnegative.")
    if sr_tuning not in ("target-minimum", "analytic-phase"):
        raise ValueError("sr_tuning must be 'target-minimum' or 'analytic-phase'.")

    gwinc_ifo = _load_unsqueezed_gwinc_ifo(ifo)
    gwinc_ifo.Optics.SRM.Transmittance = detector_config.T_SRM
    gwinc_ifo.Optics.SRM.CavityLength = detector_config.length_SR
    gwinc_ifo.Optics.Quadrature.dc = 0.0
    # Override an explicit YAML readout as well as the legacy Quadrature.dc
    # field so GWINC's consistency check and signal projection agree.
    gwinc_ifo.Squeezer.Readout = require_gwinc().Struct(Type="Homodyne", Angle=0.0)
    if sr_tuning == "analytic-phase":
        tunephase = gwinc_tunephase_from_analytic_phi(detector_config.phi_SR)
        gwinc_ifo.Optics.SRM.Tunephase = tunephase
    else:
        tunephase = tune_gwinc_srm_phase(
            gwinc_ifo, detector_config.resonance_frequency_hz
        )
    unsqueezed_psd = _gwinc_quantum_psd(freq, gwinc_ifo)
    return np.sqrt(unsqueezed_psd) * 10.0 ** (-active_squeeze_db / 20.0), tunephase


def plot_noise_curve_with_detuned_interferometer(
    *,
    output_path: Path = (
        PAPER_FIGURES_DIR / "Quantum Noise (With Detuned Interferometer).png"
    ),
    freq_min_hz: float = 10.0,
    freq_max_hz: float = 1000.0,
    points: int = 10000,
    squeeze_db: float | None = None,
    detector_config: DetectorConfig | None = None,
    gwinc_ifo: str | Path = CE2_SILICON_CONFIG_FILE,
    gwinc_sr_tuning: str = "target-minimum",
) -> float:
    if freq_max_hz <= freq_min_hz:
        raise ValueError("freq_max_hz must be greater than freq_min_hz.")
    if points < 2:
        raise ValueError("points must be at least 2.")

    active_detector = detector_config or DetectorConfig.from_yaml(DETECTOR_CONFIG_FILE)
    active_squeeze_db = active_detector.squeeze_db if squeeze_db is None else squeeze_db
    # The quantum-noise formulas are singular at DC, so compute just above 0 Hz
    # while displaying the requested 0 Hz lower axis bound.
    calculation_min_hz = max(freq_min_hz, 1.0)
    target_frequency_hz = active_detector.resonance_frequency_hz
    if not calculation_min_hz <= target_frequency_hz <= freq_max_hz:
        raise ValueError("The tuning frequency must lie inside the plotted frequency band.")
    freq = np.unique(
        np.append(np.geomspace(calculation_min_hz, freq_max_hz, points), target_frequency_hz)
    )
    previous_asd = np.sqrt(
        squeeze_quantum_noise_with_varying_angle(
            freq,
            squeeze_db=active_squeeze_db,
            config=active_detector,
        )
    )
    detuned_asd = np.sqrt(
        get_detuned_signal_recycling_noise_psd(
            freq,
            squeeze_db=active_squeeze_db,
            config=active_detector,
        )
    )
    # Use CE2 optics with no signal recycling.  The squeeze level still comes
    # from the analytic detector, so a detector YAML edit moves all curves.
    gwinc_asd = get_gwinc_quantum_asd(
        freq,
        squeeze_db=active_squeeze_db,
        ifo=gwinc_ifo,
    )
    gwinc_detuned_asd, gwinc_tunephase = get_gwinc_detuned_quantum_asd(
        freq,
        detector_config=active_detector,
        squeeze_db=active_squeeze_db,
        ifo=gwinc_ifo,
        sr_tuning=gwinc_sr_tuning,
    )

    config_name = Path(gwinc_ifo).stem
    if config_name == "CE2silicon":
        config_label = "CE2 silicon optics"
    else:
        config_label = config_name
    gwinc_sr_label = (
        "tuned SR" if gwinc_sr_tuning == "target-minimum" else "SR, analytic detuning"
    )

    target_detuned_asd = float(
        np.sqrt(
            get_detuned_signal_recycling_noise_psd(
                np.array([target_frequency_hz]),
                squeeze_db=active_squeeze_db,
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
    purple = "#CC79A7"

    with plt.rc_context(paper_style):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.loglog(
            freq,
            gwinc_asd,
            color=blue,
            linestyle="-",
            linewidth=1.4,
            label=rf"GWINC {config_label}, no SR; ideal FD {active_squeeze_db:g} dB",
        )
        ax.loglog(
            freq,
            gwinc_detuned_asd,
            color=purple,
            linestyle=":",
            linewidth=1.7,
            label=rf"GWINC {config_label}, {gwinc_sr_label}, $b_2$; ideal FD {active_squeeze_db:g} dB",
        )
        ax.loglog(
            freq,
            previous_asd,
            color=vermillion,
            linestyle="--",
            linewidth=1.4,
            label=rf"Analytic model, no SR; ideal FD {active_squeeze_db:g} dB",
        )
        ax.loglog(
            freq,
            detuned_asd,
            color=bluish_green,
            linestyle="-.",
            linewidth=1.45,
            label=rf"Analytic model, detuned SR, $b_2$; ideal FD {active_squeeze_db:g} dB",
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
        target_index = int(np.flatnonzero(freq == target_frequency_hz)[0])
        ax.plot(
            target_frequency_hz,
            gwinc_detuned_asd[target_index],
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=purple,
            markeredgewidth=1.0,
            zorder=5,
        )
        ax.annotate(
            rf"$f_0={target_frequency_hz:g}\,\mathrm{{Hz}}$"
            "\n"
            rf"$\sqrt{{S_h}}={target_asd_mantissa:.2f}\times10^{{{target_asd_exponent}}}"
            rf"\,\mathrm{{Hz}}^{{-1/2}}$",
            xy=(target_frequency_hz, target_detuned_asd),
            xytext=(-20, 10),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="0.2",
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.7},
        )
        ax.set_xlim(freq_min_hz, freq_max_hz)
        ax.text(
            0.98,
            0.97,
            rf"$T_{{\mathrm{{SRM}}}}={active_detector.T_SRM:g}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        ax.grid(which="major", color="0.88", linewidth=0.65)
        ax.grid(which="minor", color="0.94", linewidth=0.4)
        ax.set_xlabel(r"Frequency $f$ [Hz]")
        ax.set_ylabel(r"Quantum-noise ASD $\sqrt{S_h(f)}$ [$\mathrm{Hz}^{-1/2}$]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.2)
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
            loc="upper left",
            handlelength=3.0,
        )
        fig.subplots_adjust(left=0.12, right=0.99, bottom=0.16, top=0.98)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
        vector_output_path = output_path.with_suffix(".pdf")
        if vector_output_path != output_path:
            fig.savefig(vector_output_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print(f"Saved figure: {output_path}")
    print("GWINC detuned readout: b_2 (Optics.Quadrature.dc = 0 rad)")
    print(f"GWINC SR tuning: {gwinc_sr_tuning}")
    phase_label = (
        "GWINC tuned SRM Tunephase"
        if gwinc_sr_tuning == "target-minimum"
        else "GWINC SRM Tunephase (analytic detuning)"
    )
    print(
        f"{phase_label} at {target_frequency_hz:g} Hz = "
        f"{gwinc_tunephase:.14f} rad "
        f"(T_SRM={active_detector.T_SRM:g}, L_SR={active_detector.length_SR:g} m)"
    )
    target_indices = np.flatnonzero(freq == target_frequency_hz)
    if target_indices.size:
        target_index = int(target_indices[0])
        print(
            f"GWINC CE2 no-SR ideal ASD at {target_frequency_hz:g} Hz = "
            f"{gwinc_asd[target_index]:.17e} Hz^(-1/2)"
        )
        sr_value_label = (
            "tuned-SR"
            if gwinc_sr_tuning == "target-minimum"
            else "matched-detuning SR"
        )
        print(
            f"GWINC CE2 {sr_value_label} ideal ASD at {target_frequency_hz:g} Hz = "
            f"{gwinc_detuned_asd[target_index]:.17e} Hz^(-1/2)"
        )
        print(
            f"Analytic no-SR ideal ASD at {target_frequency_hz:g} Hz = "
            f"{previous_asd[target_index]:.17e} Hz^(-1/2)"
        )
    print(
        f"Analytic detuned-SR ideal ASD at {target_frequency_hz:g} Hz = "
        f"{target_detuned_asd:.17e} Hz^(-1/2)"
    )
    return target_detuned_asd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quantum-noise model comparisons.")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PAPER_FIGURES_DIR / "Quantum Noise (With Detuned Interferometer).png"
        ),
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
        default=None,
        help="Optional one-run dB override; default comes from detector.yaml Squeezer.AmplitudedB.",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=DETECTOR_CONFIG_FILE,
        help=(
            "Detector YAML for the analytic curves and shared squeezing level. "
            "Default: configs/detector.yaml."
        ),
    )
    parser.add_argument(
        "--gwinc-config",
        type=Path,
        default=CE2_SILICON_CONFIG_FILE,
        help="GWINC optics YAML for both GWINC curves. Default: configs/CE2silicon.yaml.",
    )
    parser.add_argument(
        "--target-frequency-hz",
        type=float,
        default=None,
        help="One-run tuning target for both detuned curves; default: source GW frequency.",
    )
    parser.add_argument(
        "--gwinc-sr-tuning",
        choices=("target-minimum", "analytic-phase"),
        default="target-minimum",
        help=(
            "Find a GWINC minimum at the target (default), or use the analytic "
            "detuning even when no GWINC target-frequency minimum exists."
        ),
    )
    parser.add_argument(
        "--length-sr",
        type=float,
        default=None,
        help="Override SR cavity length in both detuned curves.",
    )
    parser.add_argument(
        "--t-srm",
        type=float,
        default=None,
        help="Override SRM transmittance in both detuned curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    detector_config = DetectorConfig.from_yaml(args.detector_config)
    if args.target_frequency_hz is not None:
        if not np.isfinite(args.target_frequency_hz) or args.target_frequency_hz <= 0:
            raise ValueError("--target-frequency-hz must be finite and positive.")
        detector_config = replace(
            detector_config,
            resonance_frequency_hz=args.target_frequency_hz,
            phi_SR=None,
        )
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
        gwinc_ifo=args.gwinc_config,
        gwinc_sr_tuning=args.gwinc_sr_tuning,
    )


if __name__ == "__main__":
    try:
        main()
    except MissingOptionalDependency as exc:
        raise SystemExit(str(exc))
