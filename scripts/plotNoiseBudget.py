"""Plot the SNR noise budget with analytic quantum and GWINC thermal/gas noise."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from ghe.config import ConfigFileError, DetectorConfig, NoiseConfig, SourceConfig
from ghe.noise import (
    DETUNED_SIGNAL_RECYCLING_NOISE_MODEL,
    get_noise_psd,
    normalize_noise_model,
)
from ghe.paths import DETECTOR_CONFIG_FILE, PAPER_FIGURES_DIR
from ghe.thermal_gas import THERMAL_COMPONENTS, calculate_thermal_gas, resolve_config_path


DEFAULT_OUTPUT = PAPER_FIGURES_DIR / "SNR Noise Budget.png"

# Match scripts/quantumNoise.py's paper typography and line treatment.
PAPER_STYLE = {
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
COMPONENT_STYLES = {
    "SuspensionThermal": ("Suspension thermal", "#0072B2", "-"),
    "CoatingBrownian": ("Coating Brownian", "#D55E00", "-"),
    "CoatingThermoOptic": ("Coating thermo-optic", "#E69F00", "--"),
    "ITMThermoRefractive": ("ITM thermo-refractive", "#56B4E9", ":"),
    "SubstrateBrownian": ("Substrate Brownian", "#B79F00", "--"),
    "SubstrateThermoElastic": ("Substrate thermoelastic", "#8C564B", "-."),
    "GasScattering": ("Gas scattering", "#CC79A7", "-."),
    "GasDamping": ("Gas damping", "#009E73", ":"),
}


@dataclass
class NoiseCurves:
    frequency_hz: np.ndarray
    spectra_psd: dict[str, np.ndarray]
    detector_config: DetectorConfig
    noise_config: NoiseConfig
    frequency_band_hz: tuple[float, float]

    @property
    def target_frequency_hz(self) -> float:
        return self.detector_config.resonance_frequency_hz


def calculate_noise_curves(
    *,
    detector_path: str | Path = DETECTOR_CONFIG_FILE,
    detector_config: DetectorConfig | None = None,
    noise_config: NoiseConfig | None = None,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    points: int = 1000,
) -> NoiseCurves:
    """Evaluate the independent PSDs used by Qdot and array SNR calculations.

    GWINC supplies the six available thermal terms and gas scattering/damping.
    Its quantum budget is never evaluated. The selected analytic quantum PSD
    replaces it, with ideal squeezing applied only to that contribution.
    """
    noise = noise_config or NoiseConfig()
    lower = noise.min_frequency_hz if freq_min_hz is None else freq_min_hz
    upper = noise.max_frequency_hz if freq_max_hz is None else freq_max_hz
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0 < lower < upper:
        raise ValueError("Frequency bounds must be finite and satisfy 0 < min < max.")
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("points must be an integer >= 2.")
    path = resolve_config_path(detector_path)
    if detector_config is not None:
        detector = detector_config
    elif path == DETECTOR_CONFIG_FILE.resolve():
        # Use the active detector, including runtime environment fields.
        detector = DetectorConfig().with_source(SourceConfig())
    else:
        detector = DetectorConfig.from_yaml(path)
        detector = detector.with_source(SourceConfig(L=detector.length))
    squeeze_db = noise.effective_squeeze_db(detector)
    if not np.isfinite(squeeze_db) or squeeze_db < 0:
        raise ValueError("Squeezing must be finite and nonnegative.")

    frequency = np.geomspace(lower, upper, points)
    target = detector.resonance_frequency_hz
    if lower <= target <= upper:
        # A coarse log grid plus one exact target sample broadens the drawn
        # narrow SR dip. Resolve both wings with logarithmic offsets instead.
        offsets = target * np.geomspace(1e-9, 0.02, 160)
        local = np.concatenate((target - offsets, [target], target + offsets))
        frequency = np.unique(np.append(frequency, local[(local >= lower) & (local <= upper)]))
    estimate = calculate_thermal_gas(
        detector_config=path, frequency_hz=frequency, runtime_detector=detector
    )
    # The estimator may include a reporting target outside the plotting band.
    indices = np.searchsorted(estimate.frequency_hz, frequency)
    spectra = {
        "Quantum": get_noise_psd(frequency, noise_config=noise, detector_config=detector),
        **{
            name: estimate.spectra_psd[name][indices]
            for name in (*THERMAL_COMPONENTS, "ThermalTotal", "GasScattering",
                         "GasDamping", "ResidualGas")
        },
    }
    spectra["Total"] = spectra["Quantum"] + spectra["ThermalTotal"] + spectra["ResidualGas"]
    for name, psd in spectra.items():
        if psd.shape != frequency.shape or not np.all(np.isfinite(psd) & (psd >= 0)):
            raise ArithmeticError(f"{name} returned invalid one-sided noise PSDs.")
    return NoiseCurves(frequency, spectra, detector, noise, (float(lower), float(upper)))


def plot_noise_curves(curves: NoiseCurves, output_path=DEFAULT_OUTPUT, *, show=False) -> tuple[Path, Path]:
    """Save a paper-style PNG and vector PDF of totals and component curves."""
    cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "ghe-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = curves.frequency_hz
    detector = curves.detector_config
    quantum_model = normalize_noise_model(curves.noise_config.model)
    quantum_label = ("Analytic quantum, detuned SR"
                     if quantum_model == DETUNED_SIGNAL_RECYCLING_NOISE_MODEL
                     else "Analytic quantum, no SR")
    output_path = Path(output_path)
    png_path, pdf_path = output_path.with_suffix(".png"), output_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
        budget, components = axes
        for name, label, color, style, width in (
            ("Quantum", quantum_label, "#009E73", "-.", 1.45),
            ("ThermalTotal", "Thermal total (GWINC)", "#D55E00", "--", 1.4),
            ("ResidualGas", "Residual gas total (GWINC)", "#CC79A7", ":", 1.7),
            ("Total", "Quantum + thermal + gas", "black", "-", 2.1),
        ):
            psd = curves.spectra_psd[name]
            if np.any(psd > 0):
                # Quantum often dominates the sum. Draw its dash-dot line over
                # the wider total so both remain visible where they coincide.
                zorder = 4 if name == "Quantum" else 3 if name == "Total" else 2
                budget.loglog(frequency, np.sqrt(np.where(psd > 0, psd, np.nan)),
                              color=color, linestyle=style, linewidth=width,
                              label=label, zorder=zorder)
        for name, (label, color, style) in COMPONENT_STYLES.items():
            psd = curves.spectra_psd[name]
            if np.any(psd > 0):
                components.loglog(frequency, np.sqrt(np.where(psd > 0, psd, np.nan)),
                                  color=color, linestyle=style, linewidth=1.25, label=label)

        budget.set_title("(a) SNR noise budget", loc="left", fontsize=8.5)
        components.set_title("(b) Thermal and residual-gas components", loc="left", fontsize=8.5)
        budget.text(
            1.0, 1.04,
            rf"$L={detector.length / 1000:g}\,\mathrm{{km}}$, "
            rf"$T_{{\mathrm{{SRM}}}}={detector.T_SRM:g}$; "
            rf"ideal FD {curves.noise_config.effective_squeeze_db(detector):g} dB",
            transform=budget.transAxes, ha="right", va="bottom", fontsize=7.5,
        )
        target = curves.target_frequency_hz
        if curves.frequency_band_hz[0] <= target <= curves.frequency_band_hz[1]:
            target_index = int(np.flatnonzero(frequency == target)[0])
            target_asd = np.sqrt(curves.spectra_psd["Total"][target_index])
            budget.plot(target, target_asd, marker="o", markersize=4.2,
                        markerfacecolor="white", markeredgecolor="black",
                        markeredgewidth=1.0, zorder=5)
            budget.annotate(
                rf"$f_0={target:g}\,\mathrm{{Hz}}$", xy=(target, target_asd),
                xytext=(-10, 12), textcoords="offset points", ha="right", va="bottom",
                fontsize=7.2, color="0.2",
                arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.7},
            )
            for axis in axes:
                axis.axvline(target, color="0.55", linestyle=":", linewidth=0.9)
        for axis in axes:
            axis.set_xlim(*curves.frequency_band_hz)
            axis.set_ylabel(r"Strain ASD $\sqrt{S_h(f)}$ [$\mathrm{Hz}^{-1/2}$]")
            axis.grid(which="major", color="0.88", linewidth=0.65)
            axis.grid(which="minor", color="0.94", linewidth=0.4)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(which="both", direction="out", length=3.2)
            axis.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
                        loc="upper right", ncol=2, handlelength=3.0)
        components.set_xlabel(r"Frequency $f$ [Hz]")
        fig.subplots_adjust(left=0.12, right=0.99, bottom=0.09, top=0.95, hspace=0.27)
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
        if show:
            plt.show()
        plt.close(fig)
    return png_path, pdf_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output filename; matching PNG and PDF are saved.")
    parser.add_argument("--detector-config", type=Path, default=DETECTOR_CONFIG_FILE)
    parser.add_argument("--freq-min", type=float, default=None,
                        help="Minimum frequency; default is Noise.FrequencyBand.Minimum.")
    parser.add_argument("--freq-max", type=float, default=None,
                        help="Maximum frequency; default is Noise.FrequencyBand.Maximum.")
    parser.add_argument("--points", type=int, default=1000,
                        help="Logarithmic grid size, plus samples resolving the source-frequency dip.")
    parser.add_argument("--noise-model", default=None,
                        help="Analytic quantum model; default is Noise.Model in source.yaml.")
    parser.add_argument("--squeeze-db", type=float, default=None,
                        help="One-run quantum squeezing override; thermal/gas curves are unchanged.")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively.")
    args = parser.parse_args(argv)
    noise = NoiseConfig(squeeze_db=args.squeeze_db)
    if args.noise_model is not None:
        from dataclasses import replace
        noise = replace(noise, model=args.noise_model)
    try:
        curves = calculate_noise_curves(
            detector_path=args.detector_config, noise_config=noise,
            freq_min_hz=args.freq_min, freq_max_hz=args.freq_max, points=args.points,
        )
        paths = plot_noise_curves(curves, args.output, show=args.show)
    except (ConfigFileError, ValueError, ModuleNotFoundError, ArithmeticError) as exc:
        parser.exit(2, f"Noise-budget plot failed: {exc}\n")
    print(f"Analytic quantum model: {normalize_noise_model(curves.noise_config.model)}")
    print(f"Detector arm length: {curves.detector_config.length:g} m")
    target_indices = np.flatnonzero(curves.frequency_hz == curves.target_frequency_hz)
    if target_indices.size:
        print(f"Strain ASD at {curves.target_frequency_hz:g} Hz [Hz^(-1/2)]:")
        for name, values in curves.spectra_psd.items():
            print(f"  {name:24s} {np.sqrt(values[target_indices[0]]):.8e}")
    for path in paths:
        print(f"Saved figure: {path.resolve()}")


if __name__ == "__main__":
    main()
