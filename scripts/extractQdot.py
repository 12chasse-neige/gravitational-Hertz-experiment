"""Extract Qdot with a center-retarded reference and modeled-term subtraction.

One sinusoid has two quadratures, not five independently identifiable derivative
terms. Phase-sensitive Qdot extraction is conditional on subtracting the other
four modeled terms. Report the raw reference projection separately, calculate
coherent Qdot/total SNR, and compare their reception-time strain curves.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ghe.artifacts import file_digest, model_metadata, validate_metadata
from ghe.config import DetectorConfig, NoiseConfig, SamplingConfig, SourceConfig
from ghe.detector_response import curvature_response_components
from ghe.finite_distance import (
    TIDAL_COMPONENT_NAMES,
    rotor_quadrupole_phasor,
    validate_rotor,
)
from ghe.geometry import spherical_unit_vector
from ghe.metric import calculate_response_phasor
from ghe.noise import get_noise_psd
from ghe.optimization import load_best_geometry, optimize_best_geometry
from ghe.paths import DETECTOR_CONFIG_FILE, RUNS_DIR, SOURCE_CONFIG_FILE, YEAR_SECONDS
from ghe.quadrature import project_quadratures
from ghe.signal import synthesize_signal
from ghe.thermal_gas import calculate_thermal_gas


def phasor_record(value):
    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "amplitude": float(abs(value)),
        "phase_rad": float(np.angle(value)),
    }


def fit_record(fit):
    return {
        "phasor": phasor_record(fit.phasor),
        "cosine": fit.cosine,
        "sine": fit.sine,
        "dc_offset": fit.offset,
        "residual_rms": fit.residual_rms,
        "method": fit.method,
        "condition_number": fit.condition_number,
        "orthogonality_error": fit.orthogonality_error,
        "reference_phase_rad": fit.reference_phase_rad,
    }


def rms(values):
    """Scale before squaring, since physical strains can be extremely small."""
    scale = float(np.max(np.abs(values))) or 1.0
    return float(np.linalg.norm(values / scale) / np.sqrt(values.size) * scale)


def analyze_single_source(
    time_s, signal=None, *, angles, config=None, rotor_delay_rad=0.0
):
    """Return a report and time-series arrays; no files or shared caches changed.

    If signal is None, synthesize the independent production total response.
    Otherwise treat the supplied values as observations at the original times.
    Geometry/configuration/rotor timing are assumed known. Project h(t) and
    h(t)-remaining_model on cos(psi), sin(psi), and DC, with
    psi=Omega*(t-R/c)-2*rotor_delay_rad. The latter fit estimates Qdot only
    conditional on the supplied model for the other four terms.
    """
    cfg = config or SourceConfig()
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (4,) or not np.all(np.isfinite(angles)):
        raise ValueError("Require four finite source/rotor angles in radians")
    if not np.isfinite(rotor_delay_rad):
        raise ValueError("Rotor delay must be finite")
    # Validate times (and external data) before the more expensive field model.
    if np.iscomplexobj(time_s):
        raise ValueError("Time samples must be real")
    times = np.asarray(time_s, dtype=float)
    project_quadratures(
        times,
        np.zeros_like(times) if signal is None else signal,
        cfg.gw_angular_frequency,
    )
    moment = rotor_quadrupole_phasor(*angles[2:], config=cfg)
    source = cfg.R * spherical_unit_vector(*angles[:2])
    components = curvature_response_components(
        source,
        moment,
        cfg.gw_angular_frequency,
        arm_length=cfg.L,
        G=cfg.G,
        c=cfg.c,
        source_radius=np.hypot(cfg.D / 2, cfg.H / 2),
    )
    phase = np.exp(2j * rotor_delay_rad)
    phasors = {name: result.phasor * phase for name, result in components.items()}
    total = calculate_response_phasor(*angles, config=cfg) * phase
    component_sum = sum(phasors.values())
    # Scale the check by individual arm integrals, not a potentially nulled total.
    integration_scale = sum(result.integration_scale for result in components.values())
    recombination_error = abs(total - component_sum)
    if recombination_error > 1e-10 * max(integration_scale, np.finfo(float).tiny):
        raise ArithmeticError(
            "Derivative components do not reproduce the production response"
        )
    model_total = synthesize_signal(total, times, cfg)
    observed = model_total if signal is None else np.asarray(signal, dtype=float)
    fits = {"signal": project_quadratures(times, observed, cfg.gw_angular_frequency)}
    component_signals = {
        name: synthesize_signal(value, times, cfg) for name, value in phasors.items()
    }
    fits.update(
        {
            name: project_quadratures(times, values, cfg.gw_angular_frequency)
            for name, values in component_signals.items()
        }
    )
    fit = fits["signal"]
    # Sum the other four terms directly, avoiding subtractive cancellation.
    remaining = synthesize_signal(
        sum(value for name, value in phasors.items() if name != "Qdot"), times, cfg
    )
    # The detector center is the Michelson vertex at reception-time zero.
    # Rotate only the projection basis; the response already includes retardation.
    reference_phase = cfg.gw_angular_frequency * cfg.R / cfg.c + 2 * rotor_delay_rad
    reference_argument = cfg.gw_angular_frequency * times - reference_phase
    reference_cosine = np.cos(reference_argument)
    reference_sine = np.sin(reference_argument)
    qdot_after_subtraction = observed - remaining
    retarded_fits = {
        name: project_quadratures(
            times, values, cfg.gw_angular_frequency, reference_phase_rad=reference_phase
        )
        for name, values in {
            "signal": observed,
            "remaining": remaining,
            "qdot_model": component_signals["Qdot"],
            "qdot_extracted": qdot_after_subtraction,
        }.items()
    }
    extracted = retarded_fits["qdot_extracted"]
    # Exclude the nuisance DC offset from the monochromatic Qdot waveform/SNR.
    qdot_extracted = (
        extracted.cosine * reference_cosine + extracted.sine * reference_sine
    )
    arrays = {
        "time_s": times,
        "signal": observed,
        "model_total": model_total,
        "qdot_model": component_signals["Qdot"],
        "remaining_model": remaining,
        "fitted_signal": fit.fitted,
        "cosine_quadrature": fit.cosine * np.cos(cfg.gw_angular_frequency * times),
        "sine_quadrature": fit.sine * np.sin(cfg.gw_angular_frequency * times),
        "fit_residual": observed - fit.fitted,
        "model_residual": observed - model_total,
        "reference_cosine": reference_cosine,
        "reference_sine": reference_sine,
        "reference_qdot": -reference_sine,
        "retarded_cosine_quadrature": retarded_fits["signal"].cosine * reference_cosine,
        "retarded_sine_quadrature": retarded_fits["signal"].sine * reference_sine,
        "qdot_after_subtraction": qdot_after_subtraction,
        "qdot_extracted": qdot_extracted,
        "qdot_fit_residual": qdot_after_subtraction - extracted.fitted,
    }
    for name in TIDAL_COMPONENT_NAMES:
        arrays[f"component_{name}"] = component_signals[name]
        arrays[f"fitted_{name}"] = fits[name].fitted
    denominator = abs(total)
    report = {
        "metadata": model_metadata(cfg),
        "interpretation": {
            "qdot_model": "Model-predicted explicit Qdot(u)/(c r^4) contribution, integrated through the finite-arm detector response.",
            "signal_projection": "Recovered total-signal quadratures; same-frequency physical contributions cannot be separated by orthogonality alone.",
            "qdot_extraction": "Fit h(t) minus the other four modeled terms using the center-retarded sine/cosine reference; Qdot recovery assumes those terms are known exactly.",
            "assumptions": "One monochromatic rotor; source configuration, geometry, phase delay and timestamp reference are known; ideal free-mass Michelson.",
        },
        "signal_origin": "generated" if signal is None else "loaded",
        "geometry": {
            "angles_rad": angles.tolist(),
            "distance_m": cfg.R,
            "rotor_delay_rad": float(rotor_delay_rad),
        },
        "sampling": {
            "num_samples": times.size,
            "first_time_s": float(times[0]),
            "last_time_s": float(times[-1]),
            "angular_frequency_rad_s": cfg.gw_angular_frequency,
        },
        "modeled_total_phasor": phasor_record(total),
        "component_sum_phasor": phasor_record(component_sum),
        "recombination_absolute_error": float(recombination_error),
        "recombination_error_over_integration_scale": float(
            recombination_error / integration_scale
        )
        if integration_scale
        else None,
        "signal_projection": fit_record(fit),
        "retarded_reference": {
            "location": "Detector center: Michelson vertex, x=y=z=0",
            "propagation_delay_s": float(cfg.R / cfg.c),
            "rotor_delay_s": float(rotor_delay_rad / cfg.omega),
            "phase_rad": float(reference_phase),
            "argument": "psi = Omega*(time_s - R/c) - 2*rotor_delay_rad",
            "q_reference": "cos(psi)",
            "qdot_reference": "-sin(psi), normalized time derivative of cos(psi)",
            "phasor_convention": "H = (cosine + i*sine)*exp(i*reference_phase_rad)",
        },
        "qdot_extraction": {
            "projection": fit_record(extracted),
            "raw_total_projection": fit_record(retarded_fits["signal"]),
            "remaining_model_projection": fit_record(retarded_fits["remaining"]),
            "modeled_qdot_projection": fit_record(retarded_fits["qdot_model"]),
            "raw_derivative_coefficient": -retarded_fits["signal"].sine,
            "removed_derivative_coefficient": -retarded_fits["remaining"].sine,
            "extracted_derivative_coefficient": -extracted.sine,
            "model_recovery_phasor_absolute_error": float(
                abs(extracted.phasor - phasors["Qdot"])
            ),
        },
        "components": {
            name: {
                "modeled_phasor": phasor_record(phasors[name]),
                "projection": fit_record(fits[name]),
                "integration_order": components[name].order,
                "integration_absolute_error": components[name].absolute_error,
                "integration_scale": components[name].integration_scale,
            }
            for name in TIDAL_COMPONENT_NAMES
        },
        "input_model_agreement": {
            "residual_rms": rms(observed - model_total),
            "phasor_absolute_error": float(abs(fit.phasor - total)),
            "phasor_relative_error": float(abs(fit.phasor - total) / denominator)
            if denominator
            else None,
        },
    }
    return report, arrays


def calculate_analysis_snr(
    report,
    *,
    config=None,
    observation_time_s=YEAR_SECONDS,
    noise_config=None,
    noise_budget="quantum-thermal-gas",
):
    """Coherent monochromatic SNR with one-sided PSD and peak strain amplitudes.

    The observation time is independent of the short record used for fitting
    and plotting. Qdot SNR assumes exact subtraction of the other modeled terms.
    """
    cfg = config or SourceConfig()
    noise = noise_config or NoiseConfig()
    validate_metadata(report["metadata"], cfg)
    if not np.isfinite(observation_time_s) or observation_time_s <= 0:
        raise ValueError("Observation time must be finite and positive")
    if noise_budget not in ("quantum", "quantum-thermal-gas"):
        raise ValueError("Noise budget must be quantum or quantum-thermal-gas")
    frequency = cfg.gw_frequency_hz
    if not noise.min_frequency_hz <= frequency <= noise.max_frequency_hz:
        raise ValueError("Signal frequency is outside the configured SNR band")
    detector = DetectorConfig.from_yaml(DETECTOR_CONFIG_FILE).with_source(cfg)
    quantum = float(
        get_noise_psd([frequency], noise_config=noise, detector_config=detector)[0]
    )
    noise_components = {"quantum": quantum}
    thermal_metadata = None
    if noise_budget == "quantum-thermal-gas":
        if not np.isclose(
            frequency, SourceConfig().gw_frequency_hz, rtol=1e-12, atol=0
        ):
            raise ValueError(
                "Source frequency must match the active source YAML for thermal/gas tuning"
            )
        thermal_gas = calculate_thermal_gas(
            detector_config=DETECTOR_CONFIG_FILE,
            arm_length_m=cfg.L,
            frequency_hz=[frequency],
        )
        index = int(np.flatnonzero(thermal_gas.frequency_hz == frequency)[0])
        noise_components.update(
            thermal=float(thermal_gas.spectra_psd["ThermalTotal"][index]),
            residual_gas=float(thermal_gas.spectra_psd["ResidualGas"][index]),
        )
        thermal_metadata = thermal_gas.metadata
    psd = sum(noise_components.values())
    if (
        not np.all(np.isfinite(list(noise_components.values())))
        or min(noise_components.values()) < 0
        or not np.isfinite(psd)
        or psd <= 0
    ):
        raise ArithmeticError("Noise calculation returned invalid one-sided PSDs")
    amplitude_uncertainty = float(np.sqrt(psd / observation_time_s))
    total_amplitude = report["signal_projection"]["phasor"]["amplitude"]
    extraction = report["qdot_extraction"]
    qdot_amplitude = extraction["projection"]["phasor"]["amplitude"]
    model_amplitude = report["components"]["Qdot"]["modeled_phasor"]["amplitude"]
    return {
        "frequency_hz": float(frequency),
        "observation_time_s": float(observation_time_s),
        "observation_years": float(observation_time_s / YEAR_SECONDS),
        "noise_budget": noise_budget,
        "noise_components_psd_per_hz": noise_components,
        "one_sided_psd_per_hz": float(psd),
        "strain_asd_per_sqrt_hz": float(np.sqrt(psd)),
        "quadrature_amplitude_uncertainty": amplitude_uncertainty,
        "formula": "rho = peak_strain_amplitude * sqrt(observation_time_s / S_h(f_gw))",
        "rho_total": float(total_amplitude / amplitude_uncertainty),
        "rho_qdot": float(qdot_amplitude / amplitude_uncertainty),
        "rho_qdot_model": float(model_amplitude / amplitude_uncertainty),
        "rho_qdot_derivative_quadrature": float(
            abs(extraction["extracted_derivative_coefficient"]) / amplitude_uncertainty
        ),
        "rho_raw_derivative_quadrature": float(
            abs(extraction["raw_derivative_coefficient"]) / amplitude_uncertainty
        ),
        "detector_config": asdict(detector),
        "noise_config": asdict(noise),
        "configuration_sha256": {
            str(path.relative_to(SOURCE_CONFIG_FILE.parent.parent)): file_digest(path)
            for path in (SOURCE_CONFIG_FILE, DETECTOR_CONFIG_FILE)
        },
        "thermal_gas_metadata": thermal_metadata,
        "assumptions": [
            "One monochromatic rotor with known geometry, phase and timestamp reference.",
            "Exact subtraction of the other four modeled curvature contributions; no subtraction-error allowance.",
            "Stationary noise and coherent phase tracking for the requested observation time.",
            "Ideal free-mass strain-noise calibration proxy, using the selected noise budget.",
        ],
    }


def plot_comparison(arrays, output, report=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    groups = (
        (
            ("signal", "Total h(t)"),
            ("qdot_extracted", "Extracted Qdot contribution"),
        ),
        (
            ("qdot_extracted", "Extracted Qdot contribution"),
            ("qdot_model", "Modeled Qdot contribution"),
        ),
    )
    colors = {"signal": "#245b87", "qdot_extracted": "#c75b29", "qdot_model": "#259278"}
    for ax, series in zip(axes[:2], groups):
        peak = max(float(np.max(np.abs(arrays[key]))) for key, _ in series)
        exponent = int(np.floor(np.log10(peak))) if peak else 0
        for index, (key, label) in enumerate(series):
            ax.plot(
                arrays["time_s"],
                arrays[key] / 10.0**exponent,
                label=label,
                linestyle=("-", "--", ":")[index],
                color=colors[key],
                linewidth=1.8,
            )
        ax.set_ylabel(f"Strain / $10^{{{exponent}}}$")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
    for key, label in (
        ("signal", "Total h(t)"),
        ("qdot_extracted", "Extracted Qdot"),
    ):
        amplitude = float(np.max(np.abs(arrays[key]))) or 1.0
        axes[2].plot(
            arrays["time_s"],
            arrays[key] / amplitude,
            label=label,
            color=colors[key],
            linewidth=1.8,
        )
    axes[2].plot(
        arrays["time_s"],
        arrays["reference_qdot"],
        label=r"Center-retarded derivative reference: $-\sin\psi$",
        color="#666666",
        linestyle=":",
        linewidth=1.4,
    )
    axes[2].set_ylabel("Normalized strain / reference")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(alpha=0.25)
    axes[0].set_title(
        "Total strain and Qdot extracted with a center-retarded reference"
    )
    axes[1].set_title(
        "Qdot detail after subtraction of the other modeled terms", fontsize=11
    )
    axes[2].set_title(
        "Relative waveform phases (each strain curve normalized separately)",
        fontsize=11,
    )
    if report is not None and "snr" in report:
        snr = report["snr"]
        axes[0].text(
            0.02,
            0.05,
            f"Coherent SNR ({snr['observation_years']:g} yr): "
            f"total {snr['rho_total']:.3g}, Qdot {snr['rho_qdot']:.3g}",
            transform=axes[0].transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    axes[-1].set_xlabel("Reception time [s]")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_analysis(report, arrays, output_dir):
    """Write reproducible, phase-preserving outputs without touching data/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if "snr" in report:
        (output_dir / "snr.json").write_text(
            json.dumps(report["snr"], indent=2, allow_nan=False) + "\n"
        )
    metadata = {
        **report["metadata"],
        "geometry": report["geometry"],
        "kind": "single_source_qdot_extraction",
        "retarded_reference": report["retarded_reference"],
    }
    np.savez_compressed(
        output_dir / "signals.npz",
        **arrays,
        metadata=np.array(json.dumps(metadata, allow_nan=False)),
    )
    np.savetxt(
        output_dir / "signals.csv",
        np.column_stack(list(arrays.values())),
        delimiter=",",
        header=",".join(arrays),
        comments="",
    )
    plot_comparison(arrays, output_dir / "comparison.png", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angles",
        type=float,
        nargs=4,
        metavar=("THETA_SRC", "PHI_SRC", "THETA_ROT", "PHI_ROT"),
        help="Detector-frame angles [rad]; required for loaded signals",
    )
    parser.add_argument(
        "--distance",
        type=float,
        help="COM-to-vertex distance [m]; default uses current R/L",
    )
    parser.add_argument(
        "--rotor-delay-rad",
        type=float,
        default=0.0,
        help="Mechanical phase delay delta [rad], multiplying H by exp(2i delta); default 0",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Generated record duration [s]; defaults to SamplingConfig",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        help="Generated record sample rate [Hz]; defaults to SamplingConfig",
    )
    parser.add_argument(
        "--signal-input",
        type=Path,
        help="Single-source NPZ with time_s and signal; timestamps keep their original time zero",
    )
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "qdot-extraction")
    parser.add_argument(
        "--observation-years",
        type=float,
        default=1.0,
        help="Coherent SNR observation time [years]; independent of --duration; default 1",
    )
    parser.add_argument(
        "--noise-budget",
        choices=("quantum", "quantum-thermal-gas"),
        default="quantum-thermal-gas",
        help="One-sided strain-noise PSD for SNR; default quantum + thermal + residual gas",
    )
    parser.add_argument(
        "--noise-model",
        help="Analytic quantum-noise model; default uses NoiseConfig",
    )
    args = parser.parse_args(argv)
    if args.signal_input and args.angles is None:
        parser.error("--signal-input requires explicit --angles")
    if args.signal_input and (
        args.duration is not None or args.sample_rate is not None
    ):
        parser.error("--duration and --sample-rate apply only to generated signals")
    try:
        cfg = SourceConfig()
        if args.distance is not None:
            cfg = replace(cfg, R=args.distance)
        validate_rotor(cfg)
        if not np.isfinite(args.rotor_delay_rad):
            raise ValueError("Rotor delay must be finite")
        if not np.isfinite(args.observation_years) or args.observation_years <= 0:
            raise ValueError("Observation time must be finite and positive")
        angles = args.angles
        if angles is None:
            geometry = load_best_geometry(config=cfg)
            if geometry is None:
                geometry = optimize_best_geometry(config=cfg)
            angles = geometry.angles
        signal = None
        if args.signal_input:
            with np.load(args.signal_input, allow_pickle=False) as payload:
                times, signal = payload["time_s"], payload["signal"]
                if "metadata" in payload:
                    metadata = json.loads(str(payload["metadata"]))
                    validate_metadata(metadata, cfg)
                    if "geometry" in metadata and metadata["geometry"] != {
                        "angles_rad": list(angles),
                        "distance_m": cfg.R,
                        "rotor_delay_rad": args.rotor_delay_rad,
                    }:
                        raise ValueError(
                            "Input signal geometry/rotor delay does not match the supplied model"
                        )
        else:
            sampling = SamplingConfig()
            if args.duration is not None:
                sampling = replace(sampling, duration_s=args.duration)
            if args.sample_rate is not None:
                sampling = replace(sampling, sample_rate_hz=args.sample_rate)
            if (
                not np.all(np.isfinite([sampling.duration_s, sampling.sample_rate_hz]))
                or min(sampling.duration_s, sampling.sample_rate_hz) <= 0
            ):
                raise ValueError("Duration and sample rate must be finite and positive")
            times = sampling.time_axis()
        report, arrays = analyze_single_source(
            times,
            signal,
            angles=angles,
            config=cfg,
            rotor_delay_rad=args.rotor_delay_rad,
        )
        report["snr"] = calculate_analysis_snr(
            report,
            config=cfg,
            observation_time_s=args.observation_years * YEAR_SECONDS,
            noise_config=NoiseConfig(model=args.noise_model)
            if args.noise_model
            else None,
            noise_budget=args.noise_budget,
        )
        if args.signal_input:
            report["signal_input"] = str(args.signal_input.resolve())
        save_analysis(report, arrays, args.output_dir)
    except (ValueError, ArithmeticError, OSError, KeyError) as exc:
        parser.error(str(exc))
    measured = report["signal_projection"]["phasor"]
    predicted = report["components"]["Qdot"]["modeled_phasor"]
    extracted = report["qdot_extraction"]["projection"]["phasor"]
    print(
        f"Total-signal quadratures: cosine={measured['real']:.12e}, sine={measured['imag']:.12e}"
    )
    print(f"Model-predicted Qdot peak amplitude: {predicted['amplitude']:.12e}")
    print(f"Extracted Qdot peak amplitude: {extracted['amplitude']:.12e}")
    print(
        f"Center-retarded reference: psi=Omega*(t-R/c)-2*delta; "
        f"R/c={report['retarded_reference']['propagation_delay_s']:.12e} s"
    )
    print(
        "Qdot extraction subtracts the other four modeled terms before fitting the retarded reference."
    )
    snr = report["snr"]
    print(
        f"Coherent SNR ({snr['observation_years']:g} yr, {snr['noise_budget']}): "
        f"Qdot={snr['rho_qdot']:.12e}, total={snr['rho_total']:.12e}"
    )
    print(f"Report, arrays and plot: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
