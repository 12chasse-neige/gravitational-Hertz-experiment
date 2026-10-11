"""Compare a model's Qdot contribution with measured trigonometric quadratures.

One sinusoid has two quadratures, not five independently identifiable derivative
terms. Qdot outputs are therefore MODEL PREDICTIONS, even when an input signal
is supplied. Projection outputs describe the supplied or generated total signal.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ghe.artifacts import model_metadata, validate_metadata
from ghe.config import SamplingConfig, SourceConfig
from ghe.detector_response import curvature_response_components
from ghe.finite_distance import (
    TIDAL_COMPONENT_NAMES,
    rotor_quadrupole_phasor,
    validate_rotor,
)
from ghe.geometry import spherical_unit_vector
from ghe.metric import calculate_response_phasor
from ghe.optimization import load_best_geometry, optimize_best_geometry
from ghe.paths import RUNS_DIR
from ghe.quadrature import project_quadratures
from ghe.signal import synthesize_signal


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
    Geometry/configuration/rotor timing are assumed known; no source parameters
    are inferred from the measured sine and cosine amplitudes.
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


def plot_comparison(arrays, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    groups = (
        (
            ("signal", "Input signal"),
            ("fitted_signal", "Quadrature fit"),
            ("model_total", "Model total"),
        ),
        (
            ("qdot_model", "Model-predicted Qdot"),
            ("fitted_Qdot", "Projection of Qdot model"),
        ),
        (
            ("remaining_model", "Other four modeled terms"),
            ("model_residual", "Input minus model total"),
        ),
    )
    for ax, series in zip(axes, groups):
        peak = max(float(np.max(np.abs(arrays[key]))) for key, _ in series)
        exponent = int(np.floor(np.log10(peak))) if peak else 0
        for index, (key, label) in enumerate(series):
            ax.plot(
                arrays["time_s"],
                arrays[key] / 10.0**exponent,
                label=label,
                linestyle=("-", "--", ":")[index],
            )
        ax.set_ylabel(f"Strain / $10^{{{exponent}}}$")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_title(
        "Single-source signal: modeled Qdot term and measured quadratures"
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
    metadata = {
        **report["metadata"],
        "geometry": report["geometry"],
        "kind": "single_source_qdot_decomposition",
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
    plot_comparison(arrays, output_dir / "comparison.png")


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
        if args.signal_input:
            report["signal_input"] = str(args.signal_input.resolve())
        save_analysis(report, arrays, args.output_dir)
    except (ValueError, ArithmeticError, OSError, KeyError) as exc:
        parser.error(str(exc))
    measured = report["signal_projection"]["phasor"]
    predicted = report["components"]["Qdot"]["modeled_phasor"]
    print(
        f"Total-signal quadratures: cosine={measured['real']:.12e}, sine={measured['imag']:.12e}"
    )
    print(f"Model-predicted Qdot peak amplitude: {predicted['amplitude']:.12e}")
    print(
        "Qdot is a model contribution; the total sine quadrature contains other same-frequency terms."
    )
    print(f"Report, arrays and plot: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
