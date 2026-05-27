from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ghe.config import (
    DATA_DIR,
    DetectorConfig,
    IMG_DIR,
    NoiseConfig,
    SamplingConfig,
    SourceConfig,
)
from ghe.metric import calculate_metric_response
from ghe.noise import get_detuned_signal_recycling_noise_psd
from ghe.optimization import (
    BestGeometry,
    FALLBACK_BEST_POSITION,
    get_signal_amplitude,
    load_best_geometry,
    scaled_spherical_function,
    scipy_gradient_descent,
    spherical_function,
)
from ghe.spectrum import calculate_spectrum
from scr.noiseAnalysis import calculate_snr_from_arrays
from scr.runSNR import parse_float_list


@dataclass(frozen=True)
class SweepResult:
    arm_length_m: float
    source_distance_m: float
    detuned_asd_per_sqrt_hz: float
    single_source_snr_year: float
    phi_sr_rad: float
    signal_amplitude: float
    theta_src: float
    phi_src: float
    theta_rot: float
    phi_rot: float


def source_omega_from_gw_frequency(gw_frequency_hz: float) -> float:
    """Return mechanical angular velocity for a target quadrupole GW frequency."""

    return float(np.pi * gw_frequency_hz)


def build_source_config(arm_length_m: float, gw_frequency_hz: float) -> SourceConfig:
    return SourceConfig(
        L=float(arm_length_m),
        omega=source_omega_from_gw_frequency(gw_frequency_hz),
    )


def build_detector_config(arm_length_m: float, gw_frequency_hz: float) -> DetectorConfig:
    return DetectorConfig(
        length=float(arm_length_m),
        resonance_frequency_hz=float(gw_frequency_hz),
        phi_SR=None,
    )


def geometry_from_angles(
    config: SourceConfig,
    angles: tuple[float, float, float, float],
) -> BestGeometry:
    amplitude = get_signal_amplitude(*angles, config=config)
    return BestGeometry(*angles, signal_amplitude=amplitude)


def optimize_geometry(config: SourceConfig) -> BestGeometry:
    def objective(
        theta_src: float,
        phi_src: float,
        theta_rot: float,
        phi_rot: float,
    ) -> float:
        return scaled_spherical_function(
            theta_src,
            phi_src,
            theta_rot,
            phi_rot,
            config=config,
        )

    theta_src, phi_src, theta_rot, phi_rot = scipy_gradient_descent(
        objective,
        1.0,
        0.0,
        1.0,
        0.0,
    )
    amplitude = spherical_function(theta_src, phi_src, theta_rot, phi_rot, config=config)
    return BestGeometry(theta_src, phi_src, theta_rot, phi_rot, signal_amplitude=amplitude)


def build_single_source_spectrum(
    config: SourceConfig,
    geometry: BestGeometry,
    sampling: SamplingConfig,
):
    time_axis = sampling.time_axis()
    signal = np.array(
        [
            calculate_metric_response(
                float(t),
                *geometry.angles,
                config=config,
            )
            for t in time_axis
        ],
        dtype=float,
    )
    return calculate_spectrum(signal, sampling=sampling)


def detuned_asd_at_frequency(
    frequency_hz: float,
    *,
    squeeze_db: float,
    detector_config: DetectorConfig,
) -> float:
    psd = get_detuned_signal_recycling_noise_psd(
        np.array([frequency_hz], dtype=float),
        squeeze_db=squeeze_db,
        config=detector_config,
    )
    return float(np.sqrt(psd[0]))


def run_sweep(
    arm_lengths_m: list[float],
    *,
    gw_frequency_hz: float,
    squeeze_db: float,
    sampling: SamplingConfig,
    optimize_each_length: bool,
    min_snr_frequency_hz: float,
    max_snr_frequency_hz: float,
) -> list[SweepResult]:
    results: list[SweepResult] = []
    cached = None if optimize_each_length else load_best_geometry()
    cached_angles = cached.angles if cached is not None else FALLBACK_BEST_POSITION
    noise_config = NoiseConfig(
        model="detuned_signal_recycling",
        squeeze_db=squeeze_db,
        min_frequency_hz=min_snr_frequency_hz,
        max_frequency_hz=max_snr_frequency_hz,
    )

    for arm_length_m in arm_lengths_m:
        source_config = build_source_config(arm_length_m, gw_frequency_hz)
        detector_config = build_detector_config(arm_length_m, gw_frequency_hz)
        geometry = (
            optimize_geometry(source_config)
            if optimize_each_length
            else geometry_from_angles(source_config, cached_angles)
        )
        spectrum = build_single_source_spectrum(source_config, geometry, sampling)
        snr_year = calculate_snr_from_arrays(
            spectrum.magnitude,
            spectrum.freqs,
            noise_config=noise_config,
            detector_config=detector_config,
            sampling_config=sampling,
            verbose=False,
        )
        asd = detuned_asd_at_frequency(
            gw_frequency_hz,
            squeeze_db=squeeze_db,
            detector_config=detector_config,
        )
        results.append(
            SweepResult(
                arm_length_m=float(arm_length_m),
                source_distance_m=float(source_config.R),
                detuned_asd_per_sqrt_hz=asd,
                single_source_snr_year=snr_year,
                phi_sr_rad=float(detector_config.phi_SR),
                signal_amplitude=float(geometry.signal_amplitude),
                theta_src=float(geometry.theta_src),
                phi_src=float(geometry.phi_src),
                theta_rot=float(geometry.theta_rot),
                phi_rot=float(geometry.phi_rot),
            )
        )
        print(
            f"L={arm_length_m:g} m, R={source_config.R:g} m, "
            f"ASD({gw_frequency_hz:g} Hz)={asd:.6e}, SNR_yr={snr_year:.6e}"
        )

    return results


def save_results_csv(results: list[SweepResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SweepResult.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: getattr(result, name) for name in fieldnames})


def maybe_log_scale(ax, x_values: np.ndarray, y_values: np.ndarray) -> None:
    if np.all(x_values > 0.0):
        ax.set_xscale("log")
    if np.all(y_values > 0.0):
        ax.set_yscale("log")


def plot_results(
    results: list[SweepResult],
    *,
    gw_frequency_hz: float,
    output_path: Path,
) -> None:
    matplotlib_cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "ghe-matplotlib-cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    import matplotlib.pyplot as plt

    lengths = np.array([result.arm_length_m for result in results], dtype=float)
    asd = np.array([result.detuned_asd_per_sqrt_hz for result in results], dtype=float)
    snr = np.array([result.single_source_snr_year for result in results], dtype=float)

    best_snr_index = int(np.nanargmax(snr))
    min_asd_index = int(np.nanargmin(asd))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(lengths, asd, marker="o", linewidth=2)
    axes[0].axvline(lengths[min_asd_index], color="tab:green", linestyle="--", alpha=0.6)
    axes[0].set_ylabel("ASD [1/sqrt(Hz)]")
    axes[0].set_title(f"Detuned Interferometer ASD at {gw_frequency_hz:g} Hz")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.35)
    maybe_log_scale(axes[0], lengths, asd)

    axes[1].plot(lengths, snr, marker="o", linewidth=2, color="tab:orange")
    axes[1].axvline(lengths[best_snr_index], color="tab:red", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Arm length L [m]")
    axes[1].set_ylabel("Single-source SNR, 1 year")
    axes[1].set_title("Single-Source Detection Scaling")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.35)
    maybe_log_scale(axes[1], lengths, snr)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    defaults = SamplingConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Sweep interferometer arm length, plotting detuned ASD at the source "
            "GW frequency and the corresponding single-source 1-year SNR."
        )
    )
    parser.add_argument(
        "--lengths",
        default="[500,10000,500]",
        help="Arm lengths in m: comma list or [start,stop,step]. Default: [500,10000,500].",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=600.0,
        help="Target GW frequency in Hz. Default: 600.",
    )
    parser.add_argument(
        "--squeeze-db",
        type=float,
        default=10.0,
        help="Squeezing level in dB. Default: 10.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=defaults.duration_s,
        help=f"Signal duration in seconds. Default: {defaults.duration_s:g}.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=defaults.sample_rate_hz,
        help=f"Sampling rate in Hz. Default: {defaults.sample_rate_hz:g}.",
    )
    parser.add_argument(
        "--snr-min-frequency",
        type=float,
        default=1.0,
        help="Minimum frequency included in SNR integration. Default: 1.",
    )
    parser.add_argument(
        "--snr-max-frequency",
        type=float,
        default=5000.0,
        help="Maximum frequency included in SNR integration. Default: 5000.",
    )
    parser.add_argument(
        "--optimize-geometry",
        action="store_true",
        help="Re-optimize the single-source geometry at every arm length.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / "Arm Length Scaling.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DATA_DIR / "arm_length_scaling.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    arm_lengths_m = parse_float_list(args.lengths)
    sampling = SamplingConfig(duration_s=args.duration, sample_rate_hz=args.sample_rate)

    results = run_sweep(
        arm_lengths_m,
        gw_frequency_hz=args.frequency,
        squeeze_db=args.squeeze_db,
        sampling=sampling,
        optimize_each_length=args.optimize_geometry,
        min_snr_frequency_hz=args.snr_min_frequency,
        max_snr_frequency_hz=args.snr_max_frequency,
    )
    save_results_csv(results, args.csv_output)
    plot_results(results, gw_frequency_hz=args.frequency, output_path=args.output)

    best_snr = max(results, key=lambda result: result.single_source_snr_year)
    min_asd = min(results, key=lambda result: result.detuned_asd_per_sqrt_hz)
    print(f"\nSaved CSV: {args.csv_output}")
    print(f"Saved figure: {args.output}")
    print(
        "Best single-source SNR: "
        f"L={best_snr.arm_length_m:g} m, SNR_yr={best_snr.single_source_snr_year:.6e}"
    )
    print(
        "Lowest detuned ASD: "
        f"L={min_asd.arm_length_m:g} m, ASD={min_asd.detuned_asd_per_sqrt_hz:.6e}"
    )


if __name__ == "__main__":
    main()
