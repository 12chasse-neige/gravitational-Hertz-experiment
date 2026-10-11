"""Find the best one-rotor Qdot SNR over a bounded arm-length range."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ghe.paths import DETECTOR_CONFIG_FILE, RUNS_DIR, YEAR_SECONDS
from ghe.qdot_scaling import sweep_qdot_arm_lengths


def save_results(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qdot_arm_length.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    fields = [
        "arm_length_m",
        "status",
        "reason",
        "source_distance_m",
        "frequency_hz",
        "qdot_peak_strain",
        "qdot_phasor_real",
        "qdot_phasor_imag",
        "quantum_psd",
        "thermal_psd",
        "gas_psd",
        "total_psd",
        "total_asd",
        "rho_qdot",
        "theta_src_rad",
        "phi_src_rad",
        "theta_rot_rad",
        "phi_rot_rad",
        "g1_g2",
        "itm_beam_radius_m",
        "etm_beam_radius_m",
        "itm_clipping_ppm",
        "etm_clipping_ppm",
        "integration_order",
        "integration_absolute_error",
        "minimum_source_arm_distance_m",
    ]
    rows = []
    for case in report["grid"] + report["refined_peaks"]:
        row = {key: case[key] for key in fields if key in case}
        row.update(status="valid", reason="")
        row.update(zip(fields[14:18], case["angles_rad"]))
        row.update({key: case["cavity"][key] for key in fields[18:23]})
        rows.append(row)
    rows += [
        {
            "arm_length_m": row["arm_length_m"],
            "status": "excluded",
            "reason": row["reason"],
        }
        for row in report["rejected_lengths"]
    ]
    with (output_dir / "qdot_arm_length.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["arm_length_m"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = report["grid"]
    length_km = np.array([row["arm_length_m"] / 1000 for row in grid])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].loglog(length_km, [row["rho_qdot"] for row in grid], color="#0072B2")
    best = report["best"]
    axes[0].scatter(
        best["arm_length_m"] / 1000,
        best["rho_qdot"],
        color="#D55E00",
        zorder=4,
        label=f"Best found: {best['arm_length_m']:g} m, SNR={best['rho_qdot']:.4g}",
    )
    axes[0].set_ylabel(r"$\rho_{\dot Q}$")
    axes[0].legend()
    axes[0].set_title(
        "One rotor: analytic quantum + thermal + residual-gas noise\n"
        f"R/L={report['distance_arm_ratio']:g}; fixed mirror curvatures; "
        f"{report['observation_time_s'] / YEAR_SECONDS:g} year coherent integration"
    )
    axes[1].loglog(
        length_km, [row["qdot_peak_strain"] for row in grid], color="#0072B2"
    )
    axes[1].set_ylabel(r"Peak $|H_{\dot Q}|$")
    for key, label, color in [
        ("quantum_psd", "Analytic quantum", "#009E73"),
        ("thermal_psd", "Thermal", "#D55E00"),
        ("gas_psd", "Residual gas", "#CC79A7"),
        ("total_psd", "PSD sum", "#000000"),
    ]:
        axes[2].loglog(
            length_km, np.sqrt([row[key] for row in grid]), label=label, color=color
        )
    axes[2].set_ylabel(r"Strain ASD [$\mathrm{Hz}^{-1/2}$]")
    axes[2].legend()
    axes[-1].set_xlabel("Arm length [km]")
    for ax in axes:
        ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "qdot_arm_length.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-length",
        type=float,
        default=100.0,
        help="Minimum arm length [m], default 100",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=59000.0,
        help="Maximum arm length [m], default 59000",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=121,
        help="Logarithmic sweep grid size, default 121",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs=4,
        help="Fix source/rotor angles [rad]; default optimizes Qdot at each L",
    )
    parser.add_argument("--observation-years", type=float, default=1.0)
    parser.add_argument("--detector-config", type=Path, default=DETECTOR_CONFIG_FILE)
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "qdot-arm-length")
    args = parser.parse_args(argv)
    if (
        not np.all(np.isfinite([args.min_length, args.max_length]))
        or not 0 < args.min_length < args.max_length
    ):
        parser.error("Require 0 < min-length < max-length, both finite")
    if args.points < 3:
        parser.error("At least three sweep points are required")
    lengths = np.geomspace(args.min_length, args.max_length, args.points)
    # Include the existing 40 km reference and the 30 km GWINC degeneracy so
    # both appear explicitly in the reproducible sweep report when in range.
    lengths = np.unique(
        np.append(
            lengths,
            [
                value
                for value in (30000, 40000)
                if args.min_length <= value <= args.max_length
            ],
        )
    )
    count = 0

    def progress(row):
        nonlocal count
        count += 1
        if count == 1 or count % 10 == 0:
            print(
                f"Evaluated {count} feasible lengths; L={row['arm_length_m']:.6g} m, "
                f"rho_Qdot={row['rho_qdot']:.6g}",
                flush=True,
            )

    try:
        report = sweep_qdot_arm_lengths(
            lengths,
            detector_config=args.detector_config,
            angles=args.angles,
            observation_time_s=args.observation_years * YEAR_SECONDS,
            progress=progress,
        )
        save_results(report, args.output_dir)
    except (ValueError, ArithmeticError, OSError) as exc:
        parser.error(str(exc))
    best = report["best"]
    print(
        f"Best found: L={best['arm_length_m']:.12g} m, R={best['source_distance_m']:.12g} m, "
        f"rho_Qdot={best['rho_qdot']:.12g}"
    )
    print(
        f"Location: {report['best_location']}; excluded {len(report['rejected_lengths'])} optical configurations"
    )
    if report["best_location"] == "lower_bound":
        print(
            "The maximum is at the chosen minimum length; this is not a finite interior optimum."
        )
    print(f"CSV, JSON and plot: {args.output_dir.resolve()}")
    return report


if __name__ == "__main__":
    main()
