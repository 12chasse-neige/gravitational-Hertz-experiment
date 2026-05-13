from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import sys
from typing import Optional, Tuple

import numpy as np

from ghe.config import BEST_POSITION_FILE, IMAGES_DIR, ExperimentConfig, build_time_axis
from ghe.geometry import rotation_body_to_detector, spherical_unit_vector
from ghe.metric import (
    calculate_delta_t,
    calculate_delta_t_prime,
    calculate_metric_response as _calculate_metric_response,
    calculate_whole_tensor,
    get_hole_coordinate,
    get_metric_tensor_body_frame,
    project_to_tt_gauge_dynamic,
    second_derivative_of_tensor,
)
from ghe.optimization import FALLBACK_BEST_POSITION, parse_best_position_text

_BEST_POSITION_CACHE: Optional[Tuple[float, float, float, float]] = None


def _get_best_position_defaults() -> Tuple[float, float, float, float]:
    global _BEST_POSITION_CACHE
    if _BEST_POSITION_CACHE is not None:
        return _BEST_POSITION_CACHE
    if BEST_POSITION_FILE.is_file():
        try:
            parsed = parse_best_position_text(BEST_POSITION_FILE.read_text(encoding="utf-8"))
            if parsed is not None:
                _BEST_POSITION_CACHE = parsed
                return _BEST_POSITION_CACHE
        except OSError:
            pass
    _BEST_POSITION_CACHE = FALLBACK_BEST_POSITION
    return _BEST_POSITION_CACHE


def calculate_metric_response(
    t: float,
    theta_src: Optional[float] = None,
    phi_src: Optional[float] = None,
    theta_rot: Optional[float] = None,
    phi_rot: Optional[float] = None,
    R: Optional[float] = None,
) -> float:
    d1, d2, d3, d4 = _get_best_position_defaults()
    return _calculate_metric_response(
        t,
        d1 if theta_src is None else theta_src,
        d2 if phi_src is None else phi_src,
        d3 if theta_rot is None else theta_rot,
        d4 if phi_rot is None else phi_rot,
        R=R,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=(
            "example: python metricCalculate.py -t 0.01 "
            "-ts 3.1 -ps 0.0 -tr 1.57 -pr 0.0"
        )
    )
    parser.add_argument("-t", "--time", type=str, required=True, help="current time in seconds")
    parser.add_argument(
        "-ts",
        "--thetasource",
        type=str,
        required=True,
        help="polar angle (detector frame) of the vector from detector toward the source",
    )
    parser.add_argument(
        "-ps",
        "--phisource",
        type=str,
        required=True,
        help="azimuthal angle (detector frame) of the vector from detector toward the source",
    )
    parser.add_argument(
        "-tr",
        "--thetarotation",
        type=str,
        required=True,
        help="polar angle (detector frame) of the rotor symmetry axis (body +z)",
    )
    parser.add_argument(
        "-pr",
        "--phirotation",
        type=str,
        required=True,
        help="azimuthal angle (detector frame) of the rotor symmetry axis (body +z)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed output")
    parser.add_argument("-o", "--output", type=str, default=None, help="path for the output file")
    parser.add_argument(
        "-R",
        "--distance",
        type=str,
        default=None,
        help="distance from source to detector in meters (overrides default R)",
    )
    return parser.parse_args()


def signal_test() -> None:
    t = build_time_axis()
    h_values = np.array([calculate_metric_response(ti) for ti in t])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(t, h_values)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal [1]")
    plt.title("Input Signal Curve")
    plt.savefig(IMAGES_DIR / "Signal.png")


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        print(f"Begin processing, current time: {args.time} s")
        if args.output:
            print(f"Output file: {args.output}")

    result = calculate_metric_response(
        float(args.time),
        float(args.thetasource),
        float(args.phisource),
        float(args.thetarotation),
        float(args.phirotation),
        R=float(args.distance) if args.distance is not None else None,
    )
    print(result)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(str(result))
            print(f"\nResult saved at: {args.output}")
        except Exception as e:
            print(f"Error while saving file: {e}")

    sys.exit(0)
