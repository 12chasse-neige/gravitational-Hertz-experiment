from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import (
    BEST_POSITION_FILE,
    PAPER_FIGURES_DIR,
    ExperimentConfig,
    build_time_axis,
)
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
    t: float = 0.0,
    theta_src: Optional[float] = None,
    phi_src: Optional[float] = None,
    theta_rot: Optional[float] = None,
    phi_rot: Optional[float] = None,
    R: Optional[float] = None,
    config: Optional[ExperimentConfig] = None,
) -> float:
    d1, d2, d3, d4 = _get_best_position_defaults()
    return _calculate_metric_response(
        t,
        d1 if theta_src is None else theta_src,
        d2 if phi_src is None else phi_src,
        d3 if theta_rot is None else theta_rot,
        d4 if phi_rot is None else phi_rot,
        config=config,
        R=R,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=(
            "example: python metricCalculate.py -t 0.01 "
            "-ts 3.1 -ps 0.0 -tr 1.57 -pr 0.0"
        )
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=0.0,
        help="current time in seconds (default: 0)",
    )
    parser.add_argument(
        "-ts",
        "--thetasource",
        type=float,
        default=None,
        help=(
            "polar angle (detector frame) of the vector from detector toward the "
            "source (default: cached best position)"
        ),
    )
    parser.add_argument(
        "-ps",
        "--phisource",
        type=float,
        default=None,
        help=(
            "azimuthal angle (detector frame) of the vector from detector toward "
            "the source (default: cached best position)"
        ),
    )
    parser.add_argument(
        "-tr",
        "--thetarotation",
        type=float,
        default=None,
        help=(
            "polar angle (detector frame) of the rotor symmetry axis (body +z; "
            "default: cached best position)"
        ),
    )
    parser.add_argument(
        "-pr",
        "--phirotation",
        type=float,
        default=None,
        help=(
            "azimuthal angle (detector frame) of the rotor symmetry axis (body +z; "
            "default: cached best position)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed output")
    parser.add_argument("-o", "--output", type=str, default=None, help="path for the output file")
    parser.add_argument(
        "-R",
        "--distance",
        type=float,
        default=None,
        help="distance from source to detector in meters (overrides default R)",
    )
    return parser.parse_args()


def plot_single_source_signal(
    time_s: np.ndarray,
    response: np.ndarray,
    *,
    output_path: Path,
) -> None:
    """Plot the single-source response using the paper figure style."""

    time_s = np.asarray(time_s, dtype=float)
    response = np.asarray(response, dtype=float)
    if time_s.ndim != 1 or response.ndim != 1 or time_s.shape != response.shape:
        raise ValueError("time_s and response must be one-dimensional arrays of equal length")
    if time_s.size < 2 or not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(response)):
        raise ValueError("time_s and response must contain at least two finite samples")

    max_response = float(np.max(np.abs(response)))
    response_exponent = 0 if max_response == 0.0 else int(np.floor(np.log10(max_response)))
    scaled_response = response / 10.0**response_exponent

    matplotlib_cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "ghe-matplotlib-cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(paper_style):
        # Match the 7.2-inch canvas used by the paper's other full-width
        # figures so that labels and strokes have the same rendered scale.
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.plot(time_s * 1e3, scaled_response, color="#0072B2", linewidth=1.45)
        ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.9, zorder=1)
        ax.set_xlabel(r"Time $t$ [ms]")
        ax.set_ylabel(
            rf"Detector response $h_{{\mathrm{{det}}}}$ [$10^{{{response_exponent}}}$]"
        )
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(which="major", color="0.88", linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.2)

        fig.subplots_adjust(left=0.12, right=0.99, bottom=0.16, top=0.98)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
        vector_output_path = output_path.with_suffix(".pdf")
        if vector_output_path != output_path:
            fig.savefig(vector_output_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def signal_test(
    *,
    arm_length_m: float | None = None,
    output_path: Path = PAPER_FIGURES_DIR / "Signal.png",
) -> None:
    t = build_time_axis()
    config = ExperimentConfig() if arm_length_m is None else ExperimentConfig(L=float(arm_length_m))
    h_values = np.array(
        [calculate_metric_response(ti, config=config) for ti in t],
        dtype=float,
    )
    plot_single_source_signal(t, h_values, output_path=output_path)


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        print(f"Begin processing, current time: {args.time} s")
        if args.output:
            print(f"Output file: {args.output}")

    result = calculate_metric_response(
        args.time,
        args.thetasource,
        args.phisource,
        args.thetarotation,
        args.phirotation,
        R=args.distance,
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
