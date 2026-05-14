from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from ghe.config import BEST_POSITION_FILE, DATA_DIR, ExperimentConfig
from ghe.geometry import (
    _get_perpendicular_axis,
    cartesian_to_spherical,
    rotate_reference_vector,
    spherical_to_cartesian,
    spherical_unit_vector,
)
from ghe.optimization import (
    BestGeometry,
    BestGeometry as RelativeGeometry,
    parse_best_position_text as _parse_best_position_text,
    save_best_geometry,
    solve_best_geometry,
)
from ghe.source_array.generation import (
    DEFAULT_APPROXIMATION_CHUNK_SIZE,
    ArrayContext,
    build_array_context,
    build_chunk,
    construct_source_array,
    iter_source_chunks,
    summary_lines,
    format_vector,
    write_source_array_csv,
    write_source_array_npz,
)
from ghe.source_array.layout import choose_lattice_dimensions, positions_for_index_range
from ghe.source_array.phase import get_signal_amplitude_and_phase, wrap_phase
from ghe.source_array.schema import SOURCE_ARRAY_DTYPE
from ghe.source_array.strategies import (
    approximate_chunk_center_parameters,
    exact_anchor_for_source_index,
    optimize_rotor_for_fixed_source,
)

_DEFAULT_OUTPUT_FILE = DATA_DIR / "source_array_distribution.csv"
_DEFAULT_NPZ_OUTPUT_FILE = DATA_DIR / "source_array_distribution.npz"
_DEFAULT_APPROXIMATION_CHUNK_SIZE = DEFAULT_APPROXIMATION_CHUNK_SIZE

_build_chunk = build_chunk
_positions_for_index_range = positions_for_index_range
_optimize_rotor_for_fixed_source = optimize_rotor_for_fixed_source
_exact_anchor_for_source_index = exact_anchor_for_source_index
_approximate_chunk_center_parameters = approximate_chunk_center_parameters
_format_vector = format_vector


def _write_best_position_file(
    angles: tuple[float, float, float, float],
    amplitude: float,
    path: Path = BEST_POSITION_FILE,
) -> None:
    save_best_geometry(BestGeometry(*angles, signal_amplitude=amplitude), path=path)


def _print_summary(context: ArrayContext, preview_rows=None) -> None:
    print("\n".join(summary_lines(context, preview_rows)))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a coherent source array around the best geometry from "
            "bestPosition.py and compute the phase offset for each source."
        )
    )
    parser.add_argument(
        "--num-sources",
        type=int,
        default=10_000_000,
        help="Total number of sources in the array. Default: 10000000.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=ExperimentConfig.D * 3 / 2,
        help="Center-to-center spacing of neighboring sources in meters. Default: source diameter D * 3/2.",
    )
    parser.add_argument(
        "--theta-array",
        type=float,
        default=None,
        help="Optional polar angle for the array center direction. Defaults to data/bestPosition.txt.",
    )
    parser.add_argument(
        "--phi-array",
        type=float,
        default=None,
        help="Optional azimuthal angle for the array center direction. Defaults to data/bestPosition.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_FILE,
        help=f"CSV output path. Default: {_DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--npz-output",
        type=Path,
        default=_DEFAULT_NPZ_OUTPUT_FILE,
        help=f"NPZ output path. Default: {_DEFAULT_NPZ_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "npz", "both"),
        default="csv",
        help="Output format. Default: csv.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of sources processed per chunk. Default: 100000.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of sources to print as a preview. Default: 5.",
    )
    parser.add_argument(
        "--recompute-best-position",
        action="store_true",
        help="Run the optimizer from bestPosition.py instead of only reusing data/bestPosition.txt.",
    )
    parser.add_argument(
        "--optimize-each-source",
        action="store_true",
        default=True,
        help="Run bestPosition optimization for each source individually. Enabled by default.",
    )
    parser.add_argument(
        "--no-optimize-each-source",
        dest="optimize_each_source",
        action="store_false",
        help="Use rigidly transported rotor axes instead of per-source optimization.",
    )
    parser.add_argument(
        "--chunk-center-approximation",
        action="store_true",
        help=(
            "Speed up generation by optimizing and phase-extracting only the center "
            "source of each approximation chunk, then approximating nearby sources."
        ),
    )
    parser.add_argument(
        "--approximation-chunk-size",
        type=int,
        default=_DEFAULT_APPROXIMATION_CHUNK_SIZE,
        help=(
            "Number of sources per chunk-center approximation group. "
            f"Default: {_DEFAULT_APPROXIMATION_CHUNK_SIZE}."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the geometry summary and preview rows without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    context = build_array_context(
        num_sources=args.num_sources,
        spacing=args.spacing,
        theta_array=args.theta_array,
        phi_array=args.phi_array,
        recompute_best_position=args.recompute_best_position,
        optimize_each_source=args.optimize_each_source,
        chunk_center_approximation=args.chunk_center_approximation,
        approximation_chunk_size=args.approximation_chunk_size,
    )

    preview_count = min(args.preview, args.num_sources)
    preview_rows = build_chunk(context, 0, preview_count) if preview_count > 0 else None
    print("\n".join(summary_lines(context, preview_rows)))

    if args.summary_only:
        return

    if args.format in ("csv", "both"):
        write_source_array_csv(
            output_path=args.output,
            num_sources=args.num_sources,
            theta_array=args.theta_array,
            phi_array=args.phi_array,
            spacing=args.spacing,
            recompute_best_position=False,
            optimize_each_source=args.optimize_each_source,
            chunk_center_approximation=args.chunk_center_approximation,
            approximation_chunk_size=args.approximation_chunk_size,
            chunk_size=args.chunk_size,
        )
        print(f"\nSaved source array table to {args.output}")

    if args.format in ("npz", "both"):
        write_source_array_npz(
            output_path=args.npz_output,
            num_sources=args.num_sources,
            theta_array=args.theta_array,
            phi_array=args.phi_array,
            spacing=args.spacing,
            recompute_best_position=False,
            optimize_each_source=args.optimize_each_source,
            chunk_center_approximation=args.chunk_center_approximation,
            approximation_chunk_size=args.approximation_chunk_size,
            chunk_size=args.chunk_size,
        )
        print(f"Saved source array NPZ to {args.npz_output}")


if __name__ == "__main__":
    main()
