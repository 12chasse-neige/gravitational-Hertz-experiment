"""Generate coherent rotor arrays with phases matched to the full response."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from ghe.config import SourceArrayConfig
from ghe.paths import DATA_DIR
from ghe.source_array.generation import (
    build_array_context,
    build_chunk,
    summary_lines,
    write_source_array_csv,
    write_source_array_npz,
)

_DEFAULT_OUTPUT_FILE = DATA_DIR / "source_array_distribution.csv"
_DEFAULT_NPZ_OUTPUT_FILE = DATA_DIR / "source_array_distribution.npz"
_SOURCE_ARRAY_DEFAULTS = SourceArrayConfig()
_DEFAULT_APPROXIMATION_CHUNK_SIZE = _SOURCE_ARRAY_DEFAULTS.approximation_chunk_size

def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a coherent source array around the optimized geometry "
            "and compute the full-response phase offset for each source."
        )
    )
    parser.add_argument(
        "--num-sources",
        type=int,
        default=_SOURCE_ARRAY_DEFAULTS.num_sources,
        help=(
            "Total number of sources in the array. "
            f"Default: {_SOURCE_ARRAY_DEFAULTS.num_sources}."
        ),
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=_SOURCE_ARRAY_DEFAULTS.spacing,
        help="Center-to-center source spacing in metres. Default: source diameter D * 3/2.",
    )
    parser.add_argument(
        "--theta-array",
        type=float,
        default=_SOURCE_ARRAY_DEFAULTS.theta_array,
        help="Optional polar angle for the array center direction. Defaults to data/bestPosition.txt.",
    )
    parser.add_argument(
        "--phi-array",
        type=float,
        default=_SOURCE_ARRAY_DEFAULTS.phi_array,
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
        default=_SOURCE_ARRAY_DEFAULTS.chunk_size,
        help=(
            "Number of sources processed per chunk. "
            f"Default: {_SOURCE_ARRAY_DEFAULTS.chunk_size}."
        ),
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
        default=_SOURCE_ARRAY_DEFAULTS.recompute_best_position,
        help="Recompute geometry with ghe.optimization instead of reusing data/bestPosition.txt.",
    )
    parser.add_argument(
        "--optimize-each-source",
        action="store_true",
        default=_SOURCE_ARRAY_DEFAULTS.optimize_each_source,
        help="Optimize each rotor orientation individually. Enabled by default.",
    )
    parser.add_argument(
        "--no-optimize-each-source",
        dest="optimize_each_source",
        action="store_false",
        help="Use rigidly transported rotor axes instead of per-source optimization.",
    )
    approximation_group = parser.add_mutually_exclusive_group()
    approximation_group.add_argument(
        "--chunk-center-approximation",
        dest="chunk_center_approximation",
        action="store_true",
        help=(
            "Optimize the rotor axis once per approximation chunk; still "
            "calculate the complete response phase for every source."
        ),
    )
    approximation_group.add_argument(
        "--no-chunk-center-approximation",
        dest="chunk_center_approximation",
        action="store_false",
        help="Disable chunk-center approximation and optimize each source exactly.",
    )
    parser.set_defaults(
        chunk_center_approximation=_SOURCE_ARRAY_DEFAULTS.chunk_center_approximation
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
    args = parser.parse_args(argv)
    if args.num_sources < 1 or args.chunk_size < 1 or args.approximation_chunk_size < 1:
        parser.error("Source count and chunk sizes must be positive.")
    if args.preview < 0:
        parser.error("--preview must be nonnegative.")
    return args


def main(argv=None) -> None:
    args = parse_arguments(argv)

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
