"""Sweep coherent source-array size and measure departure from ideal SNR scaling.

Each requested array is generated independently so its lattice remains centered at
the reference source position. Results are written with enough precision to resolve
small deviations from ``SNR(N) = N * SNR(1)``.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from pathlib import Path

from ghe.config import DATA_DIR, SourceConfig
from ghe.signal import compute_phasor_sum
from ghe.snr import calculate_snr_from_phasor
from ghe.source_array.generation import build_array_context, iter_source_chunks


DEFAULT_ARRAY_SIZES = "1,10,100,1000,10000,100000"
DEFAULT_OUTPUT = DATA_DIR / "source_array_snr_scaling.csv"


def parse_array_sizes(raw: str) -> list[int]:
    """Parse a comma-separated, positive, strictly increasing list of sizes."""

    try:
        sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Array sizes must be comma-separated integers.") from exc

    if not sizes:
        raise argparse.ArgumentTypeError("At least one array size is required.")
    if any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("Every array size must be positive.")
    if sizes[0] != 1:
        raise argparse.ArgumentTypeError("The first array size must be 1 to define SNR_1.")
    if any(right <= left for left, right in zip(sizes, sizes[1:])):
        raise argparse.ArgumentTypeError("Array sizes must be unique and strictly increasing.")
    return sizes


def calculate_array_snr(
    num_sources: int,
    *,
    strategy: str,
    generation_chunk_size: int,
    approximation_chunk_size: int,
    spacing: float | None,
) -> tuple[float, float]:
    """Generate one centered array and return its one-year SNR and phasor magnitude."""

    optimize_each_source = strategy != "rigid"
    chunk_center_approximation = strategy == "chunk-center"
    context = build_array_context(
        num_sources=num_sources,
        spacing=spacing,
        optimize_each_source=optimize_each_source,
        chunk_center_approximation=chunk_center_approximation,
        approximation_chunk_size=approximation_chunk_size,
    )

    source_config = SourceConfig()
    phasor = complex(0.0, 0.0)
    for chunk in iter_source_chunks(context, chunk_size=generation_chunk_size):
        phasor += compute_phasor_sum(
            chunk,
            config=source_config,
            chunk_size=generation_chunk_size,
        )

    snr = calculate_snr_from_phasor(phasor, source_config.gw_frequency_hz)
    return snr, abs(phasor)


def sweep_array_snr(
    array_sizes: list[int],
    *,
    strategy: str,
    generation_chunk_size: int,
    approximation_chunk_size: int,
    spacing: float | None,
) -> list[dict[str, int | float]]:
    """Calculate numerical, ideal, and relative-deviation SNR values."""

    results: list[dict[str, int | float]] = []
    snr_1: float | None = None

    for num_sources in array_sizes:
        print(f"Calculating N={num_sources} with strategy={strategy} ...", flush=True)
        numerical_snr, phasor_magnitude = calculate_array_snr(
            num_sources,
            strategy=strategy,
            generation_chunk_size=generation_chunk_size,
            approximation_chunk_size=approximation_chunk_size,
            spacing=spacing,
        )
        if snr_1 is None:
            snr_1 = numerical_snr

        ideal_snr = num_sources * snr_1
        relative_deviation = (numerical_snr - ideal_snr) / ideal_snr
        results.append(
            {
                "num_sources": num_sources,
                "numerical_snr_year": numerical_snr,
                "ideal_snr_year": ideal_snr,
                "relative_deviation": relative_deviation,
                "phasor_magnitude": phasor_magnitude,
            }
        )
        print(
            f"  numerical={numerical_snr:.17e}  ideal={ideal_snr:.17e}  "
            f"deviation={relative_deviation:.17e}",
            flush=True,
        )

    return results


def write_results(rows: list[dict[str, int | float]], output_path: Path) -> None:
    """Write sweep results without truncating floating-point precision."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "num_sources",
        "numerical_snr_year",
        "ideal_snr_year",
        "relative_deviation",
        "phasor_magnitude",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "num_sources": row["num_sources"],
                    **{
                        name: format(float(row[name]), ".17e")
                        for name in fieldnames[1:]
                    },
                }
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep coherent-array size and quantify deviation from SNR proportional to N."
    )
    parser.add_argument(
        "--array-sizes",
        type=parse_array_sizes,
        default=parse_array_sizes(DEFAULT_ARRAY_SIZES),
        help=f"Comma-separated sizes beginning with 1. Default: {DEFAULT_ARRAY_SIZES}.",
    )
    parser.add_argument(
        "--strategy",
        choices=("chunk-center", "exact", "rigid"),
        default="chunk-center",
        help="Array construction strategy. Default: chunk-center.",
    )
    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=100_000,
        help="Number of generated rows held at once. Default: 100000.",
    )
    parser.add_argument(
        "--approximation-chunk-size",
        type=int,
        default=1_000,
        help="Sources sharing one optimized anchor in chunk-center mode. Default: 1000.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Optional source spacing in metres. Defaults to the project configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}.",
    )
    args = parser.parse_args()
    if args.generation_chunk_size < 1:
        parser.error("--generation-chunk-size must be positive.")
    if args.approximation_chunk_size < 1:
        parser.error("--approximation-chunk-size must be positive.")
    if args.spacing is not None and args.spacing <= 0.0:
        parser.error("--spacing must be positive.")
    return args


def main() -> None:
    args = parse_arguments()
    rows = sweep_array_snr(
        args.array_sizes,
        strategy=args.strategy,
        generation_chunk_size=args.generation_chunk_size,
        approximation_chunk_size=args.approximation_chunk_size,
        spacing=args.spacing,
    )
    write_results(rows, args.output)
    print(f"Saved high-precision sweep table to {args.output}")


if __name__ == "__main__":
    main()
