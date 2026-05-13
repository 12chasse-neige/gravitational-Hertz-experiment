"""
Top-level compatibility CLI for a source-array SNR run.

This file intentionally stays small: the numerical work lives in ``ghe/``.  The
pipeline here is:

1. Optionally regenerate a coherent source-array table.
2. Load the source-array rows from CSV or NPZ.
3. Build the shared time axis.
4. Accumulate the detector strain time series from all sources.
5. Fourier transform the strain.
6. Integrate the signal against the detector noise PSD to estimate 1-year SNR.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ghe.config import RunConfig, build_time_axis
from ghe.paths import (
    SOURCE_ARRAY_DISTRIBUTION_FILE,
    SOURCE_ARRAY_NPZ_FILE,
    TOTAL_FREQS_FILE,
    TOTAL_MAGNITUDE_FILE,
    make_run_dir,
)
from ghe.signal import (
    calculate_single_source_response,
    calculate_source_array_signal_from_file,
    choose_source_array_input,
)
from ghe.snr import calculate_snr, save_snr_json
from ghe.source_array.generation import write_source_array_csv, write_source_array_npz
from ghe.source_array.io import read_source_array
from ghe.spectrum import calculate_spectrum, save_spectrum_arrays, save_spectrum_npz

DEFAULT_SOURCE_ARRAY_NUM_SOURCES = 10_000_000
DEFAULT_SOURCE_ARRAY_CHUNK_SIZE = 100_000


def read_csv_line(ID: int) -> np.ndarray:
    """
    Legacy helper: return one numeric row from the compatibility CSV table.

    New code should prefer ``ghe.source_array.io.read_source_array`` because it
    supports both CSV and NPZ and avoids repeatedly materializing the file.
    """

    with open(SOURCE_ARRAY_DISTRIBUTION_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if ID + 1 >= len(rows):
        raise IndexError(f"Source ID {ID} is out of range ({len(rows) - 1} sources available)")

    return np.array([float(value) for value in rows[ID + 1]], dtype=float)


def get_single_source_metric_response(t: float, ID: int) -> float:
    """
    Compatibility helper for the old script API.

    It looks up one source row, applies that row's phase compensation, and then
    calls the package metric response calculation through ``ghe.signal``.
    """

    source_array = read_source_array(SOURCE_ARRAY_DISTRIBUTION_FILE)
    if ID >= len(source_array):
        raise IndexError(f"Source ID {ID} is out of range ({len(source_array)} sources available)")
    return calculate_single_source_response(t, source_array[ID])


def get_source_num(input_path=SOURCE_ARRAY_DISTRIBUTION_FILE) -> int:
    """Return the number of source rows in a CSV table, excluding the header."""

    with open(input_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        return max(sum(1 for _ in reader) - 1, 0)


def get_total_signal(t: float) -> float:
    """
    Compatibility helper: compute total strain at one time sample.

    The main pipeline uses the chunked array path below because this one-sample
    helper is mainly useful for simple debugging and older notebooks.
    """

    source_array = read_source_array(choose_source_array_input())
    return float(sum(calculate_single_source_response(t, row) for row in source_array))


def parse_arguments() -> argparse.Namespace:
    """Parse the original command-line options plus refactor-era output controls."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate the total source-array detector signal and SNR from the "
            "source-array distribution."
        )
    )
    parser.add_argument(
        "--renew-source-array",
        action="store_true",
        help="Regenerate the source array before calculating the total SNR.",
    )
    parser.add_argument(
        "--source-array-num-sources",
        type=int,
        default=DEFAULT_SOURCE_ARRAY_NUM_SOURCES,
        help=(
            "Number of sources to use when --renew-source-array is set. "
            f"Default: {DEFAULT_SOURCE_ARRAY_NUM_SOURCES}."
        ),
    )
    parser.add_argument(
        "--source-array-chunk-size",
        type=int,
        default=DEFAULT_SOURCE_ARRAY_CHUNK_SIZE,
        help=(
            "Chunk size for source-array regeneration and signal accumulation. "
            f"Default: {DEFAULT_SOURCE_ARRAY_CHUNK_SIZE}."
        ),
    )
    parser.add_argument(
        "--no-optimize-each-source",
        dest="optimize_each_source",
        action="store_false",
        default=True,
        help="Use the faster rigid-transport source-array approximation during regeneration.",
    )
    parser.add_argument(
        "--source-array-exact-optimization",
        action="store_true",
        help=(
            "Disable chunk-center approximation and optimize every source exactly "
            "when per-source optimization is enabled."
        ),
    )
    parser.add_argument(
        "--source-array-format",
        choices=("csv", "npz", "both"),
        default="csv",
        help="Output format used when --renew-source-array is set. Default: csv.",
    )
    parser.add_argument(
        "--source-array-input",
        type=Path,
        default=None,
        help="Optional source-array CSV or NPZ input path. Defaults to NPZ when present, otherwise CSV.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run output directory. Writes config, signal, spectrum, and SNR artifacts there.",
    )
    return parser.parse_args()


def renew_source_array(
    num_sources: int,
    chunk_size: int,
    optimize_each_source: bool,
    chunk_center_approximation: bool,
    output_format: str = "csv",
) -> Path:
    """
    Regenerate source-array storage and return the path main should read.

    ``output_format='csv'`` preserves the historical workflow. ``npz`` is the
    preferred package format for larger experiments because the structured array
    can be loaded directly without CSV parsing.
    """

    if num_sources < 1:
        raise ValueError("--source-array-num-sources must be positive.")
    if chunk_size < 1:
        raise ValueError("--source-array-chunk-size must be positive.")

    print("\nRenewing source array distribution...")
    if chunk_center_approximation:
        print(f"Using chunk-center approximation (one exact optimization per {chunk_size} sources).")
    elif optimize_each_source:
        print("Using exact per-source optimization.")
    else:
        print("Using rigid-transport source-array approximation.")

    input_path = SOURCE_ARRAY_DISTRIBUTION_FILE
    if output_format in ("csv", "both"):
        write_source_array_csv(
            output_path=SOURCE_ARRAY_DISTRIBUTION_FILE,
            num_sources=num_sources,
            optimize_each_source=optimize_each_source,
            chunk_center_approximation=chunk_center_approximation,
            approximation_chunk_size=chunk_size,
            chunk_size=chunk_size,
        )
        input_path = SOURCE_ARRAY_DISTRIBUTION_FILE
    if output_format in ("npz", "both"):
        write_source_array_npz(
            output_path=SOURCE_ARRAY_NPZ_FILE,
            num_sources=num_sources,
            optimize_each_source=optimize_each_source,
            chunk_center_approximation=chunk_center_approximation,
            approximation_chunk_size=chunk_size,
            chunk_size=chunk_size,
        )
        input_path = SOURCE_ARRAY_NPZ_FILE

    print("Source array distribution renewed.")
    return input_path


def main() -> None:
    """Run the end-to-end source-array signal, spectrum, and SNR calculation."""

    args = parse_arguments()

    source_array_input = args.source_array_input
    if args.renew_source_array:
        source_array_input = renew_source_array(
            num_sources=args.source_array_num_sources,
            chunk_size=args.source_array_chunk_size,
            optimize_each_source=args.optimize_each_source,
            chunk_center_approximation=(
                args.optimize_each_source and not args.source_array_exact_optimization
            ),
            output_format=args.source_array_format,
        )

    if source_array_input is None:
        source_array_input = choose_source_array_input()

    # A run directory is optional while legacy ``data/`` outputs remain the
    # default. When requested, it captures enough artifacts to replay a small run.
    run_dir = args.run_dir
    if run_dir is not None:
        run_dir = make_run_dir(run_dir.name, root=run_dir.parent)
        RunConfig.from_environment().to_json(run_dir / "config.json")

    # The expensive part: for every source row, evaluate the detector response on
    # the shared time grid and add it coherently after phase compensation.
    time_axis = build_time_axis()
    h_values = calculate_source_array_signal_from_file(
        time_axis,
        source_array_input,
        chunk_size=args.source_array_chunk_size,
    )

    # Keep the FFT normalization inherited from the original scripts. The saved
    # arrays are consumed by both the compatibility SNR CLI and package tests.
    spectrum = calculate_spectrum(h_values)
    save_spectrum_arrays(spectrum, magnitude_path=TOTAL_MAGNITUDE_FILE, freq_path=TOTAL_FREQS_FILE)
    if run_dir is not None:
        np.save(run_dir / "signal.npy", h_values)
        save_spectrum_npz(spectrum, run_dir / "spectrum.npz")

    # ``calculate_snr`` loads the saved arrays so this CLI still exercises the
    # same file-based workflow that older scripts used.
    snr_year = calculate_snr(TOTAL_MAGNITUDE_FILE, TOTAL_FREQS_FILE)
    print(f"Calculated SNR (1 year) = {snr_year:.4e}")
    if run_dir is not None:
        save_snr_json(snr_year, run_dir / "snr.json")


if __name__ == "__main__":
    main()
