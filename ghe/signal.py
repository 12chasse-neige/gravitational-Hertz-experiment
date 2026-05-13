"""
Time-domain accumulation for single sources and coherent source arrays.

Rows in a source-array table already contain each source's detector-frame sky
angles, rotor-axis angles, distance, and phase offset.  This module turns those
rows into a total strain time series by evaluating the single-source metric
response and summing all sources on the same time axis.

The chunked path avoids the original anti-pattern of rereading the full CSV for
each time sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .config import SourceConfig
from .metric import calculate_metric_response
from .paths import SOURCE_ARRAY_DISTRIBUTION_FILE, SOURCE_ARRAY_NPZ_FILE
from .source_array.io import read_source_array


def source_phase_time_offset(row: np.void, config: SourceConfig) -> float:
    """
    Convert stored rotor phase offset into a time shift for response evaluation.

    The table stores the mechanical rotor phase; dividing by ``omega`` gives the
    equivalent time offset applied to the source response.
    """

    return float(row["rotor_phase_offset_rad"]) / config.omega


def calculate_single_source_response(
    t: float,
    row: np.void,
    *,
    config: SourceConfig | None = None,
) -> float:
    """
    Evaluate one source row at detector time ``t``.

    The row supplies geometry and distance.  Phase compensation is implemented as
    a time shift before calling the single-source metric response.
    """

    active_config = config or SourceConfig()
    return calculate_metric_response(
        t - source_phase_time_offset(row, active_config),
        float(row["theta_src"]),
        float(row["phi_src"]),
        float(row["theta_rot"]),
        float(row["phi_rot"]),
        config=active_config,
        R=float(row["distance_to_detector_m"]),
    )


def calculate_chunk_response(
    time_axis: np.ndarray,
    chunk: np.ndarray,
    *,
    config: SourceConfig | None = None,
) -> np.ndarray:
    """
    Accumulate a chunk of source rows over the complete time axis.

    This is intentionally simple and behavior-preserving.  It is the natural
    place to add vectorization later because all source rows for a chunk are
    already loaded together.
    """

    active_config = config or SourceConfig()
    h_total = np.zeros_like(time_axis, dtype=float)
    for row in chunk:
        phase_time_offset = source_phase_time_offset(row, active_config)
        shifted_times = time_axis - phase_time_offset
        h_total += np.array(
            [
                calculate_metric_response(
                    float(t),
                    float(row["theta_src"]),
                    float(row["phi_src"]),
                    float(row["theta_rot"]),
                    float(row["phi_rot"]),
                    config=active_config,
                    R=float(row["distance_to_detector_m"]),
                )
                for t in shifted_times
            ],
            dtype=float,
        )
    return h_total


def iter_loaded_source_chunks(source_array: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield slices of an already loaded structured source-array table."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, len(source_array), chunk_size):
        yield source_array[start : start + chunk_size]


def calculate_source_array_signal(
    time_axis: np.ndarray,
    source_array: np.ndarray,
    *,
    config: SourceConfig | None = None,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    Sum a full source-array table into a total strain time series.

    The returned array has the same shape as ``time_axis`` and represents the
    coherent superposition of all sources after their stored phase corrections.
    """

    active_config = config or SourceConfig()
    h_total = np.zeros_like(time_axis, dtype=float)
    for chunk in iter_loaded_source_chunks(source_array, chunk_size):
        h_total += calculate_chunk_response(time_axis, chunk, config=active_config)
    return h_total


def choose_source_array_input(
    preferred_npz: str | Path = SOURCE_ARRAY_NPZ_FILE,
    fallback_csv: str | Path = SOURCE_ARRAY_DISTRIBUTION_FILE,
) -> Path:
    """
    Pick the default source-array storage path.

    NPZ is preferred when available because it preserves the structured dtype and
    is faster to load; CSV remains the compatibility fallback.
    """

    npz_path = Path(preferred_npz)
    if npz_path.is_file():
        return npz_path
    return Path(fallback_csv)


def calculate_source_array_signal_from_file(
    time_axis: np.ndarray,
    input_path: str | Path | None = None,
    *,
    config: SourceConfig | None = None,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """Load a source-array file and calculate its total strain time series."""

    source_path = Path(input_path) if input_path is not None else choose_source_array_input()
    source_array = read_source_array(source_path)
    return calculate_source_array_signal(
        time_axis,
        source_array,
        config=config,
        chunk_size=chunk_size,
    )
