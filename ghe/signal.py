"""
Time-domain accumulation for single sources and coherent source arrays.

Rows in a source-array table already contain each source's detector-frame sky
angles, rotor-axis angles, distance, and phase offset.  This module turns those
rows into a total strain time series by evaluating the single-source metric
response and summing all sources on the same time axis.

The chunked path avoids the original anti-pattern of rereading the full CSV for
each time sample.

Optimisation notes
------------------

* Per-source geometry vectors (``n_src_to_det`` and ``R_body_to_det``) are
  precomputed once and re-used for every time sample of that source.
* Intermediate per-source time-series arrays are avoided; response samples are
  accumulated directly into ``h_total``.
* Scalar fields are accessed via ``.item()`` to avoid repeated ``float()``
  boxing overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .config import SourceConfig
from .geometry import rotation_body_to_detector, spherical_unit_vector
from .metric import _calculate_metric_response_prepared
from .paths import SOURCE_ARRAY_DISTRIBUTION_FILE, SOURCE_ARRAY_NPZ_FILE
from .source_array.io import read_source_array
from .source_array.phase import get_signal_amplitude_and_phase


def source_phase_time_offset(row: np.void, config: SourceConfig) -> float:
    """
    Convert stored rotor phase offset into a time shift for response evaluation.

    The table stores the mechanical rotor phase; dividing by ``omega`` gives the
    equivalent time offset applied to the source response.
    """

    return row["rotor_phase_offset_rad"].item() / config.omega


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

    from .metric import calculate_metric_response

    active_config = config or SourceConfig()
    return calculate_metric_response(
        t - source_phase_time_offset(row, active_config),
        row["theta_src"].item(),
        row["phi_src"].item(),
        row["theta_rot"].item(),
        row["phi_rot"].item(),
        config=active_config,
        R=row["distance_to_detector_m"].item(),
    )


def _precompute_source_geometry(
    row: np.void,
    config: SourceConfig,
) -> tuple[SourceConfig, np.ndarray, np.ndarray]:
    """
    Return ``(config_with_R, n_src_to_det, R_body_to_det)`` for one source row.

    These are invariant across time samples and are used by the fast metric path.
    """

    from dataclasses import replace

    R = row["distance_to_detector_m"].item()
    config_with_R = replace(config, R=R) if R != config.R else config
    n_src_to_det = spherical_unit_vector(
        row["theta_src"].item(),
        row["phi_src"].item(),
    )
    R_body_to_det = rotation_body_to_detector(
        row["theta_rot"].item(),
        row["phi_rot"].item(),
    )
    return config_with_R, n_src_to_det, R_body_to_det


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
        phase_time_offset = row["rotor_phase_offset_rad"].item() / active_config.omega
        cfg, n_src_to_det, R_body_to_det = _precompute_source_geometry(row, active_config)

        for i in range(len(time_axis)):
            shifted_t = float(time_axis[i] - phase_time_offset)
            h_total[i] += _calculate_metric_response_prepared(
                shifted_t, n_src_to_det, R_body_to_det, cfg,
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


def compute_phasor_sum(
    source_array: np.ndarray,
    *,
    config: SourceConfig | None = None,
    chunk_size: int = 10_000,
) -> complex:
    """
    Compute the coherent phasor sum of all source amplitudes.

    For each source, the raw amplitude *A_i* and phase *φ_i* at *t=0* are
    recovered via ``get_signal_amplitude_and_phase``.  The stored GW phase
    correction ``gw_phase_offset_rad`` is then applied to align every source
    with the reference, producing the complex sum::

        H = Σ A_i · exp(j · (φ_i − gw_offset_i))

    ``|H|`` is the total coherent strain amplitude at the dominant GW frequency.
    """

    active_config = config or SourceConfig()
    phasor_sum = complex(0.0, 0.0)

    for chunk in iter_loaded_source_chunks(source_array, chunk_size):
        for row in chunk:
            amplitude, raw_phase = get_signal_amplitude_and_phase(
                row["theta_src"].item(),
                row["phi_src"].item(),
                row["theta_rot"].item(),
                row["phi_rot"].item(),
                row["distance_to_detector_m"].item(),
                config=active_config,
            )
            gw_offset = row["gw_phase_offset_rad"].item()
            phasor_sum += complex(
                amplitude * np.cos(raw_phase - gw_offset),
                amplitude * np.sin(raw_phase - gw_offset),
            )

    return phasor_sum


def compute_phasor_sum_from_file(
    input_path: str | Path | None = None,
    *,
    config: SourceConfig | None = None,
    chunk_size: int = 10_000,
) -> complex:
    """Load a source-array file and compute the monochromatic phasor sum."""

    source_path = Path(input_path) if input_path is not None else choose_source_array_input()
    source_array = read_source_array(source_path)
    return compute_phasor_sum(source_array, config=config, chunk_size=chunk_size)
