"""Coherent superposition of monochromatic source responses.

Only this layer interprets stored rotor phase delays. Spatial integration is
performed once per source, independent of the number of time samples. CSV input
is streamed; NPZ remains a convenient in-memory format for smaller arrays.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import SourceConfig
from .metric import calculate_response_phasor
from .paths import SOURCE_ARRAY_DISTRIBUTION_FILE, SOURCE_ARRAY_NPZ_FILE
from .source_array.io import iter_source_array_file, validate_source_array_file


def source_phase_time_offset(row: np.void, config: SourceConfig) -> float:
    """Stored mechanical delay [rad] divided by spin frequency [rad/s]."""
    return float(row["rotor_phase_offset_rad"]) / config.omega


def source_response_phasor(row, config=None) -> complex:
    """Return a row's complete dimensionless response INCLUDING its phase delay.

    For h(t)=Re[H exp(-iΩt)], delaying the source by δ/ω gives H exp(+2iδ).
    Source retardation is already in H; propagation_compensation_s must NOT be
    applied again. The historical gw offset is checked as a redundant invariant.
    """
    cfg = config or SourceConfig()
    delta = float(row["rotor_phase_offset_rad"])
    gw_delta = float(row["gw_phase_offset_rad"])
    if (
        not np.isfinite(delta)
        or not np.isfinite(gw_delta)
        or abs(np.exp(2j * delta) - np.exp(1j * gw_delta)) > 1e-10
    ):
        raise ValueError(
            "Array phase offsets must satisfy gw_offset = 2*rotor_offset modulo 2*pi"
        )
    H = calculate_response_phasor(
        *(
            float(row[name])
            for name in ("theta_src", "phi_src", "theta_rot", "phi_rot")
        ),
        R=float(row["distance_to_detector_m"]),
        config=cfg,
    )
    return H * np.exp(2j * delta)


def synthesize_signal(phasor, time_axis, config=None):
    """Synthesize reception-time samples [s] from one peak strain phasor."""
    times = np.asarray(time_axis, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("Signal times must be finite")
    return np.real(
        phasor * np.exp(-1j * (config or SourceConfig()).gw_angular_frequency * times)
    )


def calculate_single_source_response(t, row, *, config=None):
    """Single row at reception time t [s]; scalar counterpart of synthesis."""
    return float(synthesize_signal(source_response_phasor(row, config), t, config))


def iter_loaded_source_chunks(source_array, chunk_size):
    """Bound working response storage independently of the loaded table size."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(source_array), chunk_size):
        yield source_array[start : start + chunk_size]


def _sum_chunks(chunks, config):
    # Compensated complex summation reduces cancellation/roundoff and makes the
    # result independent of I/O chunk boundaries; order remains source-row order.
    total, correction = 0j, 0j
    for chunk in chunks:
        for row in chunk:
            adjusted = source_response_phasor(row, config) - correction
            updated = total + adjusted
            correction = (updated - total) - adjusted
            total = updated
    return total


def compute_phasor_sum(source_array, *, config=None, chunk_size=10_000):
    """Sum peak exp(-iΩt) phasors of an in-memory structured source table."""
    return _sum_chunks(
        iter_loaded_source_chunks(source_array, chunk_size), config or SourceConfig()
    )


def calculate_chunk_response(time_axis, chunk, *, config=None):
    """Integrate each source once, then synthesize the chunk's total signal."""
    return synthesize_signal(
        compute_phasor_sum(chunk, config=config), time_axis, config
    )


def calculate_source_array_signal(
    time_axis, source_array, *, config=None, chunk_size=10_000
):
    """Return a real array with the same shape as time_axis."""
    H = compute_phasor_sum(source_array, config=config, chunk_size=chunk_size)
    return synthesize_signal(H, time_axis, config)


def choose_source_array_input(
    preferred_npz=SOURCE_ARRAY_NPZ_FILE,
    fallback_csv=SOURCE_ARRAY_DISTRIBUTION_FILE,
    *,
    config=None,
):
    """Choose a compatible artifact; stale NPZ cannot shadow a current CSV."""
    errors = []
    for candidate in map(Path, (preferred_npz, fallback_csv)):
        if candidate.is_file():
            try:
                validate_source_array_file(candidate, config=config)
                return candidate
            except ValueError as exc:
                errors.append(str(exc))
    raise ValueError(
        "No compatible source array. Regenerate with --renew-source-array. "
        + "; ".join(errors)
    )


def compute_phasor_sum_from_file(input_path=None, *, config=None, chunk_size=10_000):
    """Validate model/configuration before streaming source rows into the sum."""
    path = (
        Path(input_path)
        if input_path is not None
        else choose_source_array_input(config=config)
    )
    return _sum_chunks(
        iter_source_array_file(path, chunk_size, config=config),
        config or SourceConfig(),
    )


def calculate_source_array_signal_from_file(
    time_axis, input_path=None, *, config=None, chunk_size=10_000
):
    """Validated file-based counterpart of calculate_source_array_signal."""
    H = compute_phasor_sum_from_file(input_path, config=config, chunk_size=chunk_size)
    return synthesize_signal(H, time_axis, config)
