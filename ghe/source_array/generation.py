"""
High-level source-array construction.

A source-array row is the reusable description of one physical source:
position in the array frame, distance to the detector, detector-frame sky
direction, rotor-axis direction, and phase corrections needed for coherent
arrival at the detector.

The generation pipeline is:

1. Load or recompute the best single-source reference geometry.
2. Place sources on a compact 3D lattice around the array center.
3. For each source, compute its actual line of sight and distance.
4. Choose rotor-axis angles by exact optimization, rigid transport, or the
   chunk-anchor approximation.
5. Recover the response phase and store the rotor phase offset.
6. Stream rows to CSV or write a structured NPZ table.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ghe.config import SourceConfig
from ghe.geometry import cartesian_to_spherical, spherical_to_cartesian, spherical_unit_vector
from ghe.optimization import BestGeometry, scaled_spherical_function, scipy_gradient_descent, solve_best_geometry
from ghe.paths import SOURCE_ARRAY_DISTRIBUTION_FILE, SOURCE_ARRAY_NPZ_FILE

from .io import write_csv_rows, write_source_array_npz_file
from .layout import choose_lattice_dimensions, positions_for_index_range
from .phase import get_signal_amplitude_and_phase, rotor_phase_from_gw_phase, wrap_phase
from .schema import SOURCE_ARRAY_DTYPE
from .strategies import (
    approximate_chunk_center_parameters,
    optimize_rotor_for_fixed_source,
    rigid_rotor_angles,
)

DEFAULT_APPROXIMATION_CHUNK_SIZE = 1000


@dataclass(frozen=True)
class ArrayContext:
    """
    Shared geometry and strategy state for source-array generation.

    The context is immutable so chunk generation can be reasoned about as a pure
    function of ``(context, start, stop)``.  Large runs can therefore stream
    chunks without carrying mutable global state.
    """

    num_sources: int
    spacing: float
    layout: tuple[int, int, int]
    config: SourceConfig
    reference_geometry: BestGeometry
    n_src_center_vec: np.ndarray = field(repr=False)
    detector_position: np.ndarray = field(repr=False)
    gw_angular_frequency: float
    reference_signal_phase: float
    reference_rotor_theta: float
    reference_rotor_phi: float
    optimize_each_source: bool
    chunk_center_approximation: bool
    approximation_chunk_size: int

    def geometry_vector(self, theta: float, phi: float) -> np.ndarray:
        """Return a detector-frame unit vector for spherical angles."""

        return spherical_unit_vector(theta, phi)

    def metadata(self) -> dict[str, object]:
        """Return run metadata written alongside NPZ source-array artifacts."""

        return {
            "num_sources": self.num_sources,
            "spacing": self.spacing,
            "layout": self.layout,
            "source_config": asdict(self.config),
            "reference_geometry": asdict(self.reference_geometry),
            "generation_strategy": generation_strategy_name(
                self.optimize_each_source,
                self.chunk_center_approximation,
            ),
            "approximation_chunk_size": self.approximation_chunk_size,
            "git_commit": _git_commit(),
        }


def _git_commit() -> str | None:
    """Best-effort short git commit for reproducibility metadata."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def generation_strategy_name(optimize_each_source: bool, chunk_center_approximation: bool) -> str:
    """Map generation flags to the strategy names used in docs and metadata."""

    if chunk_center_approximation:
        return "chunk_anchor"
    if optimize_each_source:
        return "exact"
    return "rigid"


def build_array_context(
    num_sources: int,
    spacing: float | None = None,
    theta_array: float | None = None,
    phi_array: float | None = None,
    recompute_best_position: bool = False,
    optimize_each_source: bool = False,
    chunk_center_approximation: bool = False,
    approximation_chunk_size: int = DEFAULT_APPROXIMATION_CHUNK_SIZE,
    config: SourceConfig | None = None,
) -> ArrayContext:
    """
    Build the fixed context needed before emitting source rows.

    ``theta_array`` and ``phi_array`` optionally override the array-center sky
    direction.  If omitted, the source array points along the cached best
    single-source geometry.  The detector vertex is placed at
    ``-R * n_src_center`` in the array frame so the array center sits at the
    configured source-detector distance.
    """

    active_config = config or SourceConfig()
    if num_sources < 1:
        raise ValueError("num_sources must be positive.")
    if approximation_chunk_size < 1:
        raise ValueError("approximation_chunk_size must be positive.")

    spacing = (active_config.D * 3 / 2) if spacing is None else float(spacing)
    reference_geometry = solve_best_geometry(recompute=recompute_best_position)

    if theta_array is None or phi_array is None:
        # Default center direction: reuse the optimized single-source pointing.
        n_src_center_vec = np.asarray(reference_geometry.n_src_to_det_vec, dtype=float)
        n_src_center_vec = n_src_center_vec / np.linalg.norm(n_src_center_vec)
    else:
        n_src_center_vec = spherical_to_cartesian(theta_array, phi_array)
        if n_src_center_vec.ndim != 1:
            raise ValueError("theta_array/phi_array must be scalars for the center direction.")
        n_src_center_vec = np.asarray(n_src_center_vec, dtype=float).reshape(3)
        n_src_center_vec = n_src_center_vec / np.linalg.norm(n_src_center_vec)

    detector_position = -active_config.R * n_src_center_vec
    theta_center, phi_center = cartesian_to_spherical(np.asarray([n_src_center_vec]))
    theta_center = float(theta_center[0])
    phi_center = float(phi_center[0])

    if not optimize_each_source and theta_array is None and phi_array is None:
        # Fast rigid mode can reuse the reference rotor axis and phase exactly at
        # the array center.
        reference_rotor_theta = reference_geometry.theta_rot
        reference_rotor_phi = reference_geometry.phi_rot
        _, reference_signal_phase = get_signal_amplitude_and_phase(
            reference_geometry.theta_src,
            reference_geometry.phi_src,
            reference_rotor_theta,
            reference_rotor_phi,
            active_config.R,
            config=active_config,
        )
    else:
        # Exact and chunk-anchor strategies need a rotor optimum for the chosen
        # array-center sky direction.  Only the rotor angles move here; the source
        # direction is fixed by the array placement.
        _, _, reference_rotor_theta, reference_rotor_phi = scipy_gradient_descent(
            scaled_spherical_function,
            theta_center,
            phi_center,
            reference_geometry.theta_rot,
            reference_geometry.phi_rot,
            fix_source_angles=True,
        )
        _, reference_signal_phase = get_signal_amplitude_and_phase(
            theta_center,
            phi_center,
            reference_rotor_theta,
            reference_rotor_phi,
            active_config.R,
            config=active_config,
        )

    return ArrayContext(
        num_sources=num_sources,
        spacing=spacing,
        layout=choose_lattice_dimensions(num_sources),
        config=active_config,
        reference_geometry=reference_geometry,
        n_src_center_vec=n_src_center_vec,
        detector_position=detector_position,
        gw_angular_frequency=2.0 * active_config.omega,
        reference_signal_phase=reference_signal_phase,
        reference_rotor_theta=float(reference_rotor_theta),
        reference_rotor_phi=float(reference_rotor_phi),
        optimize_each_source=optimize_each_source,
        chunk_center_approximation=chunk_center_approximation,
        approximation_chunk_size=approximation_chunk_size,
    )


def build_chunk(context: ArrayContext, start: int, stop: int) -> np.ndarray:
    """
    Build one structured-array chunk of source rows.

    This is the central row-generation function.  It turns lattice positions into
    detector geometry, selects rotor angles according to the context strategy,
    derives phase compensation, and packs everything into ``SOURCE_ARRAY_DTYPE``.
    """

    indices, positions = positions_for_index_range(start, stop, context.layout, context.spacing)

    # Source coordinates are in the array frame.  The detector vertex is fixed in
    # that frame, so every off-center source has a slightly different distance and
    # line of sight.
    detector_vectors = context.detector_position - positions
    distances = np.linalg.norm(detector_vectors, axis=1)
    u_src_to_detector = detector_vectors / distances[:, None]

    # Metric response expects the vector from detector to source, the opposite of
    # the source-to-detector vector computed above.
    n_det_to_src = -u_src_to_detector
    theta_src, phi_src = cartesian_to_spherical(n_det_to_src)

    if context.chunk_center_approximation:
        theta_rot, phi_rot, gw_phase_offset = approximate_chunk_center_parameters(
            context,
            indices,
            distances,
            u_src_to_detector,
        )
    elif context.optimize_each_source:
        # Exact strategy: optimize the rotor axis for every fixed source sky
        # direction. This is slow but closest to the single-source optimum.
        theta_rot_list = []
        phi_rot_list = []
        for i in range(len(indices)):
            theta_rot_i, phi_rot_i = optimize_rotor_for_fixed_source(
                context,
                float(theta_src[i]),
                float(phi_src[i]),
            )
            theta_rot_list.append(theta_rot_i)
            phi_rot_list.append(phi_rot_i)
        theta_rot = np.array(theta_rot_list)
        phi_rot = np.array(phi_rot_list)
    else:
        # Rigid strategy: rotate the reference rotor axis with the line of sight.
        # This is much faster and keeps neighboring sources geometrically smooth.
        theta_rot, phi_rot = rigid_rotor_angles(context, u_src_to_detector)

    # These diagnostics preserve geometric timing information in the output
    # table. The actual coherent phase correction below comes from response phase.
    distance_offset = distances - context.config.R
    propagation_compensation = distance_offset / context.config.c

    if not context.chunk_center_approximation:
        # Recover the detector-response phase directly. This captures near-field
        # effects that a simple distance/c phase approximation can miss.
        signal_phases = np.empty(len(indices), dtype=float)
        for i in range(len(indices)):
            _, signal_phases[i] = get_signal_amplitude_and_phase(
                float(theta_src[i]),
                float(phi_src[i]),
                float(theta_rot[i]),
                float(phi_rot[i]),
                float(distances[i]),
                config=context.config,
            )
        gw_phase_offset = wrap_phase(signal_phases - context.reference_signal_phase)

    # Mechanical rotor phase is half the emitted GW phase because the quadrupole
    # radiation oscillates at twice the rotor frequency.
    rotor_phase_offset = rotor_phase_from_gw_phase(gw_phase_offset)

    chunk = np.empty(stop - start, dtype=SOURCE_ARRAY_DTYPE)
    chunk["source_id"] = indices
    chunk["x_m"] = positions[:, 0]
    chunk["y_m"] = positions[:, 1]
    chunk["z_m"] = positions[:, 2]
    chunk["distance_to_detector_m"] = distances
    chunk["distance_offset_m"] = distance_offset
    chunk["propagation_compensation_s"] = propagation_compensation
    chunk["theta_src"] = theta_src
    chunk["phi_src"] = phi_src
    chunk["theta_rot"] = theta_rot
    chunk["phi_rot"] = phi_rot
    chunk["gw_phase_offset_rad"] = gw_phase_offset
    chunk["rotor_phase_offset_rad"] = rotor_phase_offset
    return chunk


def iter_source_chunks(context: ArrayContext, chunk_size: int = 100_000) -> Iterator[np.ndarray]:
    """Yield source-array rows in chunks suitable for streaming to disk."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, context.num_sources, chunk_size):
        stop = min(start + chunk_size, context.num_sources)
        yield build_chunk(context, start, stop)


def construct_source_array(
    num: int,
    theta_array: float | None = None,
    phi_array: float | None = None,
    spacing: float | None = None,
    recompute_best_position: bool = False,
    optimize_each_source: bool = False,
    chunk_center_approximation: bool = False,
    approximation_chunk_size: int = DEFAULT_APPROXIMATION_CHUNK_SIZE,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """
    Build a complete source array in memory.

    This is convenient for tests and small analyses.  For production-scale arrays
    near the project goal, use the streaming CSV/NPZ writers instead.
    """

    if num > 2_000_000:
        raise ValueError(
            "construct_source_array() materializes the full table in memory. "
            "For very large arrays, use write_source_array_csv()."
        )

    context = build_array_context(
        num_sources=num,
        spacing=spacing,
        theta_array=theta_array,
        phi_array=phi_array,
        recompute_best_position=recompute_best_position,
        optimize_each_source=optimize_each_source,
        chunk_center_approximation=chunk_center_approximation,
        approximation_chunk_size=approximation_chunk_size,
    )
    chunks = [chunk for chunk in iter_source_chunks(context, chunk_size=chunk_size)]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=SOURCE_ARRAY_DTYPE)


def write_source_array_csv(
    output_path: str | Path = SOURCE_ARRAY_DISTRIBUTION_FILE,
    num_sources: int = 10_000_000,
    theta_array: float | None = None,
    phi_array: float | None = None,
    spacing: float | None = None,
    recompute_best_position: bool = False,
    optimize_each_source: bool = False,
    chunk_center_approximation: bool = False,
    approximation_chunk_size: int = DEFAULT_APPROXIMATION_CHUNK_SIZE,
    chunk_size: int = 100_000,
) -> ArrayContext:
    """
    Stream source-array rows to the legacy CSV format.

    CSV remains useful for inspection and compatibility, but it is slower and
    larger than NPZ for high-source-count experiments.
    """

    context = build_array_context(
        num_sources=num_sources,
        spacing=spacing,
        theta_array=theta_array,
        phi_array=phi_array,
        recompute_best_position=recompute_best_position,
        optimize_each_source=optimize_each_source,
        chunk_center_approximation=chunk_center_approximation,
        approximation_chunk_size=approximation_chunk_size,
    )
    write_csv_rows(output_path, iter_source_chunks(context, chunk_size=chunk_size))
    return context


def write_source_array_npz(
    output_path: str | Path = SOURCE_ARRAY_NPZ_FILE,
    num_sources: int = 10_000,
    theta_array: float | None = None,
    phi_array: float | None = None,
    spacing: float | None = None,
    recompute_best_position: bool = False,
    optimize_each_source: bool = False,
    chunk_center_approximation: bool = False,
    approximation_chunk_size: int = DEFAULT_APPROXIMATION_CHUNK_SIZE,
    chunk_size: int = 100_000,
) -> ArrayContext:
    """
    Write a structured NPZ source-array artifact plus metadata.

    NPZ is intended as the preferred package format when the array fits in memory
    during writing.  It preserves dtypes and avoids CSV conversion overhead.
    """

    context = build_array_context(
        num_sources=num_sources,
        spacing=spacing,
        theta_array=theta_array,
        phi_array=phi_array,
        recompute_best_position=recompute_best_position,
        optimize_each_source=optimize_each_source,
        chunk_center_approximation=chunk_center_approximation,
        approximation_chunk_size=approximation_chunk_size,
    )
    chunks = [chunk for chunk in iter_source_chunks(context, chunk_size=chunk_size)]
    source_array = np.concatenate(chunks) if chunks else np.empty(0, dtype=SOURCE_ARRAY_DTYPE)
    write_source_array_npz_file(output_path, source_array, metadata=context.metadata())
    return context


def format_vector(vector: np.ndarray) -> str:
    """Format a three-vector for human-readable CLI summaries."""

    return ", ".join(f"{component:.6f}" for component in vector)


def summary_lines(context: ArrayContext, preview_rows: np.ndarray | None = None) -> list[str]:
    """Build the human-readable source-array summary printed by the CLI wrapper."""

    theta_center, phi_center = cartesian_to_spherical(context.n_src_center_vec)
    lines = [
        f"Reference best-position amplitude = {context.reference_geometry.signal_amplitude:.6e}",
        "Reference rotor axis (detector frame) = "
        f"({format_vector(context.reference_geometry.rot_axis_vec)})",
    ]
    if context.chunk_center_approximation:
        lines.append(
            "Chunk-center approximation: one exact anchor per "
            f"{context.approximation_chunk_size} sources, with local rotor/phase approximation."
        )
    elif context.optimize_each_source:
        lines.append(
            "Per-source mode: source sky angles are fixed by lattice geometry; "
            "each source re-optimizes only (theta_rot, phi_rot)."
        )
    lines.extend(
        [
            "Array layout (nx, ny, nz) = "
            f"{context.layout}, spacing = {context.spacing:.3f} m, sources = {context.num_sources}",
            "Array-center direction (det -> src) = "
            f"(theta={theta_center[0]:.8f}, phi={phi_center[0]:.8f})",
            f"Detector vertex position [m] = ({format_vector(context.detector_position)})",
        ]
    )
    if preview_rows is not None and preview_rows.size > 0:
        lines.append("")
        lines.append("Preview rows:")
        for row in preview_rows:
            lines.append(
                "source_id={source_id}, pos=({x_m:.3f}, {y_m:.3f}, {z_m:.3f}), "
                "gw_phase={gw_phase_offset_rad:.6e}, rotor_phase={rotor_phase_offset_rad:.6e}".format(
                    **{name: row[name] for name in row.dtype.names}
                )
            )
    return lines
