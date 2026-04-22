from __future__ import annotations

"""
Build a many-source array that stays consistent with the single-source model.

This file does three main jobs:

1. Read or recompute the "best" single-source geometry from ``bestPosition.py`` (four
   detector-frame angles: sky location + rotor axis).
2. Place many identical sources on a 3D lattice around an array center.
3. For each source, compute the local sky angles, rigidly transport the optimized
   rotor axis to the local line of sight, and compute the phase shift needed so
   that all GW signals arrive coherently at the detector.

The interferometer arms are fixed globally along ``+x`` and ``+y`` as in 
``metricCalculate.py``, so arm directions are no longer rotated per source.
"""

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import scripts.bestPosition as bestPosition
from scripts.metricCalculate import ExperimentConfig, spherical_unit_vector

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_BEST_POSITION_FILE = _DATA_DIR / "bestPosition.txt"
_DEFAULT_OUTPUT_FILE = _DATA_DIR / "source_array_distribution.csv"
TWO_PI = 2.0 * np.pi

SOURCE_ARRAY_DTYPE = np.dtype(
    [
        ("source_id", np.int64),
        ("x_m", np.float64),
        ("y_m", np.float64),
        ("z_m", np.float64),
        ("distance_to_detector_m", np.float64),
        ("distance_offset_m", np.float64),
        ("propagation_compensation_s", np.float64),
        # Detector frame (same convention as ``metricCalculate.calculate_metric_response``):
        # ``(theta_src, phi_src)`` describe the unit vector from the detector vertex toward
        # the source; ``(theta_rot, phi_rot)`` describe the rotor symmetry axis.
        ("theta_src", np.float64),
        ("phi_src", np.float64),
        ("theta_rot", np.float64),
        ("phi_rot", np.float64),
        ("gw_phase_offset_rad", np.float64),
        ("rotor_phase_offset_rad", np.float64),
    ]
)


@dataclass(frozen=True)
class RelativeGeometry:
    """
    Optimized single-source geometry in the **detector frame**.

    The four angles match ``data/bestPosition.txt`` (see ``bestPosition.py``). The
    cached unit vectors are convenient for rigidly transporting the rotor axis when
    the line of sight changes across the array.
    """
    theta_src: float
    phi_src: float
    theta_rot: float
    phi_rot: float
    signal_amplitude: float
    # unit vector from the detector vertex toward the nominal (center) source.
    n_src_to_det_vec: np.ndarray = field(repr=False)
    # unit vector from the nominal (center) source location toward the detector.
    u_src_to_detector_center_vec: np.ndarray = field(repr=False)
    # rotor symmetry axis (body +z) expressed in detector coordinates.
    rot_axis_vec: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class ArrayContext:
    """
    Shared information needed to generate array members.

    This is separated from the per-source table because for very large arrays
    we stream rows chunk by chunk instead of materializing everything at once.
    """
    num_sources: int
    spacing: float
    layout: tuple[int, int, int]
    config: ExperimentConfig
    reference_geometry: RelativeGeometry | None
    # Unit vector from the detector vertex toward the array center (nominal pointing).
    n_src_center_vec: np.ndarray = field(repr = False)
    detector_position: np.ndarray = field(repr = False)
    gw_angular_frequency: float
    optimize_each_source: bool


def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical angles into a unit vector (detector-frame convention).

    **Note:** This is kept as a small wrapper so bulk call sites can still broadcast,
    but it uses the same polar/azimuth definition as ``metricCalculate.spherical_unit_vector``.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )


def cartesian_to_spherical(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert one or more Cartesian vectors into (theta, phi)."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]

    radii = np.linalg.norm(vectors, axis=1)
    theta = np.arccos(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    phi = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), TWO_PI)
    return theta, phi


def _parse_best_position_text(text: str) -> tuple[float, float, float, float] | None:
    """
    Read the four best-position angles from ``data/bestPosition.txt``.

    Expected machine-readable line::

        BEST_POSITION: theta_src, phi_src, theta_rot, phi_rot

    Legacy six-angle outputs are not parsed anymore; rerun ``bestPosition.py`` after
    the detector-frame refactor.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("BEST_POSITION:"):
            values = [x.strip() for x in stripped.split(":", 1)[1].split(",")]
            if len(values) == 4:
                return tuple(float(value) for value in values)  # type: ignore[return-value]

    return None


def _write_best_position_file(
    angles: tuple[float, float, float, float],
    amplitude: float,
    path: Path = _BEST_POSITION_FILE,
) -> None:
    """
    Write the optimizer result back using the same compact format as ``bestPosition.py``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# Detector frame: vertex at origin; arm1 +x; arm2 +y; +z completes RHS.\n"
            "# (theta_src, phi_src): unit vector from detector toward the source.\n"
            "# (theta_rot, phi_rot): rotor symmetry axis (body +z) in detector frame.\n"
        )
        handle.write(
            f"BEST_POSITION: {angles[0]:.8f}, {angles[1]:.8f}, {angles[2]:.8f}, {angles[3]:.8f}\n"
        )
        handle.write(f"max_signal_amplitude: {amplitude:.12e}\n")


def solve_best_geometry(recompute: bool = False) -> RelativeGeometry:
    """
    Get the best single-source geometry used as the array reference.

    If recompute=False, we prefer the cached values in data/bestPosition.txt.
    If recompute=True, we call the optimizer from bestPosition.py and refresh
    the file so the rest of the project sees the same updated angles.
    """
    angles: tuple[float, float, float, float] | None = None
    used_optimizer = False

    if not recompute and _BEST_POSITION_FILE.is_file():
        try:
            angles = _parse_best_position_text(_BEST_POSITION_FILE.read_text(encoding="utf-8"))
        except OSError:
            angles = None

    if angles is None:
        used_optimizer = True
        # Match ``bestPosition.py`` cold start so CLI runs and library calls stay aligned.
        initial_theta_src = 0.1
        initial_phi_src = 0.0
        initial_theta_rot = 1.0
        initial_phi_rot = 0.0
        angles = bestPosition.scipy_gradient_descent(
            bestPosition.scaled_spherical_function,
            initial_theta_src,
            initial_phi_src,
            initial_theta_rot,
            initial_phi_rot,
        )

    amplitude = bestPosition.spherical_function(*angles)
    if used_optimizer:
        _write_best_position_file(angles, amplitude)

    theta_src, phi_src, theta_rot, phi_rot = angles
    n_src_to_det_vec = spherical_unit_vector(theta_src, phi_src)
    # Nominal array layout places the array center at the origin and the vertex at
    # ``detector_position = -R * n_src`` so that ``n_src`` is indeed det->center-source.
    u_src_to_detector_center_vec = -n_src_to_det_vec
    rot_axis_vec = spherical_unit_vector(theta_rot, phi_rot)

    return RelativeGeometry(
        theta_src=float(theta_src),
        phi_src=float(phi_src),
        theta_rot=float(theta_rot),
        phi_rot=float(phi_rot),
        signal_amplitude=float(amplitude),
        n_src_to_det_vec=n_src_to_det_vec,
        u_src_to_detector_center_vec=u_src_to_detector_center_vec,
        rot_axis_vec=rot_axis_vec,
    )


def _get_perpendicular_axis(vector: np.ndarray) -> np.ndarray:
    """
    Return any unit vector perpendicular to the input unit vector.

    This is only needed for the nearly-180-degree rotation case where the
    usual cross-product formula becomes numerically singular.
    """
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)
    trial = np.array([1.0, 0.0, 0.0]) if abs(vector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(vector, trial)
    axis /= np.linalg.norm(axis)
    return axis


def rotate_reference_vector(
    reference_vector: np.ndarray,
    reference_direction: np.ndarray,
    target_directions: np.ndarray,
) -> np.ndarray:
    """
    Rotate one reference vector so the reference propagation direction is mapped
    onto each target propagation direction by the minimal rigid rotation.

    In this file, the "reference direction" is the center-source direction
    toward the detector, and the rotated vectors are the local detector arms
    for each off-center source.
    """
    reference_vector = np.asarray(reference_vector, dtype=float)
    reference_direction = np.asarray(reference_direction, dtype=float)
    reference_direction = reference_direction / np.linalg.norm(reference_direction)

    target_directions = np.asarray(target_directions, dtype=float)
    if target_directions.ndim == 1:
        target_directions = target_directions[np.newaxis, :]
    target_directions = target_directions / np.linalg.norm(target_directions, axis=1, keepdims=True)

    ref = np.broadcast_to(reference_direction, target_directions.shape)
    u = np.broadcast_to(reference_vector, target_directions.shape)
    cross_term = np.cross(ref, target_directions)
    dot_term = np.einsum("ij,ij->i", ref, target_directions)

    rotated = np.empty_like(target_directions)
    same_mask = dot_term > 1.0 - 1e-12
    opposite_mask = dot_term < -1.0 + 1e-12
    general_mask = ~(same_mask | opposite_mask)

    if np.any(same_mask):
        rotated[same_mask] = u[same_mask]

    if np.any(general_mask):
        v = cross_term[general_mask]
        u_general = u[general_mask]
        # Rodrigues-style closed form for the minimal rotation that sends
        # reference_direction -> target_direction without constructing a full
        # rotation matrix for every source.
        correction = np.cross(v, np.cross(v, u_general)) / (1.0 + dot_term[general_mask])[:, None]
        rotated[general_mask] = u_general + np.cross(v, u_general) + correction

    if np.any(opposite_mask):
        axis = _get_perpendicular_axis(reference_direction)
        u_opposite = u[opposite_mask]
        rotated[opposite_mask] = -u_opposite + 2.0 * np.outer(u_opposite @ axis, axis)

    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    return rotated


def choose_lattice_dimensions(num_sources: int) -> tuple[int, int, int]:
    """
    Choose a factorization (nx, ny, nz) that is as cube-like as possible.

    A near-cubic layout is a simple default because it keeps the array compact
    and minimizes the largest propagation-delay spread for a given spacing.
    """
    if num_sources < 1:
        raise ValueError("num_sources must be positive.")

    best_dims = (1, 1, num_sources)
    best_score = (num_sources - 1, num_sources - 1, num_sources)

    max_a = int(round(num_sources ** (1.0 / 3.0))) + 2
    for a in range(1, max_a + 1):
        if num_sources % a != 0:
            continue

        remainder = num_sources // a
        b = int(math.isqrt(remainder))
        while b >= a:
            if remainder % b == 0:
                c = remainder // b
                dims = tuple(sorted((a, b, c)))
                score = (dims[2] - dims[0], dims[1] - dims[0], dims[2])
                if score < best_score:
                    best_dims = dims
                    best_score = score
                break
            b -= 1

    return best_dims


def build_array_context(
    num_sources: int,
    spacing: float | None = None,
    theta_array: float | None = None,
    phi_array: float | None = None,
    recompute_best_position: bool = False,
    optimize_each_source: bool = False,
) -> ArrayContext:
    """
    Build the geometry shared by the full source array.

    If ``theta_array``/``phi_array`` are omitted, the array center direction matches
    the optimized single-source pointing from ``data/bestPosition.txt``.

    **Detector frame placement.** Sources live near the origin; the interferometer
    vertex is placed at ``detector_position = -R * n_src_center`` so that the
    nominal separation matches ``ExperimentConfig.R`` and the unit vector from the
    vertex toward the array center equals ``n_src_center``.
    """
    config = ExperimentConfig()
    spacing = (config.D * 3/2) if spacing is None else float(spacing)

    reference_geometry = None
    if not optimize_each_source:
        reference_geometry = solve_best_geometry(recompute=recompute_best_position)
        if theta_array is None or phi_array is None:
            # Reuse the optimized sky location for the array center.
            n_src_center_vec = np.asarray(reference_geometry.n_src_to_det_vec, dtype=float)
            n_src_center_vec = n_src_center_vec / np.linalg.norm(n_src_center_vec)
        else:
            n_src_center_vec = spherical_to_cartesian(theta_array, phi_array)
            if n_src_center_vec.ndim != 1:
                raise ValueError("theta_array/phi_array must be scalars for the center direction.")
            n_src_center_vec = np.asarray(n_src_center_vec, dtype=float).reshape(3)
            n_src_center_vec = n_src_center_vec / np.linalg.norm(n_src_center_vec)
    else:
        # For per-source optimization, use default center direction if not specified
        if theta_array is None or phi_array is None:
            # Use a default direction, e.g., overhead
            theta_array = 0.01
            phi_array = 0.0
        n_src_center_vec = spherical_to_cartesian(theta_array, phi_array)
        n_src_center_vec = np.asarray(n_src_center_vec, dtype=float).reshape(3)
        n_src_center_vec = n_src_center_vec / np.linalg.norm(n_src_center_vec)

    detector_position = -config.R * n_src_center_vec

    return ArrayContext(
        num_sources=num_sources,
        spacing=spacing,
        layout=choose_lattice_dimensions(num_sources),
        config=config,
        reference_geometry=reference_geometry,
        n_src_center_vec=n_src_center_vec,
        detector_position=detector_position,
        # The single source rotates at omega, but its quadrupole radiation is at
        # 2*omega. That is the phase that must be matched at the detector.
        gw_angular_frequency=2.0 * config.omega,
        optimize_each_source=optimize_each_source,
    )


def _positions_for_index_range(
    start: int,
    stop: int,
    layout: tuple[int, int, int],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat source indices into centered Cartesian lattice coordinates.

    The index mapping is:
    source_id -> (ix, iy, iz) -> (x, y, z)
    where the coordinates are centered so the array midpoint sits at (0, 0, 0).
    """
    nx, ny, _ = layout
    indices = np.arange(start, stop, dtype=np.int64)
    ix = indices % nx
    iy = (indices // nx) % ny
    iz = indices // (nx * ny)

    positions = np.column_stack(
        [
            (ix - (nx - 1) / 2.0) * spacing,
            (iy - (ny - 1) / 2.0) * spacing,
            (iz - ((layout[2] - 1) / 2.0)) * spacing,
        ]
    )
    return indices, positions


def _build_chunk(context: ArrayContext, start: int, stop: int) -> np.ndarray:
    """
    Build one chunk of the source table.

    Each row contains:
    - the ID of the souce,
    - source position in the array frame,
    - actual distance and line of sight to the detector,
    - the four detector-frame orientation angles for that lattice site,
    - propagation-delay correction,
    - GW phase and rotor phase needed for coherent arrival.

    **Important modeling note (post-refactor).** The interferometer arms are fixed
    globally along ``+x`` and ``+y`` in ``metricCalculate.py``. Off-center sources
    therefore do **not** change the arm directions; only the rotor axis is rigidly
    transported with the changing line of sight, matching the idea that the rotor is
    a body-fixed object viewed from a moving vantage point.
    """
    indices, positions = _positions_for_index_range(start, stop, context.layout, context.spacing)

    # The detector vertex is fixed in the array frame. Every source sees a slightly
    # different line of sight because it is displaced from the center.
    detector_vectors = context.detector_position - positions
    distances = np.linalg.norm(detector_vectors, axis=1)
    u_src_to_detector = detector_vectors / distances[:, None]

    # Detector -> source unit vector is the opposite of source -> detector.
    n_det_to_src = -u_src_to_detector
    theta_src, phi_src = cartesian_to_spherical(n_det_to_src)

    if context.optimize_each_source:
        # Optimize geometry for each source individually
        theta_rot_list = []
        phi_rot_list = []
        for i in range(len(indices)):
            # Use the local theta_src and phi_src as starting point
            initial_theta_src = theta_src[i]
            initial_phi_src = phi_src[i]
            initial_theta_rot = 1.0  # Default starting point
            initial_phi_rot = 0.0
            angles = bestPosition.scipy_gradient_descent(
                bestPosition.scaled_spherical_function,
                initial_theta_src,
                initial_phi_src,
                initial_theta_rot,
                initial_phi_rot,
                fix_source_angles=True,
            )
            _, _, theta_rot, phi_rot = angles
            theta_rot_list.append(theta_rot)
            phi_rot_list.append(phi_rot)
        theta_rot = np.array(theta_rot_list)
        phi_rot = np.array(phi_rot_list)
    else:
        # Rigidly map the optimized rotor axis onto each local line of sight.
        rot_axis_local = rotate_reference_vector(
            context.reference_geometry.rot_axis_vec,
            context.reference_geometry.u_src_to_detector_center_vec,
            u_src_to_detector,
        )
        theta_rot, phi_rot = cartesian_to_spherical(rot_axis_local)

    # Extra propagation time relative to the center source.
    distance_offset = distances - context.config.R
    propagation_compensation = distance_offset / context.config.c

    # Phase seen at the detector:
    #   phi_GW = Omega_GW * Delta t
    # where Omega_GW = 2 * omega_rotor for a quadrupole source.
    gw_phase_offset = context.gw_angular_frequency * propagation_compensation

    # If the user wants to preset the *mechanical* phase of each rotor instead
    # of the emitted GW phase directly, the needed rotor phase is half of the
    # GW phase because the radiation oscillates at twice the rotation rate.
    rotor_phase_offset = 0.5 * gw_phase_offset

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


def iter_source_chunks(
    context: ArrayContext,
    chunk_size: int = 100_000,
) -> Iterator[np.ndarray]:
    """
    Yield the full source table in manageable chunks.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    for start in range(0, context.num_sources, chunk_size):
        stop = min(start + chunk_size, context.num_sources)
        yield _build_chunk(context, start, stop)


def construct_source_array(
    num: int,
    theta_array: float | None = None,
    phi_array: float | None = None,
    spacing: float | None = None,
    recompute_best_position: bool = False,
    optimize_each_source: bool = False,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """
    Build the source array in memory.

    For very large arrays, prefer `write_source_array_csv`, which streams to disk
    in chunks instead of holding the entire array in RAM.

    This function is convenient for analysis notebooks and small tests.
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
    )

    chunks = [chunk for chunk in iter_source_chunks(context, chunk_size=chunk_size)]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=SOURCE_ARRAY_DTYPE)


def write_source_array_csv(
    output_path: str | Path,
    num_sources: int,
    theta_array: float | None = None,
    phi_array: float | None = None,
    spacing: float | None = None,
    recompute_best_position: bool = False,
    optimize_each_source: bool = False,
    chunk_size: int = 100_000,
) -> ArrayContext:
    """
    Stream the source array table to a CSV file.

    This is the intended path for large arrays such as 10^7 sources.
    """
    context = build_array_context(
        num_sources=num_sources,
        spacing=spacing,
        theta_array=theta_array,
        phi_array=phi_array,
        recompute_best_position=recompute_best_position,
        optimize_each_source=optimize_each_source,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_ARRAY_DTYPE.names)
        for chunk in iter_source_chunks(context, chunk_size=chunk_size):
            rows = zip(
                chunk["source_id"],
                chunk["x_m"],
                chunk["y_m"],
                chunk["z_m"],
                chunk["distance_to_detector_m"],
                chunk["distance_offset_m"],
                chunk["propagation_compensation_s"],
                chunk["theta_src"],
                chunk["phi_src"],
                chunk["theta_rot"],
                chunk["phi_rot"],
                chunk["gw_phase_offset_rad"],
                chunk["rotor_phase_offset_rad"],
            )
            writer.writerows(rows)

    return context


def _format_vector(vector: np.ndarray) -> str:
    return ", ".join(f"{component:.6f}" for component in vector)


def _print_summary(context: ArrayContext, preview_rows: np.ndarray | None = None) -> None:
    """
    Print a compact human-readable summary before or instead of CSV output.
    """
    theta_center, phi_center = cartesian_to_spherical(context.n_src_center_vec)
    if context.reference_geometry is not None:
        print(
            "Reference best-position amplitude = "
            f"{context.reference_geometry.signal_amplitude:.6e}"
        )
        print(
            "Reference rotor axis (detector frame) = "
            f"({_format_vector(context.reference_geometry.rot_axis_vec)})"
        )
    else:
        print("Optimizing geometry for each source individually.")
    print(
        "Array layout (nx, ny, nz) = "
        f"{context.layout}, spacing = {context.spacing:.3f} m, sources = {context.num_sources}"
    )
    print(
        "Array-center direction (det -> src) = "
        f"(theta={theta_center[0]:.8f}, phi={phi_center[0]:.8f})"
    )
    print(f"Detector vertex position [m] = ({_format_vector(context.detector_position)})")

    if preview_rows is not None and preview_rows.size > 0:
        print("\nPreview rows:")
        for row in preview_rows:
            print(
                "source_id={source_id}, pos=({x_m:.3f}, {y_m:.3f}, {z_m:.3f}), "
                "gw_phase={gw_phase_offset_rad:.6e}, rotor_phase={rotor_phase_offset_rad:.6e}".format(
                    **{name: row[name] for name in row.dtype.names}
                )
            )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options for array generation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Construct a coherent source array around the best geometry from "
            "bestPosition.py and compute the phase offset for each source."
        )
    )
    parser.add_argument(
        "--num-sources",
        type = int,
        default = 10000000,
        help="Total number of sources in the array. Default: 10000000.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=ExperimentConfig.D * 3/2,
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
        default =_DEFAULT_OUTPUT_FILE,
        help=f"CSV output path. Default: {_DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
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
        help="Run bestPosition optimization for each source individually (very slow for large arrays).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the geometry summary and preview rows without writing the full CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.

    Typical usage:
        python sourceArray.py --num-sources 10000000 --output data/source_array_distribution.csv
    """
    args = parse_arguments()

    context = build_array_context(
        num_sources=args.num_sources,
        spacing=args.spacing,
        theta_array=args.theta_array,
        phi_array=args.phi_array,
        recompute_best_position=args.recompute_best_position,
        optimize_each_source=args.optimize_each_source,
    )

    preview_count = min(args.preview, args.num_sources)
    preview_rows = _build_chunk(context, 0, preview_count) if preview_count > 0 else None
    _print_summary(context, preview_rows)

    if args.summary_only:
        return

    write_source_array_csv(
        output_path=args.output,
        num_sources=args.num_sources,
        theta_array=args.theta_array,
        phi_array=args.phi_array,
        spacing=args.spacing,
        recompute_best_position=False,
        optimize_each_source=args.optimize_each_source,
        chunk_size=args.chunk_size,
    )
    print(f"\nSaved source array table to {args.output}")


if __name__ == "__main__":
    main()
