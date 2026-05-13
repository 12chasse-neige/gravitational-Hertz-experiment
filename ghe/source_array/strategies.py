"""
Rotor-angle and phase strategies for source-array generation.

The array generator supports three strategies:

``exact``
    Optimize the rotor axis for every source location.
``rigid``
    Rigidly transport the reference rotor axis to each local line of sight.
``chunk_anchor``
    Optimize one center source per group, then transport that local anchor to
    nearby sources and approximate their phase by relative distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ghe.geometry import cartesian_to_spherical, rotate_reference_vector
from ghe.optimization import scaled_spherical_function, scipy_gradient_descent

from .layout import positions_for_index_range
from .phase import get_signal_amplitude_and_phase, wrap_phase

if TYPE_CHECKING:
    from .generation import ArrayContext


def optimize_rotor_for_fixed_source(
    context: "ArrayContext",
    theta_src: float,
    phi_src: float,
) -> tuple[float, float]:
    """
    Optimize only the rotor-axis angles for a fixed source sky direction.

    The source position already fixes ``theta_src`` and ``phi_src``.  This helper
    is used by both exact generation and chunk-anchor centers.
    """

    _, _, theta_rot, phi_rot = scipy_gradient_descent(
        scaled_spherical_function,
        theta_src,
        phi_src,
        context.reference_rotor_theta,
        context.reference_rotor_phi,
        fix_source_angles=True,
    )
    return float(theta_rot), float(phi_rot)


def exact_anchor_for_source_index(
    context: "ArrayContext",
    source_index: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute an exact anchor for one source index.

    Returns distance, GW phase offset, optimized rotor-axis vector, and the local
    source-to-detector unit vector.  Chunk-anchor generation reuses these values
    for neighboring sources.
    """

    _, center_positions = positions_for_index_range(
        source_index,
        source_index + 1,
        context.layout,
        context.spacing,
    )
    detector_vector = context.detector_position - center_positions[0]
    distance = float(np.linalg.norm(detector_vector))
    u_src_to_detector = detector_vector / distance
    theta_src_arr, phi_src_arr = cartesian_to_spherical(-u_src_to_detector)
    theta_src = float(theta_src_arr[0])
    phi_src = float(phi_src_arr[0])
    theta_rot, phi_rot = optimize_rotor_for_fixed_source(context, theta_src, phi_src)
    _, signal_phase = get_signal_amplitude_and_phase(
        theta_src,
        phi_src,
        theta_rot,
        phi_rot,
        distance,
        config=context.config,
    )
    gw_phase_offset = wrap_phase(signal_phase - context.reference_signal_phase)
    rot_axis_vec = context.geometry_vector(theta_rot, phi_rot)
    return distance, float(gw_phase_offset), rot_axis_vec, u_src_to_detector


def approximate_chunk_center_parameters(
    context: "ArrayContext",
    indices: np.ndarray,
    distances: np.ndarray,
    u_src_to_detector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate source parameters using one exact anchor per index group.

    Each group pays for one exact rotor optimization and phase extraction.  Other
    rows in the group rigidly transport the anchor rotor axis and adjust phase by
    the distance difference to the detector.
    """

    theta_rot = np.empty(len(indices), dtype=float)
    phi_rot = np.empty(len(indices), dtype=float)
    gw_phase_offset = np.empty(len(indices), dtype=float)
    group_size = context.approximation_chunk_size

    cursor = 0
    while cursor < len(indices):
        source_index = int(indices[cursor])
        group_start = (source_index // group_size) * group_size
        group_stop = min(group_start + group_size, context.num_sources)
        local_stop = min(len(indices), cursor + group_stop - source_index)
        center_index = group_start + (group_stop - group_start) // 2

        center_distance, center_gw_phase_offset, center_rot_axis, center_u_src_to_detector = (
            exact_anchor_for_source_index(context, center_index)
        )

        local_slice = slice(cursor, local_stop)
        local_rot_axis = rotate_reference_vector(
            center_rot_axis,
            center_u_src_to_detector,
            u_src_to_detector[local_slice],
        )
        theta_rot[local_slice], phi_rot[local_slice] = cartesian_to_spherical(local_rot_axis)

        relative_time = (distances[local_slice] - center_distance) / context.config.c
        gw_phase_offset[local_slice] = wrap_phase(
            center_gw_phase_offset + context.gw_angular_frequency * relative_time
        )
        cursor = local_stop

    return theta_rot, phi_rot, gw_phase_offset


def rigid_rotor_angles(
    context: "ArrayContext",
    u_src_to_detector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return rotor angles from rigid transport of the reference rotor axis.

    This is the fastest strategy.  It preserves local orientation continuity but
    does not re-optimize each source's rotor for its exact sky direction.
    """

    rot_axis_local = rotate_reference_vector(
        context.reference_geometry.rot_axis_vec,
        context.reference_geometry.u_src_to_detector_center_vec,
        u_src_to_detector,
    )
    return cartesian_to_spherical(rot_axis_local)
