"""
Coordinate helpers shared by metric and source-array code.

The project uses one detector-frame convention everywhere:

- Origin at the interferometer vertex.
- ``+x`` is arm 1.
- ``+y`` is arm 2.
- ``+z`` completes the right-handed frame.
- ``theta`` is polar angle from ``+z``.
- ``phi`` is azimuth from ``+x`` toward ``+y``.
"""

from __future__ import annotations

import numpy as np


def spherical_unit_vector(theta: float, phi: float) -> np.ndarray:
    """
    Convert detector-frame spherical angles to a unit vector.

    ``theta`` is the polar angle from ``+z`` and ``phi`` is the azimuth from ``+x``
    toward ``+y``.
    """

    return np.array(
        [
            float(np.sin(theta) * np.cos(phi)),
            float(np.sin(theta) * np.sin(phi)),
            float(np.cos(theta)),
        ],
        dtype=float,
    )


def spherical_to_cartesian(theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
    """Vectorized spherical-to-Cartesian conversion using the project convention."""

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
    """Convert one or more Cartesian vectors into ``(theta, phi)`` arrays."""

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]

    radii = np.linalg.norm(vectors, axis=1)
    if np.any(radii == 0.0):
        raise ValueError("Cannot convert zero-length vector to spherical angles.")

    theta = np.arccos(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    phi = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
    return theta, phi


def rotation_body_to_detector(theta_rot: float, phi_rot: float) -> np.ndarray:
    """
    Build an orthonormal matrix mapping body-frame coordinates to detector frame.

    The body ``+z`` axis is the rotor symmetry axis specified by
    ``(theta_rot, phi_rot)``.
    """

    # The body z-axis is physically meaningful: it is the rotor symmetry axis.
    # Body x/y axes are any orthonormal completion in the plane perpendicular to z.
    z_hat = spherical_unit_vector(theta_rot, phi_rot)
    z_hat = z_hat / np.linalg.norm(z_hat)

    if abs(float(z_hat[2])) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)

    x_hat = np.cross(z_hat, tmp)
    x_hat = x_hat / np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    return np.column_stack([x_hat, y_hat, z_hat])


def _get_perpendicular_axis(vector: np.ndarray) -> np.ndarray:
    """Return a stable perpendicular axis for the 180-degree rotation case."""

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
    Rigidly transport ``reference_vector`` as ``reference_direction`` maps to targets.

    The minimal rotation is used for each target direction, preserving vector norm.
    Source-array rigid mode uses this to move a reference rotor axis to each
    source's local line of sight.
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
        # No rotation needed.
        rotated[same_mask] = u[same_mask]

    if np.any(general_mask):
        v = cross_term[general_mask]
        u_general = u[general_mask]
        correction = np.cross(v, np.cross(v, u_general)) / (1.0 + dot_term[general_mask])[:, None]
        rotated[general_mask] = u_general + np.cross(v, u_general) + correction

    if np.any(opposite_mask):
        # Minimal rotation is undefined for exactly opposite vectors; choose any
        # stable axis perpendicular to the reference direction.
        axis = _get_perpendicular_axis(reference_direction)
        u_opposite = u[opposite_mask]
        rotated[opposite_mask] = -u_opposite + 2.0 * np.outer(u_opposite @ axis, axis)

    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    return rotated
