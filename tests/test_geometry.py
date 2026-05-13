from __future__ import annotations

import numpy as np

from ghe.geometry import (
    cartesian_to_spherical,
    rotate_reference_vector,
    rotation_body_to_detector,
    spherical_to_cartesian,
    spherical_unit_vector,
)


def test_spherical_unit_vector_norm() -> None:
    vector = spherical_unit_vector(1.2, 2.3)
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_spherical_round_trip() -> None:
    theta = np.array([0.2, 1.1, 2.4])
    phi = np.array([0.3, 2.2, 5.9])
    vectors = spherical_to_cartesian(theta, phi)
    theta_out, phi_out = cartesian_to_spherical(vectors)
    assert np.allclose(theta_out, theta)
    assert np.allclose(phi_out, phi)


def test_rotation_matrix_orthogonality() -> None:
    rotation = rotation_body_to_detector(1.4, 0.7)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)


def test_vector_transport_preserves_norm() -> None:
    reference_vector = spherical_unit_vector(1.0, 0.5)
    reference_direction = spherical_unit_vector(0.7, 1.1)
    target_directions = spherical_to_cartesian(np.array([0.4, 1.8]), np.array([0.2, 3.1]))
    transported = rotate_reference_vector(reference_vector, reference_direction, target_directions)
    assert np.allclose(np.linalg.norm(transported, axis=1), 1.0)
