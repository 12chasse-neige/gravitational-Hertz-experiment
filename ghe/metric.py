"""
Single-source gravitational signal and detector response.

The calculation follows the physical model in ``docs/theoreticalDerivation.md``:

1. Model the rotating holes as a negative-mass quadrupole in the source body
   frame.
2. Take the second time derivative of the quadrupole tensor.
3. Convert it to a weak metric perturbation at retarded time.
4. Rotate the body-frame tensor into the detector frame.
5. Project the tensor into TT gauge for the local source-to-arm direction.
6. Integrate the induced light-travel-time delay along each interferometer arm.
7. Return the differential arm response as dimensionless strain.

This module deliberately requires explicit source and rotor angles.  The legacy
``data/bestPosition.txt`` default lookup lives only in ``scripts.metricCalculate``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.integrate import quad

from .config import SourceConfig
from .geometry import rotation_body_to_detector, spherical_unit_vector


def get_hole_coordinate(k: int, t: float, config: SourceConfig) -> tuple[float, float]:
    """
    Return the body-frame position of one hole center at time ``t``.

    The holes rotate in the body ``x-y`` plane with angular velocity
    ``config.omega``.  ``k`` indexes equally spaced holes around the rotor.
    """

    x_k = config.s * np.cos(config.omega * t + k * (2 * np.pi / config.num))
    y_k = config.s * np.sin(config.omega * t + k * (2 * np.pi / config.num))
    return float(x_k), float(y_k)


def calculate_whole_tensor(t: float, config: SourceConfig) -> np.ndarray:
    """
    Build the traceless quadrupole tensor ``I_ij`` in the source body frame.

    Holes are represented as missing mass, hence the negative mass contribution.
    The tensor is still in the frame where the rotor symmetry axis is ``+z``.
    """

    tensor = np.zeros((3, 3))
    volume = np.pi * config.d**2 / 4.0 * config.H
    mass = -config.rho * volume

    for k in range(config.num):
        x, y = get_hole_coordinate(k, t, config)
        coords = np.array([x, y, 0.0])
        r_squared = x**2 + y**2
        tensor += mass * (np.outer(coords, coords) - (1.0 / 3.0) * np.eye(3) * r_squared)

    return tensor


def second_derivative_of_tensor(t: float, config: SourceConfig) -> np.ndarray:
    """
    Return ``d^2 I_ij / dt^2`` for the rotating quadrupole.

    For two opposite holes the non-constant tensor terms oscillate at ``2*omega``,
    so the second derivative is captured by multiplying by ``-4*omega**2``.
    """

    tensor = calculate_whole_tensor(t, config)
    return -4.0 * config.omega**2 * tensor


def get_metric_tensor_body_frame(r: float, t: float, config: SourceConfig) -> np.ndarray:
    """
    Convert quadrupole acceleration into the raw body-frame metric perturbation.

    ``r`` is the instantaneous source-to-field-point distance.  The tensor is
    evaluated at retarded time ``t - r/c``.
    """

    t_rev = t - r / config.c
    coeff = 2.0 * config.G / (r * config.c**4)
    return coeff * second_derivative_of_tensor(t_rev, config)


def project_to_tt_gauge_dynamic(h_matrix: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    """
    Project a metric tensor into transverse-traceless gauge.

    ``r_vec`` points from the source to the field point on the detector arm.  The
    projector changes along the arm because the source is near-field relative to
    the detector arm length.
    """

    r_norm = np.linalg.norm(r_vec)
    k = r_vec / r_norm
    P = np.eye(3) - np.outer(k, k)
    trace = np.sum(P * h_matrix)
    return P @ h_matrix @ P.T - 0.5 * P * trace


def calculate_delta_t(
    t: float,
    n_src_to_det: np.ndarray,
    a_vec: np.ndarray,
    config: SourceConfig,
    R_body_to_det: np.ndarray,
) -> float:
    """
    Integrate the forward-trip light delay along one detector arm.

    ``a_vec`` is the arm direction in detector coordinates.  For each arm
    coordinate ``x`` this evaluates the retarded metric at the photon location,
    projects it to TT gauge, and contracts it with the arm direction.
    """

    n_src_to_det = np.asarray(n_src_to_det, dtype=float)
    n_src_to_det = n_src_to_det / np.linalg.norm(n_src_to_det)
    R_body_to_det = np.asarray(R_body_to_det, dtype=float)

    def integrand(x: float) -> float:
        # Detector vertex is at the origin; the source is at R*n_src_to_det.
        # A photon at distance x along the arm has position x*a_vec, so this is
        # the source-to-photon separation vector used for retarded propagation.
        r_vec = x * a_vec - config.R * n_src_to_det
        r_distance = np.linalg.norm(r_vec)

        # Build h_ij in the source body frame, rotate components into detector
        # coordinates, then remove gauge components transverse to propagation.
        h_body = get_metric_tensor_body_frame(r_distance, t + x / config.c, config)
        h_det = R_body_to_det @ h_body @ R_body_to_det.T
        h_tt = project_to_tt_gauge_dynamic(h_det, r_vec)
        return float((a_vec.T @ h_tt @ a_vec) / (2.0 * config.c))

    result, _ = quad(integrand, 0.0, config.L)
    return float(result)


def calculate_delta_t_prime(
    t: float,
    n_src_to_det: np.ndarray,
    a_vec: np.ndarray,
    config: SourceConfig,
    R_body_to_det: np.ndarray,
) -> float:
    """
    Integrate the return-trip light delay along one detector arm.

    The geometry is the same as ``calculate_delta_t``.  Only the photon time
    argument differs because the light is traveling back from end mirror to
    vertex.
    """

    n_src_to_det = np.asarray(n_src_to_det, dtype=float)
    n_src_to_det = n_src_to_det / np.linalg.norm(n_src_to_det)
    R_body_to_det = np.asarray(R_body_to_det, dtype=float)

    def integrand(x: float) -> float:
        r_vec = x * a_vec - config.R * n_src_to_det
        r_distance = np.linalg.norm(r_vec)
        h_body = get_metric_tensor_body_frame(r_distance, t + (config.L - x) / config.c, config)
        h_det = R_body_to_det @ h_body @ R_body_to_det.T
        h_tt = project_to_tt_gauge_dynamic(h_det, r_vec)
        return float((a_vec.T @ h_tt @ a_vec) / (2.0 * config.c))

    result, _ = quad(integrand, 0.0, config.L)
    return float(result)


def calculate_metric_response(
    t: float,
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
    R: float | None = None,
) -> float:
    """
    Compute the detector strain response for explicit source and rotor geometry.

    Parameters are detector-frame spherical angles:
    ``(theta_src, phi_src)`` points from the detector vertex toward the source;
    ``(theta_rot, phi_rot)`` points along the rotor body ``+z`` axis.

    The output is the Michelson-style differential response:
    ``(delay_arm_x - delay_arm_y) * c / (2L)``.

    The legacy best-position default lookup is intentionally kept in
    ``scripts.metricCalculate``.
    """

    active_config = config or SourceConfig()
    if R is not None:
        active_config = replace(active_config, R=float(R))

    a_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    b_vec = np.array([0.0, 1.0, 0.0], dtype=float)

    n_src_to_det = spherical_unit_vector(theta_src, phi_src)
    R_body_to_det = rotation_body_to_detector(theta_rot, phi_rot)

    # The arm integrals use photon emission times.  A response observed at time t
    # contains a forward segment and a return segment shifted by light travel time.
    t_forward = t - 2.0 * active_config.L / active_config.c
    t_return = t - active_config.L / active_config.c

    delay_1 = calculate_delta_t(t_forward, n_src_to_det, a_vec, active_config, R_body_to_det)
    delay_1 += calculate_delta_t_prime(t_return, n_src_to_det, a_vec, active_config, R_body_to_det)
    delay_2 = calculate_delta_t(t_forward, n_src_to_det, b_vec, active_config, R_body_to_det)
    delay_2 += calculate_delta_t_prime(t_return, n_src_to_det, b_vec, active_config, R_body_to_det)

    return float((delay_1 - delay_2) * active_config.c / (2.0 * active_config.L))
