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
``data/bestPosition.txt`` default lookup lives only in ``scr.metricCalculate``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import SourceConfig
from .geometry import rotation_body_to_detector, spherical_unit_vector

_GL_ORDER = 7
_GL_NODES, _GL_WEIGHTS = leggauss(_GL_ORDER)
_gl_quadrature_cache: dict[float, Tuple[np.ndarray, np.ndarray]] = {}


def _get_quadrature(L: float) -> Tuple[np.ndarray, np.ndarray]:
    key = round(L * 1e12)
    if key not in _gl_quadrature_cache:
        x_nodes = 0.5 * L * (_GL_NODES + 1.0)
        w_scaled = 0.5 * L * _GL_WEIGHTS
        _gl_quadrature_cache[key] = (x_nodes, w_scaled)
    return _gl_quadrature_cache[key]


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
    Return the exact second time derivative d^2 I_ij / dt^2.

    Each hole is modeled as a negative point mass moving on a circle.
    For r(t) ⊗ r(t):

        d²(r ⊗ r)/dt²
        = a ⊗ r + 2 v ⊗ v + r ⊗ a

    The trace-subtraction term has zero second derivative because
    r² = s² is constant for circular motion.
    """

    tensor_ddot = np.zeros((3, 3), dtype=float)

    volume = np.pi * config.d**2 / 4.0 * config.H
    mass = -config.rho * volume

    for k in range(config.num):
        x, y = get_hole_coordinate(k, t, config)

        position = np.array([x, y, 0.0], dtype=float)
        velocity = config.omega * np.array([-y, x, 0.0], dtype=float)
        acceleration = -(config.omega**2) * position

        tensor_ddot += mass * (
            np.outer(acceleration, position)
            + 2.0 * np.outer(velocity, velocity)
            + np.outer(position, acceleration)
        )

    return tensor_ddot


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


def _integrate_arm_response(
    t: float,
    n_src_to_det: np.ndarray,
    a_vec: np.ndarray,
    config: SourceConfig,
    R_body_to_det: np.ndarray,
    *,
    forward: bool,
) -> float:
    """Fixed Gauss-Legendre quadrature along one arm for forward or return trip."""

    x_nodes, w_scaled = _get_quadrature(config.L)
    R_scalar = config.R
    result = 0.0

    for x, w in zip(x_nodes, w_scaled):
        r_vec = x * a_vec - R_scalar * n_src_to_det
        r_distance = np.linalg.norm(r_vec)
        if forward:
            photon_time = t + x / config.c
        else:
            photon_time = t + (config.L - x) / config.c

        h_body = get_metric_tensor_body_frame(r_distance, photon_time, config)
        h_det = R_body_to_det @ h_body @ R_body_to_det.T
        h_tt = project_to_tt_gauge_dynamic(h_det, r_vec)
        result += w * float((a_vec.T @ h_tt @ a_vec) / (2.0 * config.c))

    return float(result)


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
    return _integrate_arm_response(t, n_src_to_det, a_vec, config, R_body_to_det, forward=True)


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
    return _integrate_arm_response(t, n_src_to_det, a_vec, config, R_body_to_det, forward=False)


def _calculate_metric_response_prepared(
    t: float,
    n_src_to_det: np.ndarray,
    R_body_to_det: np.ndarray,
    config: SourceConfig,
) -> float:
    """
    Compute detector strain with pre-computed geometry vectors.

    The caller already provides the source-to-detector direction and the body-to-
    detector rotation matrix so they are not recomputed per time sample.
    """

    a_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    b_vec = np.array([0.0, 1.0, 0.0], dtype=float)

    t_forward = t - 2.0 * config.L / config.c
    t_return = t - config.L / config.c

    delay_1 = _integrate_arm_response(t_forward, n_src_to_det, a_vec, config, R_body_to_det, forward=True)
    delay_1 += _integrate_arm_response(t_return, n_src_to_det, a_vec, config, R_body_to_det, forward=False)
    delay_2 = _integrate_arm_response(t_forward, n_src_to_det, b_vec, config, R_body_to_det, forward=True)
    delay_2 += _integrate_arm_response(t_return, n_src_to_det, b_vec, config, R_body_to_det, forward=False)

    return float((delay_1 - delay_2) * config.c / (2.0 * config.L))


def calculate_metric_response(
    t: float,
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
    R: float | None = None,
    _n_src_to_det: np.ndarray | None = None,
    _R_body_to_det: np.ndarray | None = None,
) -> float:
    """
    Compute the detector strain response for explicit source and rotor geometry.

    Parameters are detector-frame spherical angles:
    ``(theta_src, phi_src)`` points from the detector vertex toward the source;
    ``(theta_rot, phi_rot)`` points along the rotor body ``+z`` axis.

    The output is the Michelson-style differential response:
    ``(delay_arm_x - delay_arm_y) * c / (2L)``.

    The legacy best-position default lookup is intentionally kept in
    ``scr.metricCalculate``.

    Internal keyword arguments ``_n_src_to_det`` and ``_R_body_to_det`` allow
    callers that already have these vectors to skip recomputation.
    """

    active_config = config or SourceConfig()
    if R is not None:
        active_config = replace(active_config, R=float(R))

    if _n_src_to_det is not None and _R_body_to_det is not None:
        return _calculate_metric_response_prepared(
            t, _n_src_to_det, _R_body_to_det, active_config,
        )

    n_src_to_det = spherical_unit_vector(theta_src, phi_src)
    R_body_to_det = rotation_body_to_detector(theta_rot, phi_rot)
    return _calculate_metric_response_prepared(
        t, n_src_to_det, R_body_to_det, active_config,
    )
