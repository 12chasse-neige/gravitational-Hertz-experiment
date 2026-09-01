"""Direct near-field metric integral for one rotating, perforated rotor.

This module evaluates the volume integral supplied in the project derivation,
without expanding ``1 / |x - y|`` in source multipoles.  The rotor is modeled as
a solid cylinder minus the cylindrical holes already defined by
:class:`ghe.config.SourceConfig`.

The material energy-momentum tensor is the moving-rest-mass contribution.  In
particular, the current numerical core does not contain an elasticity solution
for the perforated rotor, so material stresses are not included here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import SourceConfig
from .geometry import rotation_body_to_detector, spherical_unit_vector
from .metric import get_metric_tensor_body_frame


@dataclass(frozen=True)
class NearFieldMetricResult:
    """Result and geometry for the direct single-source volume integral."""

    observer_time_s: float
    retarded_time_s: float
    detector_position_from_rotor_com_m: np.ndarray
    body_to_detector_rotation: np.ndarray
    h_covariant: np.ndarray
    rotor_rest_mass_kg: float
    quadrature_volume_m3: float


def rotor_rest_volume(config: SourceConfig) -> float:
    """Return the material volume of the cylinder after subtracting its holes."""

    rotor_radius = 0.5 * config.D
    hole_radius = 0.5 * config.d
    return float(np.pi * config.H * (rotor_radius**2 - config.num * hole_radius**2))


def calculate_quadrupole_gw_metric(
    time_s: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
) -> np.ndarray:
    r"""Return the quadrupole-only comparison tensor in the detector frame.

    This evaluates

    ``h_ij = (2 G / c^4 r) d^2 I_ij(t-r/c) / dt^2``

    using the existing single-source quadrupole core.  The result is embedded in
    a 4-by-4 tensor whose time row and column are zero.  No TT projection is
    applied, matching the formula above.
    """

    active_config = config or SourceConfig()
    body_to_detector = rotation_body_to_detector(theta_rot, phi_rot)
    h_body = get_metric_tensor_body_frame(active_config.R, time_s, active_config)
    h_spatial = body_to_detector @ h_body @ body_to_detector.T

    h_covariant = np.zeros((4, 4), dtype=float)
    h_covariant[1:, 1:] = 0.5 * (h_spatial + h_spatial.T)
    return h_covariant


def _validate_inputs(
    config: SourceConfig,
    radial_order: int,
    azimuthal_order: int,
    axial_order: int,
) -> None:
    if config.num < 1:
        raise ValueError("SourceConfig.num must be at least one")
    if min(config.H, config.D, config.d, config.R, config.rho, config.c) <= 0.0:
        raise ValueError("rotor dimensions, distance, density, and c must be positive")

    rotor_radius = 0.5 * config.D
    hole_radius = 0.5 * config.d
    if config.s < 0.0 or config.s + hole_radius > rotor_radius:
        raise ValueError("every hole must lie entirely inside the rotor")
    if config.num > 1:
        nearest_center_distance = 2.0 * config.s * np.sin(np.pi / config.num)
        if nearest_center_distance < config.d:
            raise ValueError("rotor holes must not overlap")
    if abs(config.omega) * rotor_radius >= config.c:
        raise ValueError("the rotor rim speed must be below the speed of light")

    if radial_order < 2 or axial_order < 2 or azimuthal_order < 4:
        raise ValueError("quadrature orders must be >= 2 radial/axial and >= 4 azimuthal")


def _cylinder_quadrature(
    radius: float,
    height: float,
    center_x: float,
    center_y: float,
    radial_order: int,
    azimuthal_order: int,
    axial_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-frame points and volume weights for one cylinder."""

    radial_nodes, radial_weights = leggauss(radial_order)
    axial_nodes, axial_weights = leggauss(axial_order)

    radii = 0.5 * radius * (radial_nodes + 1.0)
    radial_weights = 0.5 * radius * radial_weights * radii
    z_values = 0.5 * height * axial_nodes
    axial_weights = 0.5 * height * axial_weights
    azimuths = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    azimuth_weight = 2.0 * np.pi / azimuthal_order

    rr, pp, zz = np.meshgrid(radii, azimuths, z_values, indexing="ij")
    points = np.column_stack(
        (
            center_x + (rr * np.cos(pp)).ravel(),
            center_y + (rr * np.sin(pp)).ravel(),
            zz.ravel(),
        )
    )
    weights = (
        radial_weights[:, None, None]
        * np.full((1, azimuthal_order, 1), azimuth_weight)
        * axial_weights[None, None, :]
    ).ravel()
    return points, weights


def _integrate_cylinder(
    *,
    radius: float,
    center_x: float,
    center_y: float,
    sign: float,
    spin_phase: float,
    detector_position: np.ndarray,
    body_to_detector: np.ndarray,
    config: SourceConfig,
    radial_order: int,
    azimuthal_order: int,
    axial_order: int,
) -> tuple[np.ndarray, float]:
    points, weights = _cylinder_quadrature(
        radius,
        config.H,
        center_x,
        center_y,
        radial_order,
        azimuthal_order,
        axial_order,
    )

    cos_phase = np.cos(spin_phase)
    sin_phase = np.sin(spin_phase)
    spin_rotation = np.array(
        [
            [cos_phase, -sin_phase, 0.0],
            [sin_phase, cos_phase, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points_body = points @ spin_rotation.T
    points_detector = points_body @ body_to_detector.T

    velocity_body = np.column_stack(
        (
            -config.omega * points_body[:, 1],
            config.omega * points_body[:, 0],
            np.zeros(points_body.shape[0]),
        )
    )
    velocity_detector = velocity_body @ body_to_detector.T
    speed_squared = np.einsum("ij,ij->i", velocity_detector, velocity_detector)
    gamma = 1.0 / np.sqrt(1.0 - speed_squared / config.c**2)

    separation = detector_position[None, :] - points_detector
    distances = np.linalg.norm(separation, axis=1)
    if np.any(distances == 0.0):
        raise ValueError("the detector position coincides with a quadrature point in the rotor")

    four_velocity_factor = np.column_stack(
        (np.full(points.shape[0], config.c), velocity_detector)
    )
    weighted_density = sign * config.rho * weights * gamma / distances
    integral_contravariant = np.einsum(
        "n,ni,nj->ij",
        weighted_density,
        four_velocity_factor,
        four_velocity_factor,
    )
    return integral_contravariant, float(sign * np.sum(weights))


def calculate_near_field_metric(
    time_s: float,
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
    radial_order: int = 16,
    azimuthal_order: int = 64,
    axial_order: int = 16,
) -> NearFieldMetricResult:
    r"""Evaluate the supplied direct integral at the detector vertex.

    The detector-frame source direction points from the vertex to the rotor COM,
    exactly as it does in :mod:`ghe.metric`.  Thus the vector ``x`` from the
    rotor COM to the detector is ``-R * n_src``.

    The common source time is ``time_s - |x|/c``, as specified in the requested
    formula.  The spatial denominator retains its full ``|x-y|`` dependence.
    The returned matrix has lower indices for signature ``(-,+,+,+)``.
    """

    active_config = config or SourceConfig()
    _validate_inputs(active_config, radial_order, azimuthal_order, axial_order)

    source_direction = spherical_unit_vector(theta_src, phi_src)
    detector_position = -active_config.R * source_direction
    body_to_detector = rotation_body_to_detector(theta_rot, phi_rot)
    retarded_time = float(time_s - np.linalg.norm(detector_position) / active_config.c)
    spin_phase = active_config.omega * retarded_time

    rotor_radius = 0.5 * active_config.D
    hole_radius = 0.5 * active_config.d
    integral, quadrature_volume = _integrate_cylinder(
        radius=rotor_radius,
        center_x=0.0,
        center_y=0.0,
        sign=1.0,
        spin_phase=spin_phase,
        detector_position=detector_position,
        body_to_detector=body_to_detector,
        config=active_config,
        radial_order=radial_order,
        azimuthal_order=azimuthal_order,
        axial_order=axial_order,
    )

    for hole_index in range(active_config.num):
        hole_phase = 2.0 * np.pi * hole_index / active_config.num
        hole_integral, hole_volume = _integrate_cylinder(
            radius=hole_radius,
            center_x=active_config.s * np.cos(hole_phase),
            center_y=active_config.s * np.sin(hole_phase),
            sign=-1.0,
            spin_phase=spin_phase,
            detector_position=detector_position,
            body_to_detector=body_to_detector,
            config=active_config,
            radial_order=radial_order,
            azimuthal_order=azimuthal_order,
            axial_order=axial_order,
        )
        integral += hole_integral
        quadrature_volume += hole_volume

    coefficient = 4.0 * active_config.G / active_config.c**4
    h_contravariant = coefficient * integral
    lower_index_signs = np.array([-1.0, 1.0, 1.0, 1.0])
    h_covariant = (
        lower_index_signs[:, None]
        * h_contravariant
        * lower_index_signs[None, :]
    )
    # T_mu_nu is symmetric analytically. Remove insignificant accumulation-order
    # roundoff so the serialized tensor preserves that invariant exactly.
    h_covariant = 0.5 * (h_covariant + h_covariant.T)

    return NearFieldMetricResult(
        observer_time_s=float(time_s),
        retarded_time_s=retarded_time,
        detector_position_from_rotor_com_m=detector_position,
        body_to_detector_rotation=body_to_detector,
        h_covariant=h_covariant,
        rotor_rest_mass_kg=active_config.rho * rotor_rest_volume(active_config),
        quadrature_volume_m3=quadrature_volume,
    )


__all__ = [
    "NearFieldMetricResult",
    "calculate_near_field_metric",
    "calculate_quadrupole_gw_metric",
    "rotor_rest_volume",
]
