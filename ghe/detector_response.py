"""Equal-arm, free-mass Michelson integration of the exterior quadrupole field.

The vertex is at zero, arms point along +x/+y, and t is reception time at the
vertex. Production integrates curvature (near-field-analysis.md eqs. 31–32).
The independent harmonic-gauge implementation below retains moving endpoints
and light-path terms separately to verify their sum. Neither calculation is a
suspended Fabry–Perot cavity model. Inputs use SI units and PEAK exp(-iΩt) phasors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from .finite_distance import quadrupole_field, newtonian_acceleration


@dataclass(frozen=True)
class IntegrationSettings:
    """Numerical accuracy controls, independent of the physical approximation.

    Compare successive doubled Gauss–Legendre rules, starting at 48 points per
    arm. The error scale is the sum of absolute arm integrands, so a physical
    differential-response null does not cause meaningless relative errors.
    """

    initial_order: int = 48
    max_order: int = 768
    relative_tolerance: float = 1e-9

    def __post_init__(self):
        if (
            not isinstance(self.initial_order, int)
            or not isinstance(self.max_order, int)
            or self.initial_order < 4
            or self.max_order < 2 * self.initial_order
        ):
            raise ValueError(
                "Integration orders must be integers, initial >=4 and max >=2*initial"
            )
        if (
            not np.isfinite(self.relative_tolerance)
            or not 0 < self.relative_tolerance < 1
        ):
            raise ValueError("Integration tolerance must lie between zero and one")


@dataclass(frozen=True)
class DetectorResponse:
    """Converged dimensionless phasor and inspectable numerical error estimate."""

    phasor: complex
    order: int
    absolute_error: float
    integration_scale: float


@lru_cache(maxsize=32)
def unit_quadrature(order: int):
    """Immutable nodes/weights on [0,1]; cache is shared across geometries."""
    nodes, weights = leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def validate_clearance(source_position, arm_length, source_radius=0.0):
    """Reject an enclosing source sphere touching either closed arm segment.

    The exterior multipole field is invalid inside that sphere. Checking only
    quadrature nodes would miss a source between nodes or at an endpoint.
    """
    source = np.asarray(source_position, dtype=float)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("Source position must be a finite Cartesian three-vector")
    if (
        not np.isfinite(arm_length)
        or arm_length <= 0
        or not np.isfinite(source_radius)
        or source_radius < 0
    ):
        raise ValueError("Require positive arm length and nonnegative source radius")
    for direction in np.eye(3)[:2]:
        nearest = np.clip(source @ direction, 0, arm_length) * direction
        if np.linalg.norm(source - nearest) <= source_radius:
            raise ValueError("Source enclosing sphere intersects a detector light path")
    return source


def _curvature_integral(source, quadrupole, angular_frequency, arm_length, G, c, order):
    """Evaluate both arms in one batch; return differential signal and scale."""
    nodes, weights = unit_quadrature(order)
    distances = arm_length * nodes
    # Shape (2, order, 3): each row of points follows one unperturbed light ray.
    relative = distances[None, :, None] * np.eye(3)[:2, None, :] - source
    tidal = quadrupole_field(relative, quadrupole, angular_frequency, G=G, c=c)["tidal"]
    longitudinal = np.stack((tidal[0, :, 0, 0], tidal[1, :, 1, 1]))
    k = angular_frequency / c
    # Source retardation is already inside E through exp(ikr). These factors
    # ONLY sample outward and return photons at their earlier passage times.
    photon_weight = np.exp(1j * k * (2 * arm_length - distances)) + np.exp(
        1j * k * distances
    )
    # ds=L du cancels the 1/L normalization in eq.31, leaving 1/(2 Ω²).
    terms = weights * photon_weight * longitudinal / (2 * angular_frequency**2)
    return complex(np.sum(terms[0]) - np.sum(terms[1])), float(np.sum(np.abs(terms)))


def curvature_response(
    source_position,
    quadrupole,
    angular_frequency,
    *,
    arm_length,
    G,
    c,
    source_radius=0.0,
    settings=None,
) -> DetectorResponse:
    """Return the converged free-mass response; raise on invalid/nonconverged input.

    source_position: vertex-to-COM vector [m]; quadrupole: complex STF (3,3)
    moment [kg m²]; angular_frequency: positive signal frequency [rad/s].
    source_radius is the enclosing rotor radius [m], not its distance.
    """
    settings = settings or IntegrationSettings()
    source = validate_clearance(source_position, arm_length, source_radius)
    if not np.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("The periodic free-mass response requires positive frequency")
    order = settings.initial_order
    previous, _ = _curvature_integral(
        source, quadrupole, angular_frequency, arm_length, G, c, order
    )
    while order < settings.max_order:
        order = min(2 * order, settings.max_order)
        current, scale = _curvature_integral(
            source, quadrupole, angular_frequency, arm_length, G, c, order
        )
        error = abs(current - previous)
        if np.isfinite(error) and error <= settings.relative_tolerance * max(
            scale, np.finfo(float).tiny
        ):
            return DetectorResponse(current, order, error, scale)
        previous = current
    raise ArithmeticError(
        f"Arm integration did not converge by order {order}; refine settings or inspect geometry"
    )


def reference_michelson_response(
    source_position,
    quadrupole,
    angular_frequency,
    *,
    arm_length,
    G,
    c,
    quadrature_order=48,
):
    """Equal-arm point-mass Michelson, vertex at zero, arms along +x and +y.

    Both end mirrors and the common vertex follow free geodesics at the signal
    frequency. No L/R or Omega*L/c expansion is made. Two equivalent evaluations
    are returned: harmonic metric + moving endpoints, and synchronous-gauge
    curvature integration. The common vertex clock term cancels between arms.
    """
    if angular_frequency <= 0 or arm_length <= 0 or quadrature_order < 4:
        raise ValueError("Require positive frequency/length and quadrature order >=4")
    source = validate_clearance(source_position, arm_length)
    T = arm_length / c
    phase = np.exp(1j * angular_frequency * T)
    nodes, weights = leggauss(quadrature_order)
    nodes = (nodes + 1) * arm_length / 2
    weights = weights * arm_length / 2
    vertex = quadrupole_field(-source, quadrupole, angular_frequency, G=G, c=c)
    # With exp(-iΩt), twice differentiating displacement multiplies by -Ω².
    # This periodic particular solution excludes DC equilibrium and drift.
    vertex_displacement = -vertex["acceleration"] / angular_frequency**2
    endpoints = 0j
    path = 0j
    curvature = 0j
    newtonian_differential = 0j
    for axis, sign in [(0, 1), (1, -1)]:
        direction = np.eye(3)[axis]
        end_relative = arm_length * direction - source
        end = quadrupole_field(end_relative, quadrupole, angular_frequency, G=G, c=c)
        displacement = -end["acceleration"] / angular_frequency**2
        # Reflection occurs at t-T; the vertex appears at emission t-2T and
        # reception t. Eq.27 therefore gives factors 2e^(iΩT), 1+e^(2iΩT).
        endpoints += (
            sign
            * (
                2 * phase * displacement[axis]
                - (1 + phase**2) * vertex_displacement[axis]
            )
            / (2 * arm_length)
        )
        newtonian_differential += sign * (
            newtonian_acceleration(end_relative, quadrupole, G=G)[axis]
            - newtonian_acceleration(-source, quadrupole, G=G)[axis]
        )
        for s, weight in zip(nodes, weights):
            field = quadrupole_field(
                s * direction - source, quadrupole, angular_frequency, G=G, c=c
            )
            h = field["metric"]
            out = np.exp(1j * angular_frequency * (2 * T - s / c))
            back = np.exp(1j * angular_frequency * s / c)
            # Null condition: even=h00+h_pp, odd=2h_0p. Reversing the ray
            # reverses only the mixed term; the equal-arm vertex clock cancels.
            even = h[0, 0] + h[axis + 1, axis + 1]
            odd = 2 * h[0, axis + 1]
            path += (
                sign
                * weight
                * ((even + odd) * out + (even - odd) * back)
                / (4 * arm_length)
            )
            curvature += (
                sign
                * weight
                * (out + back)
                * field["tidal"][axis, axis]
                / (2 * arm_length * angular_frequency**2)
            )
    return {
        "total": endpoints + path,
        "harmonic_endpoint": endpoints,
        "harmonic_path": path,
        "curvature_total": curvature,
        "newtonian_instantaneous": -newtonian_differential
        / (angular_frequency**2 * arm_length),
        "newtonian_differential_acceleration": newtonian_differential,
    }
