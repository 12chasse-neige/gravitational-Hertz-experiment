"""Independent analytic and finite-hole Newtonian forces for physics tests.

No production field or detector-response functions are imported. Direct signed
mass integration and harmonic extraction provide an independent check on the
quadrupole normalization and its Newtonian detector limit.
"""

from __future__ import annotations
from decimal import Decimal, localcontext
import numpy as np
from ghe.config import SourceConfig
from ghe.geometry import rotation_body_to_detector
from tests.support.newtonian import _cylinder_quadrature


def endpoints(length: float) -> np.ndarray:
    """Vertex and the two end masses, in that order, in detector coordinates."""
    return np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0]])


def quadrupole_response(
    config: SourceConfig, source_position: np.ndarray, rotor_angles: np.ndarray
) -> tuple[complex, complex, np.ndarray, np.ndarray]:
    """Return strain, differential acceleration, all accelerations, and Q.

    For two diametrically opposed holes of signed mass m_h, the oscillatory
    STF moment is m_h s^2 [[1,i,0],[i,-1,0],[0,0,0]] in the body frame.
    The force follows by differentiating U_Q = 3G Q_ij r_i r_j / (2 r^5).
    """
    rotation = rotation_body_to_detector(*rotor_angles)
    hole_mass = -config.rho * np.pi * config.d**2 * config.H / 4.0
    q_body = (
        hole_mass
        * config.s**2
        * np.array([[1.0, 1j, 0.0], [1j, -1.0, 0.0], [0.0, 0.0, 0.0]])
    )
    q_detector = rotation @ q_body @ rotation.T
    relative_positions = endpoints(config.L) - source_position
    distance = np.linalg.norm(relative_positions, axis=1)
    qr = relative_positions @ q_detector.T
    rqr = np.einsum("ij,ij->i", relative_positions, qr)
    acceleration = (
        3.0 * config.G * qr / distance[:, None] ** 5
        - 7.5 * config.G * rqr[:, None] * relative_positions / distance[:, None] ** 7
    )
    differential_acceleration = (
        acceleration[1, 0]
        - acceleration[0, 0]
        - acceleration[2, 1]
        + acceleration[0, 1]
    )
    strain = -differential_acceleration / (config.gw_angular_frequency**2 * config.L)
    return strain, differential_acceleration, acceleration, q_detector


def direct_newtonian_response(
    config: SourceConfig,
    source_position: np.ndarray,
    rotor_angles: np.ndarray,
    *,
    finite_cylinders: bool,
    phase_samples: int,
    quadrature: tuple[int, int, int],
) -> tuple[complex, int]:
    """Integrate exact Newtonian forces and extract the 2*spin harmonic.

    The full uniform cylinder has stationary Eulerian density, so its 600-Hz
    force is zero.  Only the signed missing mass in the two holes is needed.
    Force sums and DC subtraction use 50-digit Decimal arithmetic to avoid
    subtractive loss: the quadrupole force is about 1e-7 of the hole monopole.
    Geometry/trigonometry/quadrature inputs and final Fourier sums are float64.
    """
    rotation = rotation_body_to_detector(*rotor_angles)
    if finite_cylinders:
        points, weights = _cylinder_quadrature(
            config.d / 2.0, config.H, config.s, 0.0, *quadrature
        )
        points = np.concatenate((points, -points), axis=0)
        weights = np.tile(-config.rho * weights, 2)
    else:
        points = np.array([[config.s, 0.0, 0.0], [-config.s, 0.0, 0.0]])
        hole_mass = -config.rho * np.pi * config.d**2 * config.H / 4.0
        weights = np.full(2, hole_mass)

    phase = 2.0 * np.pi * np.arange(phase_samples) / phase_samples
    with localcontext() as context:
        context.prec = 50
        decimal_weights = [Decimal(float(weight)) for weight in weights]
        positions = [
            [Decimal(float(value)) for value in point]
            for point in endpoints(config.L) - source_position
        ]
        gravitational_constant = Decimal(config.G)
        differential_samples: list[Decimal] = []
        for angle in phase:
            cosine, sine = np.cos(angle), np.sin(angle)
            spin = np.array(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            rotated_points = [
                [Decimal(float(value)) for value in point]
                for point in points @ spin.T @ rotation.T
            ]
            component_accelerations = []
            for endpoint_index, component in ((1, 0), (0, 0), (2, 1), (0, 1)):
                acceleration = Decimal(0)
                for weight, point in zip(decimal_weights, rotated_points):
                    separation = [
                        x - y for x, y in zip(positions[endpoint_index], point)
                    ]
                    radius_squared = sum(value * value for value in separation)
                    acceleration -= (
                        gravitational_constant
                        * weight
                        * separation[component]
                        / (radius_squared * radius_squared.sqrt())
                    )
                component_accelerations.append(acceleration)
            gx_end, gx_vertex, gy_end, gy_vertex = component_accelerations
            differential_samples.append(gx_end - gx_vertex - gy_end + gy_vertex)

        # Removing a constant cannot change a nonzero Fourier harmonic.
        samples_without_dc = np.array(
            [float(value - differential_samples[0]) for value in differential_samples]
        )
    differential_phasor = 2.0 * np.mean(samples_without_dc * np.exp(2j * phase))
    strain = -differential_phasor / (config.gw_angular_frequency**2 * config.L)
    return strain, len(weights)
