"""Physics checks for the diagnostic leading-quadrupole response."""

import numpy as np
import pytest

from ghe.config import SourceConfig
from ghe.finite_distance import (
    michelson_response,
    newtonian_acceleration,
    quadrupole_field,
    rotor_quadrupole_phasor,
)


Q = np.array([[2, 1j, 0.4], [1j, -0.5, -0.2j], [0.4, -0.2j, -1.5]])


def test_static_field_is_newtonian_and_curvature_is_tidal_gradient():
    x = np.array([2.0, 3.0, -4.0])
    field = quadrupole_field(x, Q, 0, G=1, c=10)
    expected = newtonian_acceleration(x, Q, G=1)
    np.testing.assert_allclose(field["acceleration"], expected, rtol=1e-14, atol=1e-17)
    step = 1e-4
    jacobian = np.column_stack(
        [
            (
                newtonian_acceleration(x + step * u, Q, G=1)
                - newtonian_acceleration(x - step * u, Q, G=1)
            )
            / (2 * step)
            for u in np.eye(3)
        ]
    )
    np.testing.assert_allclose(field["tidal"], -jacobian, rtol=5e-9, atol=1e-13)


@pytest.mark.parametrize("kr", [0.001, 0.075, 1.0, 10.0, 100.0])
def test_vacuum_trace_and_harmonic_constraint(kr):
    x = np.array([2.0, 3.0, -4.0])
    omega = kr / np.linalg.norm(x)
    field = quadrupole_field(x, Q, omega, G=1, c=1)
    assert abs(np.trace(field["tidal"])) / np.linalg.norm(field["tidal"]) < 1e-13
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])

    def bar_up(point):
        h = quadrupole_field(point, Q, omega, G=1, c=1)["metric"]
        return eta @ (h - eta * np.sum(eta * h) / 2) @ eta

    step = 1e-5 / max(1.0, omega)
    divergence = -1j * omega * bar_up(x)[0, :]
    for j, u in enumerate(np.eye(3)):
        divergence += (
            bar_up(x + step * u)[j + 1, :] - bar_up(x - step * u)[j + 1, :]
        ) / (2 * step)
    scale = max(1 / np.linalg.norm(x), omega) * np.linalg.norm(bar_up(x))
    assert np.linalg.norm(divergence) / scale < 2e-8


def test_radiative_far_limit_is_transverse_quadrupole():
    x = np.array([2.0, 3.0, -4.0]) * 1e6
    r = np.linalg.norm(x)
    n = x / r
    projection = np.eye(3) - np.outer(n, n)
    Qtt = projection @ Q @ projection - projection * np.sum(projection * Q) / 2
    expected = -np.exp(1j * r) / r * Qtt
    actual = quadrupole_field(x, Q, 1, G=1, c=1)["tidal"]
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-6


@pytest.mark.parametrize("omega", [0.005, 0.5, 5.0, 30.0])
def test_finite_arm_response_agrees_in_two_gauges(omega):
    response = michelson_response(
        [2.1, 3.2, 4.3], Q, omega, arm_length=1, G=1, c=1, quadrature_order=96
    )
    np.testing.assert_allclose(
        response["total"], response["curvature_total"], rtol=3e-12, atol=0
    )


def test_short_arm_low_frequency_response_is_tidal():
    source = np.array([2.1, 3.2, 4.3])
    omega = 0.005
    E = quadrupole_field(-source, Q, omega, G=1, c=1)["tidal"]
    expected = (E[0, 0] - E[1, 1]) / omega**2
    actual = michelson_response(source, Q, omega, arm_length=1e-5, G=1, c=1)
    np.testing.assert_allclose(actual["curvature_total"], expected, rtol=1e-5, atol=0)


def test_rotor_phasor_matches_independent_time_domain_quadrupole():
    from tests.support.rotor import calculate_whole_tensor
    from ghe.geometry import rotation_body_to_detector

    cfg = SourceConfig()
    rot = rotation_body_to_detector(0.4, 0.8)
    phase = 2 * np.pi * np.arange(64) / 64
    values = np.array(
        [
            rot @ calculate_whole_tensor(t / cfg.gw_angular_frequency, cfg) @ rot.T
            for t in phase
        ]
    )
    direct = 2 * np.mean(values * np.exp(1j * phase)[:, None, None], axis=0)
    np.testing.assert_allclose(
        rotor_quadrupole_phasor(0.4, 0.8, config=cfg), direct, rtol=2e-14, atol=1e-10
    )
