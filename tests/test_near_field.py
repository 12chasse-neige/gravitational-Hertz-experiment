from __future__ import annotations

from dataclasses import replace

import numpy as np

from ghe.config import SourceConfig
from ghe.geometry import rotation_body_to_detector
from ghe.metric import second_derivative_of_tensor
from ghe.near_field import (
    calculate_near_field_metric,
    calculate_quadrupole_gw_metric,
    rotor_rest_volume,
)
from ghe.optimization import load_best_geometry


def _best_angles() -> tuple[float, float, float, float]:
    geometry = load_best_geometry()
    assert geometry is not None
    return geometry.angles


def test_rotor_volume_matches_cylinder_minus_holes() -> None:
    config = SourceConfig()
    expected = np.pi * config.H * ((config.D / 2.0) ** 2 - config.num * (config.d / 2.0) ** 2)
    assert np.isclose(rotor_rest_volume(config), expected)


def test_static_rotor_has_only_time_time_component() -> None:
    config = replace(SourceConfig(), omega=0.0)
    result = calculate_near_field_metric(
        0.0,
        *_best_angles(),
        config=config,
        radial_order=8,
        azimuthal_order=32,
        axial_order=8,
    )

    assert result.h_covariant[0, 0] > 0.0
    assert np.count_nonzero(result.h_covariant[1:, :]) == 0
    assert np.count_nonzero(result.h_covariant[:, 1:]) == 0
    assert np.isclose(result.quadrature_volume_m3, rotor_rest_volume(config), rtol=1e-14)


def test_metric_is_symmetric_finite_and_periodic() -> None:
    config = SourceConfig()
    kwargs = {
        "config": config,
        "radial_order": 8,
        "azimuthal_order": 32,
        "axial_order": 8,
    }
    result = calculate_near_field_metric(0.0, *_best_angles(), **kwargs)
    repeated = calculate_near_field_metric(
        2.0 * np.pi / config.omega,
        *_best_angles(),
        **kwargs,
    )

    assert np.all(np.isfinite(result.h_covariant))
    assert np.array_equal(result.h_covariant, result.h_covariant.T)
    assert np.allclose(result.h_covariant, repeated.h_covariant, rtol=5e-11, atol=1e-48)


def test_default_quadrature_is_converged_against_higher_order() -> None:
    config = SourceConfig()
    baseline = calculate_near_field_metric(0.0, *_best_angles(), config=config)
    refined = calculate_near_field_metric(
        0.0,
        *_best_angles(),
        config=config,
        radial_order=20,
        azimuthal_order=80,
        axial_order=20,
    )

    scale = np.maximum(np.abs(refined.h_covariant), 1e-48)
    assert np.max(np.abs(baseline.h_covariant - refined.h_covariant) / scale) < 1e-9


def test_quadrupole_gw_metric_matches_supplied_formula() -> None:
    config = SourceConfig()
    _, _, theta_rot, phi_rot = _best_angles()
    time_s = 0.0
    retarded_time = time_s - config.R / config.c
    rotation = rotation_body_to_detector(theta_rot, phi_rot)
    expected_spatial = (
        2.0
        * config.G
        / (config.c**4 * config.R)
        * rotation
        @ second_derivative_of_tensor(retarded_time, config)
        @ rotation.T
    )

    actual = calculate_quadrupole_gw_metric(
        time_s,
        theta_rot,
        phi_rot,
        config=config,
    )

    assert np.count_nonzero(actual[0, :]) == 0
    assert np.count_nonzero(actual[:, 0]) == 0
    assert np.allclose(actual[1:, 1:], expected_spatial, rtol=1e-15, atol=1e-50)
    assert np.isclose(np.trace(actual[1:, 1:]), 0.0, atol=1e-50)
