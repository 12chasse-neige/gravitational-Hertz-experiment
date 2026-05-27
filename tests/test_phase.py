from __future__ import annotations

import numpy as np
import pytest

from ghe.config import DetectorConfig, RunConfig, SourceConfig
from ghe.noise import get_coupling_constant
from ghe.phase import (
    get_resonance_phase_for_detuned_signal_recycling,
    get_source_gw_frequency_hz,
)


def resonance_condition(phi: float, f_res: float, config: DetectorConfig) -> float:
    gamma = config.T_ITM * config.c / (4 * config.length)
    omega = 2 * np.pi * f_res
    phi_fp = np.arctan(omega / gamma) + np.mod(
        omega * config.length_SR / config.c,
        2 * np.pi,
    )
    rho = np.sqrt(1.0 - config.T_SRM)
    kappa = get_coupling_constant(omega, config=config)

    return (
        (1 + rho**2) * (np.cos(2 * phi) + 0.5 * kappa * np.sin(2 * phi))
        - 2 * rho * np.cos(2 * phi_fp)
    )


@pytest.mark.parametrize("f_res", [20.0, 100.0, 600.0, 1000.0, 3000.0])
def test_resonance_phase_solves_resonance_condition(f_res: float) -> None:
    config = DetectorConfig(T_SRM=1e-4, length_SR=55.0)

    phi = get_resonance_phase_for_detuned_signal_recycling(f_res, config=config)

    assert np.isfinite(phi)
    assert 0.0 <= phi <= np.pi
    assert abs(resonance_condition(phi, f_res, config)) < 1e-10


def test_source_distance_and_detector_length_track_source_arm_length() -> None:
    source = SourceConfig(L=2400.0, R=123.0)
    detector = DetectorConfig(length=4000.0)
    run_config = RunConfig(source=source, detector=detector)

    assert source.R == 1.5 * source.L
    assert run_config.detector.length == source.L
    assert run_config.detector.L == source.L


def test_default_detector_phase_targets_source_gw_frequency() -> None:
    source = SourceConfig()
    detector = DetectorConfig(T_SRM=1e-4, length_SR=55.0)
    expected = get_resonance_phase_for_detuned_signal_recycling(
        get_source_gw_frequency_hz(source),
        config=detector,
    )

    assert detector.resonance_frequency_hz == source.gw_frequency_hz
    assert np.isclose(detector.phi_SR, expected)


def test_run_config_recomputes_detector_phase_for_source_frequency() -> None:
    source = SourceConfig(omega=450.0 * 2.0 * np.pi)
    run_config = RunConfig(source=source, detector=DetectorConfig())
    expected = get_resonance_phase_for_detuned_signal_recycling(
        source.gw_frequency_hz,
        config=run_config.detector,
    )

    assert run_config.detector.resonance_frequency_hz == source.gw_frequency_hz
    assert np.isclose(run_config.detector.phi_SR, expected)

