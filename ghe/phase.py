"""
This module will get the resonance phase for the detuned signal, keeping the resonant frequency
fixed at the GW frequency of the sources.
"""

from __future__ import annotations

import numpy as np

from .config import DetectorConfig, SourceConfig

from scipy.optimize import root_scalar

from .noise import get_coupling_constant

_phi_SR_cache: dict[tuple, float] = {}


def _phi_SR_cache_key(f_res: float, config: DetectorConfig) -> tuple:
    return (
        round(f_res, 3),
        round(config.T_SRM, 10),
        round(config.length, 3),
        round(config.length_SR, 3),
        round(config.T_ITM, 10),
        config.c,
        config.power,
        config.testmass,
        config.wavelength,
        config.loss_mirror_ppm,
        config.loss_BS_ppm,
        config.T_PRM,
        config.T_ETM,
    )


def get_source_gw_frequency_hz(config: SourceConfig | None = None) -> float:
    """Return the dominant gravitational-wave frequency emitted by a source."""

    active_config = config or SourceConfig()
    return active_config.gw_frequency_hz


def _find_sign_change_bracket(
    func,
    *,
    lower: float,
    upper: float,
    num_samples: int = 1000,
) -> tuple[float, float]:
    """Return the first interval in ``[lower, upper]`` that brackets a root."""

    grid = np.linspace(lower, upper, num_samples + 1)
    values = np.asarray([func(point) for point in grid])

    for left, right, f_left, f_right in zip(
        grid[:-1],
        grid[1:],
        values[:-1],
        values[1:],
    ):
        if f_left == 0.0:
            return float(left), float(left)
        if f_left * f_right < 0.0:
            return float(left), float(right)

    if values[-1] == 0.0:
        return float(upper), float(upper)

    raise RuntimeError("Failed to bracket resonance phase in [0, pi].")


def get_resonance_phase_for_detuned_signal_recycling(
    f_res: float,
    config: DetectorConfig | None = None,
) -> float:
    """
    Get the resonance phase for detuned signal recycling.

    The resonant frequency of the signal recycling cavity is given by:

        (1 + rho^2) * (cos (2 * phi) + kappa * sin (2 * phi) / 2) - 2 * rho * cos (2 * phi_fp) = 0
    
    where rho = sqrt(1 - T_SRM) is the amplitude reflectivity of the SR mirror, phi is the SR cavity 
    detuning phase, phi_fp is the phase accumulated by the sidebands in the arm cavity, and kappa is the 
    optomechanical coupling constant.  This function numerically solves for phi given a target resonant 
    frequency f_res.
    """
    active_config = config or DetectorConfig()
    key = _phi_SR_cache_key(f_res, active_config)
    cached = _phi_SR_cache.get(key)
    if cached is not None:
        return cached

    omega = 2 * np.pi * f_res
    gamma = active_config.T_ITM * active_config.c / (4 * active_config.length)
    phi_fp = np.arctan(omega / gamma) + np.mod(
        omega * active_config.length_SR / active_config.c,
        2 * np.pi,
    )
    rho = np.sqrt(1.0 - active_config.T_SRM)
    kappa = get_coupling_constant(omega, config=active_config)

    def resonance_condition(phi):
        sin_2phi = np.sin(2 * phi)
        cos_2phi = np.cos(2 * phi)
        cos_2phi_fp = np.cos(2 * phi_fp)

        return (1 + rho**2) * (cos_2phi + 0.5 * kappa * sin_2phi) - 2 * rho * cos_2phi_fp

    bracket = _find_sign_change_bracket(
        resonance_condition,
        lower=0.0,
        upper=np.pi,
    )
    if bracket[0] == bracket[1]:
        return bracket[0]

    result = root_scalar(resonance_condition, bracket=bracket, method="bisect")
    if not result.converged:
        raise RuntimeError("Failed to find resonance phase for detuned signal recycling.")

    phase = float(result.root)
    _phi_SR_cache[key] = phase
    return phase
