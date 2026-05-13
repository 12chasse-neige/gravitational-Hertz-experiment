"""
Quantum-noise model for a LIGO-like interferometer.

The SNR code needs a one-sided strain-noise power spectral density (PSD)
``S_h(f)``.  This module provides an analytic quantum-noise approximation:

1. Estimate circulating arm-cavity laser power.
2. Compute the standard quantum limit ``h_SQL``.
3. Compute the optomechanical coupling ``kappa``.
4. Combine shot noise and radiation-pressure noise, optionally with squeezing.

Functions return PSD values, not amplitude spectral density.  Plotting code may
take square roots when it wants ASD units.
"""

from __future__ import annotations

import numpy as np

from .config import DetectorConfig


def get_laser_power_in_cavity(inputPower: float, config: DetectorConfig | None = None) -> float:
    """
    Estimate circulating power inside one arm cavity.

    The calculation uses a simple recycling-gain and arm-cavity-gain model based
    on mirror transmissions and optical losses in ``DetectorConfig``.
    """

    active_config = config or DetectorConfig()
    l_mirror = active_config.loss_mirror_ppm * 1e-6
    l_BS = active_config.loss_BS_ppm * 1e-6
    l_roundtrip_arm = 2 * l_mirror
    total_arm_loss = active_config.T_ETM + l_roundtrip_arm
    R_arm = 1 - (4 * total_arm_loss / active_config.T_ITM)
    r_arm = np.sqrt(R_arm)
    r_comp = r_arm * (1 - l_BS)
    r_PRM = np.sqrt(1 - active_config.T_PRM)
    t_PRM = np.sqrt(active_config.T_PRM)
    PRG = (t_PRM / (1 - r_PRM * r_comp)) ** 2
    P_BS = inputPower * PRG
    P_arm_input = P_BS / 2
    ACG = 4 / active_config.T_ITM
    return float(P_arm_input * ACG)


def get_standard_quantum_limit(
    gravitationalWaveOmega: float | np.ndarray,
    config: DetectorConfig | None = None,
) -> np.ndarray:
    """
    Return the standard quantum limit strain amplitude ``h_SQL``.

    Input is angular GW frequency ``Omega = 2*pi*f``.  The returned quantity is
    amplitude-like; callers square it when building a PSD.
    """

    active_config = config or DetectorConfig()
    omega = np.asarray(gravitationalWaveOmega, dtype=float)
    return np.sqrt(8 * active_config.hbar / (active_config.testmass * omega**2 * active_config.length**2))


def get_coupling_constant(
    gravitationalWaveOmega: float | np.ndarray,
    config: DetectorConfig | None = None,
) -> np.ndarray:
    """
    Return the optomechanical coupling ``kappa(Omega)``.

    ``kappa`` controls the balance between shot noise and radiation-pressure
    noise.  It depends on cavity bandwidth, circulating power, test mass, arm
    length, and GW angular frequency.
    """

    active_config = config or DetectorConfig()
    gw_omega = np.asarray(gravitationalWaveOmega, dtype=float)
    laser_omega = 2 * np.pi * active_config.c / active_config.wavelength
    gamma = active_config.T_ITM * active_config.c / (4 * active_config.length)
    P_0 = 2 * get_laser_power_in_cavity(active_config.power, config=active_config)
    return (8 * gamma * laser_omega * P_0) / (
        active_config.testmass
        * active_config.length
        * active_config.c
        * gw_omega**2
        * (gamma**2 + gw_omega**2)
    )


def get_quantum_noise_psd(freq: float | np.ndarray, config: DetectorConfig | None = None) -> np.ndarray:
    """
    Return unsqueezed quantum-noise strain PSD at frequency ``freq`` in Hz.

    The form ``(h_SQL^2 / 2) * (kappa + 1/kappa)`` combines radiation pressure
    and shot noise in the convention used by the original scripts.
    """

    active_config = config or DetectorConfig()
    gravitationalWaveOmega = 2 * np.pi * np.asarray(freq, dtype=float)
    kappa = get_coupling_constant(gravitationalWaveOmega, config=active_config)
    h_sql_sq = get_standard_quantum_limit(gravitationalWaveOmega, config=active_config) ** 2
    return (h_sql_sq / 2.0) * (kappa + 1 / kappa)


def squeeze_db_to_r(squeeze_db: float) -> float:
    """Convert squeezing in dB to squeeze parameter ``r``."""

    return float(squeeze_db * np.log(10.0) / 20.0)


def squeeze_quantum_noise_with_same_angle(
    freq: float | np.ndarray,
    squeeze_db: float = 10.0,
    config: DetectorConfig | None = None,
) -> np.ndarray:
    """
    Return PSD for frequency-independent squeezed vacuum.

    With a fixed squeeze angle, the shot-noise-like and radiation-pressure-like
    terms are rescaled in opposite directions.  This is useful for comparison
    with the simpler unsqueezed model.
    """

    active_config = config or DetectorConfig()
    freq = np.asarray(freq, dtype=float)
    gravitationalWaveOmega = 2 * np.pi * freq
    r = squeeze_db_to_r(squeeze_db)
    e_m2r = np.exp(-2.0 * r)
    e_p2r = np.exp(2.0 * r)
    h_sql_sq = get_standard_quantum_limit(gravitationalWaveOmega, config=active_config) ** 2
    kappa = get_coupling_constant(gravitationalWaveOmega, config=active_config)
    return (h_sql_sq / 2.0) * (e_m2r * kappa + e_p2r / kappa)


def squeeze_quantum_noise_with_varying_angle(
    freq: float | np.ndarray,
    squeeze_db: float = 10.0,
    config: DetectorConfig | None = None,
) -> np.ndarray:
    """
    Return PSD for an ideal frequency-dependent squeeze angle.

    This optimistic model applies the squeezing reduction to the total quantum
    noise curve: ``S_h_sqz = S_h_unsqz * exp(-2r)``.
    """

    active_config = config or DetectorConfig()
    freq = np.asarray(freq, dtype=float)
    gravitationalWaveOmega = 2 * np.pi * freq
    r = squeeze_db_to_r(squeeze_db)
    h_sql_sq = get_standard_quantum_limit(gravitationalWaveOmega, config=active_config) ** 2
    kappa = get_coupling_constant(gravitationalWaveOmega, config=active_config)
    return (h_sql_sq / 2.0) * (kappa + 1 / kappa) * np.exp(-2.0 * r)
