"""
Phase extraction and phase conversion helpers for coherent source arrays.

The source-array goal is constructive interference at the detector.  Distances
alone are not always enough in the near field, so this module recovers each
source's actual detector-response phase from the complete complex response.
"""

from __future__ import annotations


import numpy as np

from ghe.config import SourceConfig
from ghe.metric import calculate_response_phasor


def get_signal_amplitude_and_phase(
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    distance: float,
    *,
    config: SourceConfig | None = None,
) -> tuple[float, float]:
    """
    Recover detector-response amplitude and phase for a single source.

    Returns peak dimensionless amplitude and cosine phase at ``2 * omega``.
    Propagation and detector integration are already included in this phase.
    """

    phasor = calculate_response_phasor(
        theta_src, phi_src, theta_rot, phi_rot, R=distance, config=config
    )
    amplitude = float(abs(phasor))
    # Legacy helper returns phi in A*cos(Omega*t+phi), whereas our internal
    # complex convention is Re[H*exp(-i*Omega*t)]. Therefore phi=-arg(H).
    phase = 0.0 if amplitude <= np.finfo(float).tiny else -float(np.angle(phasor))
    return amplitude, phase


def wrap_phase(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap phase angles to ``[-pi, pi)`` for compact storage."""

    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(wrapped)
    return wrapped


def rotor_phase_from_gw_phase(
    gw_phase_offset: np.ndarray | float,
) -> np.ndarray | float:
    """
    Convert emitted GW phase correction to mechanical rotor phase correction.

    The quadrupole signal oscillates twice per mechanical rotation, so the rotor
    phase offset is half the GW phase offset.
    """

    return 0.5 * gw_phase_offset
