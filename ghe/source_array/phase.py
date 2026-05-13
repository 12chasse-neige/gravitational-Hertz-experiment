"""
Phase extraction and phase conversion helpers for coherent source arrays.

The source-array goal is constructive interference at the detector.  Distances
alone are not always enough in the near field, so this module recovers each
source's actual detector-response phase by sampling the metric response.
"""

from __future__ import annotations

import math

import numpy as np

from ghe.config import SourceConfig
from ghe.metric import calculate_metric_response


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

    The quadrupole radiation is dominated by ``2 * omega``. Sampling at ``t=0`` and
    one quarter of that GW period recovers the sinusoid phase without changing the
    physical model.
    """

    active_config = config or SourceConfig()
    t0 = 0.0
    quarter_period = np.pi / (4.0 * active_config.omega)

    signal_t0 = calculate_metric_response(
        t0,
        theta_src,
        phi_src,
        theta_rot,
        phi_rot,
        config=active_config,
        R=distance,
    )
    signal_t90 = calculate_metric_response(
        quarter_period,
        theta_src,
        phi_src,
        theta_rot,
        phi_rot,
        config=active_config,
        R=distance,
    )

    amplitude = math.hypot(signal_t0, signal_t90)
    if amplitude <= np.finfo(float).tiny:
        return amplitude, 0.0

    normalized_cos = np.clip(signal_t0 / amplitude, -1.0, 1.0)
    normalized_sin = np.clip(-signal_t90 / amplitude, -1.0, 1.0)
    phase = math.atan2(normalized_sin, normalized_cos)
    return float(amplitude), float(phase)


def wrap_phase(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap phase angles to ``[-pi, pi)`` for compact storage."""

    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(wrapped)
    return wrapped


def rotor_phase_from_gw_phase(gw_phase_offset: np.ndarray | float) -> np.ndarray | float:
    """
    Convert emitted GW phase correction to mechanical rotor phase correction.

    The quadrupole signal oscillates twice per mechanical rotation, so the rotor
    phase offset is half the GW phase offset.
    """

    return 0.5 * gw_phase_offset
