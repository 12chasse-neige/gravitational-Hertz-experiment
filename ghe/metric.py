"""Public single-source observable: geometry → moment → detector phasor.

This is the production entry point shared by optimization and array analysis.
Field equations belong to finite_distance; optical integration belongs to
detector_response. No local TT projection or radiative-only backend remains.
The output is a dimensionless equal-arm differential round-trip delay, measured
at reception, with peak convention h(t)=Re[H exp(-i Ω t)].
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .config import SourceConfig
from .geometry import spherical_unit_vector
from .finite_distance import rotor_quadrupole_phasor
from .detector_response import IntegrationSettings, curvature_response


def calculate_response_phasor(
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
    R: float | None = None,
    settings: IntegrationSettings | None = None,
) -> complex:
    """Return peak detector phasor H for one rotor; all angles are radians.

    Source angles point from the vertex toward the COM; rotor angles point along
    the body +z axis. R overrides COM-to-vertex distance [m] without mutating the
    caller's configuration. Raises ValueError for invalid geometry and
    ArithmeticError when arm integration cannot meet its requested accuracy.
    """
    cfg = config or SourceConfig()
    if R is not None:
        cfg = replace(cfg, R=float(R))
    if not np.all(np.isfinite([theta_src, phi_src, theta_rot, phi_rot])):
        raise ValueError("Source and rotor angles must be finite")
    moment = rotor_quadrupole_phasor(theta_rot, phi_rot, config=cfg)
    source = cfg.R * spherical_unit_vector(theta_src, phi_src)
    enclosing_radius = np.hypot(cfg.D / 2, cfg.H / 2)
    return curvature_response(
        source,
        moment,
        cfg.gw_angular_frequency,
        arm_length=cfg.L,
        G=cfg.G,
        c=cfg.c,
        source_radius=enclosing_radius,
        settings=settings,
    ).phasor


def calculate_metric_response(
    t: float,
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
    R: float | None = None,
    settings: IntegrationSettings | None = None,
) -> float:
    """Return dimensionless response at reception time t [s].

    For many time samples, calculate_response_phasor once and synthesize with
    numpy; do not repeat the spatial integration at every sample.
    """
    if not np.isfinite(t):
        raise ValueError("Reception time must be finite")
    cfg = config or SourceConfig()
    phasor = calculate_response_phasor(
        theta_src, phi_src, theta_rot, phi_rot, config=cfg, R=R, settings=settings
    )
    return float(np.real(phasor * np.exp(-1j * cfg.gw_angular_frequency * t)))
