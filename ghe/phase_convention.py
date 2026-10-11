"""Translate Buonanno--Chen SR detuning to the public GWINC tune phase.

The project's analytic phi_SR appears in the Buonanno--Chen input/output
coefficients. GWINC 0.6's ``SignalRecycledBnC`` implementation defines that
same angle as ``(pi - Optics.SRM.Tunephase) / 2``. Its newer
``SignalRecycled`` model uses ``Tunephase / 2`` for an internal rotation,
but that internal angle is in a different quadrature convention. Both GWINC
models consume the same public Tunephase input.
"""

from __future__ import annotations

from math import pi


def analytic_phi_from_gwinc_tunephase(tunephase: float) -> float:
    """Convert a GWINC YAML Tunephase to analytic Buonanno--Chen phi_SR [rad]."""

    return (pi - float(tunephase)) / 2.0


def gwinc_tunephase_from_analytic_phi(phi_sr: float) -> float:
    """Convert analytic Buonanno--Chen phi_SR to GWINC Tunephase [rad]."""

    return pi - 2.0 * float(phi_sr)
