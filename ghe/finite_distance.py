"""Leading conserved mass-quadrupole field and ideal free-mass Michelson response.

All complex quantities are PEAK phasors: X(t) = Re[X exp(-i Omega t)].
This retains arbitrary Omega*r/c, but assumes a compact, slow source. It does
not model suspension/control/optical-spring dynamics or cavity calibration.
See docs/near-field-analysis.md for the derivation and applicability limits.
This module owns source moments and exterior fields; detector integration lives
in detector_response. No stationary monopole or spin field is inserted into the
oscillating signal.
"""

from __future__ import annotations

import numpy as np

from .config import SourceConfig
from .geometry import rotation_body_to_detector


TIDAL_COMPONENT_NAMES = ("Q", "Qdot", "Qddot", "Q3", "Q4")


def rotor_quadrupole_phasor(theta_rot, phi_rot, *, config=None):
    """Return the oscillating STF mass moment at twice the two-hole spin rate.

    The axisymmetric cylinder and each hole's intrinsic second moment have no
    time-dependent quadrupole. This moment is exact at Newtonian order for the
    configured circular through-holes, not just a point-hole approximation.
    """
    cfg = config or SourceConfig()
    validate_rotor(cfg)
    missing_mass = -cfg.rho * np.pi * cfg.d**2 * cfg.H / 4
    q = missing_mass * cfg.s**2
    body = q * np.array([[1, 1j, 0], [1j, -1, 0], [0, 0, 0]])
    rotation = rotation_body_to_detector(theta_rot, phi_rot)
    return rotation @ body @ rotation.T


def _validate_field_inputs(relative_position, quadrupole, angular_frequency, G, c):
    """Shared exterior-field input checks; return vectors, STF moment and radii."""
    x = np.asarray(relative_position, dtype=float)
    Q = np.asarray(quadrupole, dtype=complex)
    if x.ndim < 1 or x.shape[-1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError("Field points must be finite three-vectors, shape (..., 3)")
    r = np.linalg.norm(x, axis=-1)
    if np.any(r <= 0):
        raise ValueError("Field points cannot coincide with the source")
    if Q.shape != (3, 3) or not np.all(np.isfinite(Q)):
        raise ValueError("Quadrupole must be finite, shape (3, 3)")
    scale = max(float(np.linalg.norm(Q)), np.finfo(float).tiny)
    if np.linalg.norm(Q - Q.T) > 1e-12 * scale or abs(np.trace(Q)) > 1e-12 * scale:
        raise ValueError("Quadrupole must be symmetric and trace free")
    if (
        not np.all(np.isfinite([angular_frequency, G, c]))
        or angular_frequency < 0
        or G <= 0
        or c <= 0
    ):
        raise ValueError("Require finite nonnegative frequency and positive G, c")
    return x, Q, r


def quadrupole_tidal_components(
    relative_position, quadrupole, angular_frequency, *, G, c
):
    """Return five derivative-order pieces of E_ij=c² R_0i0j, in s^-2.

    Inputs and batch shapes match quadrupole_field. Keys Q/Qdot/Qddot/Q3/Q4
    label explicit derivatives at u=t-r/c, NOT Fourier quadratures. Every
    component retains exp(ikr): expanding it would mix derivative orders and
    hide the cancellation of first-order retardation between Q and Qdot.

    The four angular coefficients below are those in theoreticalDerivation.md,
    written in powers of z=ikr. Since Q^(m) maps to (-i Omega)^m Q, z^m Q
    maps to (-r/c)^m Q^(m). In particular, the linear row is minus the static
    row, giving E[Qdot]=G/(c r^4) times the static angular tensor of Qdot(u).
    These additive pieces are model diagnostics, not independent observables.
    """
    x, Q, r = _validate_field_inputs(
        relative_position, quadrupole, angular_frequency, G, c
    )
    n = x / r[..., None]
    z = 1j * angular_frequency * r / c
    q = np.einsum("ij,...j->...i", Q, n)
    S = np.einsum("...i,...i->...", n, q)
    nq = n[..., :, None] * q[..., None, :]
    angular = (
        Q,
        nq + np.swapaxes(nq, -1, -2),
        np.eye(3) * S[..., None, None],
        n[..., :, None] * n[..., None, :] * S[..., None, None],
    )
    # Columns multiply Q_ij, n_i q_j+n_j q_i, delta_ij S, n_i n_j S.
    coefficients = (
        (-3, 15, 15 / 2, -105 / 2),
        (3, -15, -15 / 2, 105 / 2),
        (-3, 9, 3 / 2, -45 / 2),
        (2, -4, 1, 5),
        (-1, 1, -1 / 2, -1 / 2),
    )
    prefactor = G * np.exp(z) / r**5
    return {
        name: (prefactor * z**order)[..., None, None]
        * sum(coefficient * tensor for coefficient, tensor in zip(row, angular))
        for order, (name, row) in enumerate(zip(TIDAL_COMPONENT_NAMES, coefficients))
    }


def quadrupole_field(relative_position, quadrupole, angular_frequency, *, G, c):
    """Return physical covariant h, coordinate acceleration, and c² R_0i0j.

    relative_position: (..., 3) source-to-field vectors [m]. quadrupole: (3, 3)
    complex STF mass moment [kg m²]; angular_frequency: signal frequency [rad/s].
    Returned metric, acceleration and tidal arrays have trailing shapes (4,4),
    (3,), (3,3) and units dimensionless, m/s², s^-2 respectively.

    relative_position points from source COM to field point. The metric is in
    harmonic gauge and includes the time-dependent mass quadrupole only; static
    monopole/spin terms are excluded. Acceleration is for an initially slowly
    moving geodesic in these coordinates. Only the total interferometer signal
    (or curvature) is gauge invariant, not the separate acceleration/path terms.
    """
    x, Q, r = _validate_field_inputs(
        relative_position, quadrupole, angular_frequency, G, c
    )
    n = x / r[..., None]
    k = angular_frequency / c
    z = 1j * k * r
    exponential = np.exp(z)
    identity = np.eye(3)
    qn = np.einsum("ij,...j->...i", Q, n)
    qnn = np.einsum("...i,...i->...", n, qn)
    nn = n[..., :, None] * n[..., None, :]
    n_qn = n[..., :, None] * qn[..., None, :]

    # Radial derivative polynomials of g=exp(ikr)/r, not separate multipoles.
    # Each derivative acts on BOTH the exponential and the inverse distance;
    # dropping the latter would discard the near-zone terms (report eqs. 15–25).
    p1 = z - 1
    p2 = z**2 - 3 * z + 3
    p3 = z**3 - 6 * z**2 + 15 * z - 15
    p4 = z**4 - 10 * z**3 + 45 * z**2 - 105 * z + 105
    p1, p2, p3, p4 = map(np.asarray, (p1, p2, p3, p4))
    green = exponential / r
    Q_grad_green = (exponential / r**2 * p1)[..., None] * qn
    Q_hessian_green = exponential / r**3 * p2 * qnn
    grad_Q_hessian_green = (exponential / r**4)[..., None] * (
        (p3 * qnn)[..., None] * n + 2 * p2[..., None] * qn
    )
    hess_Q_hessian_green = (exponential / r**5)[..., None, None] * (
        (p4 * qnn)[..., None, None] * nn
        + p3[..., None, None]
        * (identity * qnn[..., None, None] + 2 * n_qn + 2 * np.swapaxes(n_qn, -1, -2))
        + 2 * p2[..., None, None] * Q
    )
    B = (exponential / r**3)[..., None, None] * (
        p2[..., None, None] * n_qn + p1[..., None, None] * Q
    )

    # Physical LOWER-index metric for signature (-+++), after trace reversal.
    # In particular h_ij contains delta_ij*h00 as well as the radiative term.
    # The mixed component's sign follows lowering its time index (eqs. 12–18).
    h00 = G / c**2 * Q_hessian_green
    metric = np.zeros(x.shape[:-1] + (4, 4), dtype=complex)
    metric[..., 0, 0] = h00
    metric[..., 0, 1:] = -2j * G * angular_frequency / c**3 * Q_grad_green
    metric[..., 1:, 0] = metric[..., 0, 1:]
    metric[..., 1:, 1:] = (
        identity * h00[..., None, None]
        - (2 * G * angular_frequency**2 / c**4 * green)[..., None, None] * Q
    )
    acceleration = G / 2 * grad_Q_hessian_green + 2 * G * k**2 * Q_grad_green
    # E_ij=c² R_0i0j has units s^-2 and geodesic deviation is xi_ddot=-E xi.
    # Keep all radial terms of the conserved quadrupole, including k^0 (eq.22).
    tidal = (
        -G / 2 * hess_Q_hessian_green
        - G
        * k**2
        * (B + np.swapaxes(B, -1, -2) - identity * Q_hessian_green[..., None, None] / 2)
        - (G * k**4 * green)[..., None, None] * Q
    )
    return {"metric": metric, "acceleration": acceleration, "tidal": tidal}


def newtonian_acceleration(relative_position, quadrupole, *, G):
    """Oscillatory quadrupole acceleration grad(U_Q), with U positive for mass."""
    x = np.asarray(relative_position, dtype=float)
    r = np.linalg.norm(x)
    if r <= 0:
        raise ValueError("Field point cannot coincide with source")
    n = x / r
    qn = np.asarray(quadrupole) @ n
    return G / r**4 * (3 * qn - 7.5 * n * (n @ qn))


def michelson_response(*args, **kwargs):
    """Compatibility entry point for the independent dual-gauge diagnostic.

    Production callers use ghe.metric.calculate_response_phasor instead.
    """
    from .detector_response import reference_michelson_response

    return reference_michelson_response(*args, **kwargs)


def validate_rotor(config: SourceConfig) -> None:
    """Validate the supported compact two-hole rotor (all lengths in metres).

    Slow motion and small source size remain physical approximation assumptions;
    this checks impossible geometry and superluminal rotation, not elasticity.
    """
    values = [
        config.H,
        config.D,
        config.d,
        config.R,
        config.rho,
        config.G,
        config.c,
        config.omega,
        config.L,
        config.s,
    ]
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError(
            "Rotor constants, dimensions, frequency and distances must be finite and positive"
        )
    if config.num != 2:
        raise ValueError("The oscillating quadrupole model requires two opposite holes")
    if config.s + config.d / 2 > config.D / 2 or 2 * config.s < config.d:
        raise ValueError("Holes must lie inside the rotor and must not overlap")
    if config.omega * config.D / 2 >= config.c:
        raise ValueError("Rotor rim speed must be below c")
