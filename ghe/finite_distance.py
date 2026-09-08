"""Leading conserved mass-quadrupole field and ideal free-mass Michelson response.

All complex quantities are PEAK phasors: X(t) = Re[X exp(-i Omega t)].
This retains arbitrary Omega*r/c, but assumes a compact, slow source. It does
not model suspension/control/optical-spring dynamics or cavity calibration.
See docs/near-field-analysis.md for the derivation and applicability limits.
The legacy metric and near_field modules are intentionally not redirected.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import SourceConfig
from .geometry import rotation_body_to_detector


def rotor_quadrupole_phasor(theta_rot, phi_rot, *, config=None):
    """Return the oscillating STF mass moment at twice the two-hole spin rate.

    The axisymmetric cylinder and each hole's intrinsic second moment have no
    time-dependent quadrupole. This moment is exact at Newtonian order for the
    configured circular through-holes, not just a point-hole approximation.
    """
    cfg = config or SourceConfig()
    if cfg.num != 2:
        raise ValueError("This diagnostic requires two diametrically opposite holes")
    missing_mass = -cfg.rho * np.pi * cfg.d**2 * cfg.H / 4
    q = missing_mass * cfg.s**2
    body = q * np.array([[1, 1j, 0], [1j, -1, 0], [0, 0, 0]])
    rotation = rotation_body_to_detector(theta_rot, phi_rot)
    return rotation @ body @ rotation.T


def quadrupole_field(relative_position, quadrupole, angular_frequency, *, G, c):
    """Return physical covariant h, coordinate acceleration, and c² R_0i0j.

    relative_position points from source COM to field point. The metric is in
    harmonic gauge and includes the time-dependent mass quadrupole only; static
    monopole/spin terms are excluded. Acceleration is for an initially slowly
    moving geodesic in these coordinates. Only the total interferometer signal
    (or curvature) is gauge invariant, not the separate acceleration/path terms.
    """
    x = np.asarray(relative_position, dtype=float)
    Q = np.asarray(quadrupole, dtype=complex)
    r = np.linalg.norm(x)
    if x.shape != (3,) or r <= 0:
        raise ValueError("Field point must be a nonzero three-vector")
    if Q.shape != (3, 3) or not np.allclose(Q, Q.T):
        raise ValueError("Quadrupole must be symmetric, shape (3, 3)")
    if abs(np.trace(Q)) > 1e-12 * max(np.linalg.norm(Q), 1e-300):
        raise ValueError("Quadrupole must be trace free")
    if angular_frequency < 0 or G <= 0 or c <= 0:
        raise ValueError("Require nonnegative frequency and positive G, c")
    n = x / r
    k = angular_frequency / c
    z = 1j * k * r
    exponential = np.exp(z)
    identity = np.eye(3)
    qn = Q @ n
    qnn = n @ qn
    p1 = z - 1
    p2 = z**2 - 3*z + 3
    p3 = z**3 - 6*z**2 + 15*z - 15
    p4 = z**4 - 10*z**3 + 45*z**2 - 105*z + 105
    green = exponential / r
    Q_grad_green = exponential / r**2 * p1 * qn
    Q_hessian_green = exponential / r**3 * p2 * qnn
    grad_Q_hessian_green = exponential / r**4 * (p3*qnn*n + 2*p2*qn)
    hess_Q_hessian_green = exponential / r**5 * (
        p4*qnn*np.outer(n, n)
        + p3*(identity*qnn + 2*np.outer(n, qn) + 2*np.outer(qn, n))
        + 2*p2*Q
    )
    # B_ij = Q_ja partial_ia (exp(ikr)/r).
    B = exponential / r**3 * (p2*np.outer(n, qn) + p1*Q)
    h00 = G / c**2 * Q_hessian_green
    metric = np.zeros((4, 4), dtype=complex)
    metric[0, 0] = h00
    metric[0, 1:] = -2j*G*angular_frequency/c**3 * Q_grad_green
    metric[1:, 0] = metric[0, 1:]
    metric[1:, 1:] = identity*h00 - 2*G*angular_frequency**2/c**4 * Q*green
    acceleration = G/2 * grad_Q_hessian_green + 2*G*k**2 * Q_grad_green
    tidal = (
        -G/2*hess_Q_hessian_green
        - G*k**2*(B+B.T-identity*Q_hessian_green/2)
        - G*k**4*Q*green
    )
    return {"metric": metric, "acceleration": acceleration, "tidal": tidal}


def newtonian_acceleration(relative_position, quadrupole, *, G):
    """Oscillatory quadrupole acceleration grad(U_Q), with U positive for mass."""
    x = np.asarray(relative_position, dtype=float)
    r = np.linalg.norm(x)
    if r <= 0:
        raise ValueError("Field point cannot coincide with source")
    n = x/r
    qn = np.asarray(quadrupole) @ n
    return G/r**4 * (3*qn - 7.5*n*(n@qn))


def michelson_response(source_position, quadrupole, angular_frequency, *,
                       arm_length, G, c, quadrature_order=48):
    """Equal-arm point-mass Michelson, vertex at zero, arms along +x and +y.

    Both end mirrors and the common vertex follow free geodesics at the signal
    frequency. No L/R or Omega*L/c expansion is made. Two equivalent evaluations
    are returned: harmonic metric + moving endpoints, and synchronous-gauge
    curvature integration. The common vertex clock term cancels between arms.
    """
    if angular_frequency <= 0 or arm_length <= 0 or quadrature_order < 4:
        raise ValueError("Require positive frequency/length and quadrature order >=4")
    source = np.asarray(source_position, dtype=float)
    T = arm_length/c
    phase = np.exp(1j*angular_frequency*T)
    nodes, weights = leggauss(quadrature_order)
    nodes = (nodes+1)*arm_length/2
    weights = weights*arm_length/2
    vertex = quadrupole_field(-source, quadrupole, angular_frequency, G=G, c=c)
    vertex_displacement = -vertex["acceleration"]/angular_frequency**2
    endpoints = 0j
    path = 0j
    curvature = 0j
    newtonian_differential = 0j
    for axis, sign in [(0, 1), (1, -1)]:
        direction = np.eye(3)[axis]
        end_relative = arm_length*direction-source
        end = quadrupole_field(end_relative, quadrupole, angular_frequency, G=G, c=c)
        displacement = -end["acceleration"]/angular_frequency**2
        endpoints += sign*(2*phase*displacement[axis]
                           -(1+phase**2)*vertex_displacement[axis])/(2*arm_length)
        newtonian_differential += sign*(
            newtonian_acceleration(end_relative, quadrupole, G=G)[axis]
            - newtonian_acceleration(-source, quadrupole, G=G)[axis])
        for s, weight in zip(nodes, weights):
            field = quadrupole_field(s*direction-source, quadrupole,
                                     angular_frequency, G=G, c=c)
            h = field["metric"]
            out = np.exp(1j*angular_frequency*(2*T-s/c))
            back = np.exp(1j*angular_frequency*s/c)
            even = h[0, 0]+h[axis+1, axis+1]
            odd = 2*h[0, axis+1]
            path += sign*weight*((even+odd)*out+(even-odd)*back)/(4*arm_length)
            curvature += sign*weight*(out+back)*field["tidal"][axis, axis]/(
                2*arm_length*angular_frequency**2)
    return {
        "total": endpoints+path,
        "harmonic_endpoint": endpoints,
        "harmonic_path": path,
        "curvature_total": curvature,
        "newtonian_instantaneous": -newtonian_differential/(
            angular_frequency**2*arm_length),
        "newtonian_differential_acceleration": newtonian_differential,
    }
