import numpy as np
import os
from pathlib import Path
from scipy.integrate import quad
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple


def spherical_unit_vector(theta: float, phi: float) -> np.ndarray:
    """
    Standard spherical angles on the unit sphere in a right-handed Cartesian frame:

        x = sin(theta) * cos(phi)
        y = sin(theta) * sin(phi)
        z = cos(theta)

    Here ``theta`` is the polar angle measured from +z, and ``phi`` is the azimuth
    measured from +x toward +y. This matches the detector-frame convention used
    throughout this refactor.
    """
    return np.array(
        [
            float(np.sin(theta) * np.cos(phi)),
            float(np.sin(theta) * np.sin(phi)),
            float(np.cos(theta)),
        ],
        dtype=float,
    )


def rotation_body_to_detector(theta_rot: float, phi_rot: float) -> np.ndarray:
    """
    Build an orthonormal rotation ``R`` (shape ``(3, 3)``) mapping body-frame
    coordinates into **detector** coordinates.

    Body frame (unchanged quadrupole construction in this repository):
        - ``z_body`` is the rotor symmetry axis.

    Detector frame (new convention requested for this project):
        - Origin at the interferometer vertex.
        - ``+x`` is LIGO arm 1, ``+y`` is LIGO arm 2, ``+z`` completes a right-handed
          triad (``z_hat = x_hat \\times y_hat``).

    The caller supplies ``(theta_rot, phi_rot)`` describing **where the rotor axis
    points in the detector frame** (same spherical convention as ``spherical_unit_vector``).

    Columns of ``R`` are the body basis vectors expressed in detector coordinates, so
    ``v_det = R @ v_body`` and ``h_det = R @ h_body @ R.T``.
    """
    z_hat = spherical_unit_vector(theta_rot, phi_rot)
    z_hat = z_hat / np.linalg.norm(z_hat)

    # Pick any vector not (nearly) parallel to ``z_hat`` to span the equatorial plane.
    if abs(float(z_hat[2])) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)

    x_hat = np.cross(z_hat, tmp)
    x_hat = x_hat / np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    # ``np.stack`` columns: body x, y, z map to ``x_hat``, ``y_hat``, ``z_hat``.
    return np.column_stack([x_hat, y_hat, z_hat])


@dataclass
class ExperimentConfig:
    """
    Dataclass to store all physical constants and experiment parameters.
    """
    num: int = 2               # number of the holes in the column
    H: float = 2.0             # height of the column (meter)
    D: float = 5.0             # diameter of the column (meter)
    d: float = 1.0             # diameter of the holes (meter)
    s: float = 1.5             # distance from center to holes (meter)
    R: float = 2000.0          # distance from source to detector (meter)
    rho: float = 1750.0        # density (kg/m^3)
    G: float = 6.674e-11       # gravitational constant
    c: float = 2.998e8         # speed of light
    omega: float = 300.0 * 2.0 * np.pi  # rotation frequency (rad/s)
    L: float = field(default_factory=lambda: float(os.getenv("LIGO_ARM_LENGTH", "1000.0")))  # length of the arm of the detector (meter)


# Cached angles from Data/bestPosition.txt (filled on first use; file is read at most once).
# Format is **detector-centric** with four numbers:
#   (theta_src, phi_src, theta_rot, phi_rot)
# See ``calculate_metric_response`` for precise definitions.
_BEST_POSITION_CACHE: Optional[Tuple[float, float, float, float]] = None

# Reasonable cold-start angles if ``Data/bestPosition.txt`` is missing. These are
# **not** guaranteed optimal; run ``bestPosition.py`` to refresh the cache.
_FALLBACK_BEST_POSITION: Tuple[float, float, float, float] = (
    0.01,  # source nearly along -z (almost overhead)
    0.0,
    0.01,  # rotor axis mostly in the x-y plane
    0.0,
)


def _parse_best_position_file_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse the machine-readable ``BEST_POSITION`` line from ``Data/bestPosition.txt``.

    Expected format (single line, four comma-separated floats)::

        BEST_POSITION: theta_src, phi_src, theta_rot, phi_rot

    Legacy six-angle outputs are intentionally **not** parsed here anymore; rerun
    ``bestPosition.py`` after this refactor to regenerate the cache.
    """
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("BEST_POSITION:"):
            rest = s.split(":", 1)[1].strip()
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) == 4:
                return (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
    return None


def _get_best_position_defaults() -> Tuple[float, float, float, float]:
    """
    Return cached angles from Data/bestPosition.txt; read the file at most once per process.
    """
    global _BEST_POSITION_CACHE
    if _BEST_POSITION_CACHE is not None:
        return _BEST_POSITION_CACHE
    path = Path(__file__).resolve().parent / "Data" / "bestPosition.txt"
    if path.is_file():
        try:
            parsed = _parse_best_position_file_text(path.read_text(encoding="utf-8"))
            if parsed is not None:
                _BEST_POSITION_CACHE = parsed
                return _BEST_POSITION_CACHE
        except OSError:
            pass
    _BEST_POSITION_CACHE = _FALLBACK_BEST_POSITION
    return _BEST_POSITION_CACHE


def get_hole_coordinate(k: int, t: float, config: ExperimentConfig) -> Tuple[float, float]:
    """
    Get the x and y coordinates of the k-th hole at time t.
    """
    x_k = config.s * np.cos(config.omega * t + k * (2 * np.pi / config.num)) 
    y_k = config.s * np.sin(config.omega * t + k * (2 * np.pi / config.num))
    return x_k, y_k


def calculate_whole_tensor(t: float, config: ExperimentConfig) -> np.ndarray:
    """
    Calculate the whole quadrupole tensor I_ij for the source using NumPy matrices.
    This avoids component-by-component calculation and builds the 3x3 matrix directly.
    """
    tensor = np.zeros((3, 3))
    volume = np.pi * config.d**2 / 4.0 * config.H
    mass = -config.rho * volume  # Using negative density for the holes
    
    # Add up the contributions of the 2 holes
    for k in range(config.num):
        x, y = get_hole_coordinate(k, t, config)
        coords = np.array([x, y, 0.0])
        r_squared = x**2 + y**2
        
        # I_ij = mass * (x_i * x_j - 1/3 * delta_ij * r^2)
        # np.outer creates the x_i * x_j matrix; np.eye(3) is the identity matrix (delta_ij)
        tensor += mass * (np.outer(coords, coords) - (1.0 / 3.0) * np.eye(3) * r_squared)
        
    return tensor


def second_derivative_of_tensor(t: float, config: ExperimentConfig) -> np.ndarray:
    """
    Calculate the second-order derivative of the quadrupole tensor.
    """
    tensor = calculate_whole_tensor(t, config)
    # Since the coordinates have a simple harmonic dependence on omega*t, 
    # the 2nd time derivative is simply multiplying by -4 * omega^2
    return -4.0 * config.omega**2 * tensor


def get_metric_tensor_body_frame(r: float, t: float, config: ExperimentConfig) -> np.ndarray:
    """
    Raw metric perturbation ``h_ij`` in the **source body frame** (the frame where the
    quadrupole model is built: rotor axis is +z, holes move in the x-y plane).

    This is the same physics as the pre-refactor ``get_metric_tensor``, but the name
    makes explicit that components still live in the body basis before rotating into
    the detector frame.
    """
    t_rev = t - r / config.c
    coeff = 2.0 * config.G / (r * config.c**4)
    return coeff * second_derivative_of_tensor(t_rev, config)


# def project_to_tt_gauge_static(h_matrix: np.ndarray) -> np.ndarray:
#     """
#     Project the metric tensor to TT gauge using a fixed z-axis propagation assumption.
#     """
#     P = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
                  
#     # Tr(P * h) = sum_{k,l} P_kl h_kl
#     trace_term = np.sum(P * h_matrix) 
    
#     # h_tt_ij = (P h P^T)_ij - 0.5 * P_ij * Tr(P h)
#     h_tt = P @ h_matrix @ P.T - 0.5 * P * trace_term
#     return h_tt


def project_to_tt_gauge_dynamic(h_matrix: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    """
    Project the metric tensor to TT gauge using local propagation direction.
    """
    r_norm = np.linalg.norm(r_vec)
    k = r_vec / r_norm  # Local propagation unit vector
    
    # Projection operator perpendicular to the propagation vector k
    P = np.eye(3) - np.outer(k, k)
    
    trace_term = np.sum(P * h_matrix)
    h_tt = P @ h_matrix @ P.T - 0.5 * P * trace_term
    return h_tt


def calculate_delta_t(
    t: float,
    n_src_to_det: np.ndarray,
    a_vec: np.ndarray,
    config: ExperimentConfig,
    R_body_to_det: np.ndarray,
) -> float:
    """
    Calculate the forward photon transition time delay.

    **Detector frame geometry (new convention).** The interferometer vertex is the
    origin. The outgoing arm used in this integral lies along ``a_vec`` (a unit
    vector, typically ``+x`` for arm 1). The source sits at ``S = R * n_src_to_det``,
    where ``n_src_to_det`` is the unit vector **from the detector toward the source**.

    The separation vector from the source to a point ``x`` meters down the arm is
    therefore ``r_vec = x * a_vec - R * n_src_to_det``, which is what enters both the
    propagation distance and the TT projector.
    """
    n_src_to_det = np.asarray(n_src_to_det, dtype=float)
    n_src_to_det = n_src_to_det / np.linalg.norm(n_src_to_det)
    R_body_to_det = np.asarray(R_body_to_det, dtype=float)

    def integrand(x):
        r_vec = x * a_vec - config.R * n_src_to_det
        r_distance = np.linalg.norm(r_vec)

        h_body = get_metric_tensor_body_frame(r_distance, t + x / config.c, config)
        h_matrix = R_body_to_det @ h_body @ R_body_to_det.T

        h_tt = project_to_tt_gauge_dynamic(h_matrix, r_vec)
        
        # Calculate a_i * a_j * h_tt_ij using dot products
        val = a_vec.T @ h_tt @ a_vec
        return val / (2.0 * config.c)
        
    # scipy.integrate.quad returns a tuple (integral_result, absolute_error)
    result, _ = quad(integrand, 0.0, config.L)
    return result


def calculate_delta_t_prime(
    t: float,
    n_src_to_det: np.ndarray,
    a_vec: np.ndarray,
    config: ExperimentConfig,
    R_body_to_det: np.ndarray,
) -> float:
    """
    Calculate the return photon transition time delay (same frame conventions as
    ``calculate_delta_t``).
    """
    n_src_to_det = np.asarray(n_src_to_det, dtype=float)
    n_src_to_det = n_src_to_det / np.linalg.norm(n_src_to_det)
    R_body_to_det = np.asarray(R_body_to_det, dtype=float)

    def integrand(x):
        r_vec = x * a_vec - config.R * n_src_to_det
        r_distance = np.linalg.norm(r_vec)

        h_body = get_metric_tensor_body_frame(r_distance, t + (config.L - x) / config.c, config)
        h_matrix = R_body_to_det @ h_body @ R_body_to_det.T

        h_tt = project_to_tt_gauge_dynamic(h_matrix, r_vec)
        
        val = a_vec.T @ h_tt @ a_vec
        return val / (2.0 * config.c)
        
    result, _ = quad(integrand, 0.0, config.L)
    return result


def calculate_metric_response(
    t: float,
    theta_src: Optional[float] = None,
    phi_src: Optional[float] = None,
    theta_rot: Optional[float] = None,
    phi_rot: Optional[float] = None,
) -> float:
    """
    Main entry function to compute the strain response at a given time.

    **Detector frame (requested convention).** The interferometer vertex is the origin
    of a right-handed Cartesian system with:

        - arm 1 along ``+x``,
        - arm 2 along ``+y``,
        - ``+z`` completing the triad.

    Each source is parameterized by four angles:

        1. ``(theta_src, phi_src)`` describe the **unit vector from the detector toward
           the source** using the same spherical convention as ``spherical_unit_vector``.
        2. ``(theta_rot, phi_rot)`` describe the **unit vector along the rotor symmetry
           axis** (the body ``+z`` direction) expressed in the detector frame.

    The quadrupole oscillation is still modeled in the rotating body frame, then
    rotated into the detector frame via ``rotation_body_to_detector``.

    Angle arguments default to values from ``Data/bestPosition.txt`` (read once and
    cached). Pass explicit angles to override; ``None`` means "use the cached value".
    """
    d1, d2, d3, d4 = _get_best_position_defaults()
    theta_src = d1 if theta_src is None else theta_src
    phi_src = d2 if phi_src is None else phi_src
    theta_rot = d3 if theta_rot is None else theta_rot
    phi_rot = d4 if phi_rot is None else phi_rot

    config = ExperimentConfig()

    # LIGO arm directions are fixed in the detector frame (no longer optimized).
    a_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    b_vec = np.array([0.0, 1.0, 0.0], dtype=float)

    n_src_to_det = spherical_unit_vector(theta_src, phi_src)
    R_body_to_det = rotation_body_to_detector(theta_rot, phi_rot)

    t_forward = t - 2.0 * config.L / config.c
    t_return = t - config.L / config.c

    delta_t1 = calculate_delta_t(t_forward, n_src_to_det, a_vec, config, R_body_to_det)
    delta_t_prime1 = calculate_delta_t_prime(t_return, n_src_to_det, a_vec, config, R_body_to_det)
    delay_of_transition_time1 = delta_t1 + delta_t_prime1

    delta_t2 = calculate_delta_t(t_forward, n_src_to_det, b_vec, config, R_body_to_det)
    delta_t_prime2 = calculate_delta_t_prime(t_return, n_src_to_det, b_vec, config, R_body_to_det)
    delay_of_transition_time2 = delta_t2 + delta_t_prime2

    delay_of_transition_time_delta = delay_of_transition_time1 - delay_of_transition_time2
    input_signal = delay_of_transition_time_delta * config.c / (2.0 * config.L)

    return float(input_signal)

# Alias to maintain backward compatibility for `from metricCalculate import main` in fourier.py
main = calculate_metric_response


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        epilog=(
            "example: python metricCalculate.py -t 0.01 "
            "-ts 3.1 -ps 0.0 -tr 1.57 -pr 0.0"
        )
    )

    parser.add_argument("-t", "--time", type=str, required=True, help="current time in seconds")
    parser.add_argument(
        "-ts",
        "--thetasource",
        type=str,
        required=True,
        help="polar angle (detector frame) of the vector from detector toward the source",
    )
    parser.add_argument(
        "-ps",
        "--phisource",
        type=str,
        required=True,
        help="azimuthal angle (detector frame) of the vector from detector toward the source",
    )
    parser.add_argument(
        "-tr",
        "--thetarotation",
        type=str,
        required=True,
        help="polar angle (detector frame) of the rotor symmetry axis (body +z)",
    )
    parser.add_argument(
        "-pr",
        "--phirotation",
        type=str,
        required=True,
        help="azimuthal angle (detector frame) of the rotor symmetry axis (body +z)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed output")
    parser.add_argument("-o", "--output", type=str, default=None, help="path for the output file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        print(f"Begin processing, current time: {args.time} s")
        if args.output:
            print(f"Output file: {args.output}")
    
    time_val = float(args.time)
    theta_src_val = float(args.thetasource)
    phi_src_val = float(args.phisource)
    theta_rot_val = float(args.thetarotation)
    phi_rot_val = float(args.phirotation)
    
    # Execute main calculation
    result = calculate_metric_response(
        time_val,
        theta_src_val,
        phi_src_val,
        theta_rot_val,
        phi_rot_val,
    )
    print(result)
    
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(str(result))
            print(f"\nResult saved at: {args.output}")
        except Exception as e:
            print(f"Error while saving file: {e}")
    
    sys.exit(0)