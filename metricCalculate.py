import numpy as np
import re
import os
from pathlib import Path
from scipy.integrate import quad
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

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
    R: float = 5000.0          # distance from source to detector (meter)
    rho: float = 1750.0        # density (kg/m^3)
    G: float = 6.674e-11       # gravitational constant
    c: float = 2.998e8         # speed of light
    omega: float = 300.0 * 2.0 * np.pi  # rotation frequency (rad/s)
    L: float = field(default_factory=lambda: float(os.getenv("LIGO_ARM_LENGTH", "1000.0")))  # length of the arm of the detector (meter)


# Cached angles from Data/bestPosition.txt (filled on first use; file is read at most once).
_BEST_POSITION_CACHE: Optional[Tuple[float, float, float, float, float, float]] = None

_FALLBACK_BEST_POSITION: Tuple[float, float, float, float, float, float] = (
    1.5708,
    0.1938,
    3.1416,
    2.3020,
    0.9795,
    2.2033,
)


def _parse_best_position_file_text(text: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Parse BEST_POSITION line or legacy 'Location:' line from bestPosition.txt."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("BEST_POSITION:"):
            rest = s.split(":", 1)[1].strip()
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) == 6:
                return (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
    for line in text.splitlines():
        if "Location:" in line and "theta arm 1" in line:
            nums = re.findall(r"=\s*([\d.+-eE]+)", line)
            if len(nums) >= 6:
                return (
                    float(nums[0]),
                    float(nums[1]),
                    float(nums[2]),
                    float(nums[3]),
                    float(nums[4]),
                    float(nums[5]),
                )
    return None


def _get_best_position_defaults() -> Tuple[float, float, float, float, float, float]:
    """Return cached angles from Data/bestPosition.txt; read the file at most once per process."""
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


def get_metric_tensor(r: float, t: float, config: ExperimentConfig) -> np.ndarray:
    """
    Calculate the raw metric tensor h_ij before TT gauge projection.
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


def calculate_delta_t(t: float, n_vec: np.ndarray, a_vec: np.ndarray, config: ExperimentConfig) -> float:
    """
    Calculate the forward photon transition time delay.
    """
    def integrand(x):
        r_vec = config.R * n_vec + x * a_vec
        r_distance = np.linalg.norm(r_vec)
        
        h_matrix = get_metric_tensor(r_distance, t + x / config.c, config)
        
        h_tt = project_to_tt_gauge_dynamic(h_matrix, r_vec)
        
        # Calculate a_i * a_j * h_tt_ij using dot products
        val = a_vec.T @ h_tt @ a_vec
        return val / (2.0 * config.c)
        
    # scipy.integrate.quad returns a tuple (integral_result, absolute_error)
    result, _ = quad(integrand, 0.0, config.L)
    return result


def calculate_delta_t_prime(t: float, n_vec: np.ndarray, a_vec: np.ndarray, config: ExperimentConfig) -> float:
    """
    Calculate the return photon transition time delay.
    """
    def integrand(x):
        r_vec = config.R * n_vec + x * a_vec
        r_distance = np.linalg.norm(r_vec)
        
        h_matrix = get_metric_tensor(r_distance, t + (config.L - x) / config.c, config)
        
        h_tt = project_to_tt_gauge_dynamic(h_matrix, r_vec) 
        
        val = a_vec.T @ h_tt @ a_vec
        return val / (2.0 * config.c)
        
    result, _ = quad(integrand, 0.0, config.L)
    return result


def calculate_metric_response(
    t: float,
    theta_arm1: Optional[float] = None,
    phi_arm1: Optional[float] = None,
    theta_arm2: Optional[float] = None,
    phi_arm2: Optional[float] = None,
    theta_det: Optional[float] = None,
    phi_det: Optional[float] = None,
) -> float:
    """
    Main entry function to compute the signal response at a given time.
    Calculates the relative transition time delay of the two arms.

    Angle arguments default to values from Data/bestPosition.txt (read once and cached).
    Pass explicit angles to override; None means use the cached best-position value for that angle.
    """
    d1, d2, d3, d4, d5, d6 = _get_best_position_defaults()
    theta_arm1 = d1 if theta_arm1 is None else theta_arm1
    phi_arm1 = d2 if phi_arm1 is None else phi_arm1
    theta_arm2 = d3 if theta_arm2 is None else theta_arm2
    phi_arm2 = d4 if phi_arm2 is None else phi_arm2
    theta_det = d5 if theta_det is None else theta_det
    phi_det = d6 if phi_det is None else phi_det

    config = ExperimentConfig()
    
    # Calculate orientation vectors using NumPy arrays
    a_vec = np.array([np.sin(theta_arm1) * np.cos(phi_arm1), 
                      np.sin(theta_arm1) * np.sin(phi_arm1), 
                      np.cos(theta_arm1)])
    
    b_vec = np.array([np.sin(theta_arm2) * np.cos(phi_arm2), 
                      np.sin(theta_arm2) * np.sin(phi_arm2), 
                      np.cos(theta_arm2)])
             
    n_vec = np.array([np.sin(theta_det) * np.cos(phi_det), 
                      np.sin(theta_det) * np.sin(phi_det), 
                      np.cos(theta_det)])

    t_forward = t - 2.0 * config.L / config.c
    t_return = t - config.L / config.c

    delta_t1 = calculate_delta_t(t_forward, n_vec, a_vec, config)
    delta_t_prime1 = calculate_delta_t_prime(t_return, n_vec, a_vec, config)
    delay_of_transition_time1 = delta_t1 + delta_t_prime1

    delta_t2 = calculate_delta_t(t_forward, n_vec, b_vec, config)
    delta_t_prime2 = calculate_delta_t_prime(t_return, n_vec, b_vec, config)
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
        epilog='example: python metricCalculate.py -t 0.01 -ta 1.6882 -pa 0.7850 -td 0.0000 -pd 1.1981'
    )
    
    parser.add_argument('-t', '--time', type=str, required=True, help='current time in seconds')
    parser.add_argument('-ta1', '--thetaarm1', type=str, required=True, help='polar angle of the detector arm 1')
    parser.add_argument('-pa1', '--phiarm1', type=str, required=True, help='azimuthal angle of the detector arm 1')
    parser.add_argument('-ta2', '--thetaarm2', type=str, required=True, help='polar angle of the detector arm 2')
    parser.add_argument('-pa2', '--phiarm2', type=str, required=True, help='azimuthal angle of the detector arm 2')
    parser.add_argument('-td', '--thetadetector', type=str, required=True, help='polar angle of the detector')
    parser.add_argument('-pd', '--phidetector', type=str, required=True, help='azimuthal angle of the detector')
    parser.add_argument('-v', '--verbose', action='store_true', help='show detailed output')
    parser.add_argument('-o', '--output', type=str, default=None, help='path for the output file')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        print(f"Begin processing, current time: {args.time} s")
        if args.output:
            print(f"Output file: {args.output}")
    
    time_val = float(args.time)
    theta_arm1_val = float(args.thetaarm1)
    phi_arm1_val = float(args.phiarm1)
    theta_arm2_val = float(args.thetaarm2)
    phi_arm2_val = float(args.phiarm2)
    theta_det_val = float(args.thetadetector)
    phi_det_val = float(args.phidetector)
    
    # Execute main calculation
    result = calculate_metric_response(time_val, theta_arm1_val, phi_arm1_val, theta_arm2_val, phi_arm2_val, theta_det_val, phi_det_val)
    print(result)
    
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(str(result))
            print(f"\nResult saved at: {args.output}")
        except Exception as e:
            print(f"Error while saving file: {e}")
    
    sys.exit(0)