import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.metricCalculate import calculate_metric_response, ExperimentConfig
from scipy.optimize import minimize

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_BEST_POSITION_FILE = _DATA_DIR / "bestPosition.txt"

angular_frequency = ExperimentConfig.omega
period = 2 * np.pi / angular_frequency


def get_the_signal_amplitude(theta_src: float, phi_src: float, theta_rot: float, phi_rot: float) -> float:
    """
    Scalar figure-of-merit for the optimizer: approximate GW strain amplitude.

    **Detector frame.** The interferometer arms are fixed along ``+x`` and ``+y`` in
    ``metricCalculate.calculate_metric_response``; only the source line-of-sight
    ``(theta_src, phi_src)`` and rotor axis ``(theta_rot, phi_rot)`` are varied here.
    """
    t1 = 0.0
    t2 = period / 8.0

    val1 = calculate_metric_response(t1, theta_src, phi_src, theta_rot, phi_rot)
    val2 = calculate_metric_response(t2, theta_src, phi_src, theta_rot, phi_rot)

    amplitude = np.sqrt(val1**2 + val2**2)
    return float(amplitude)


def spherical_function(theta_src: float, phi_src: float, theta_rot: float, phi_rot: float) -> float:
    """Unscaled objective (physical units)."""
    return get_the_signal_amplitude(theta_src, phi_src, theta_rot, phi_rot)


SCALE_FACTOR = 1e38


def scaled_spherical_function(theta_src: float, phi_src: float, theta_rot: float, phi_rot: float) -> float:
    """
    Scaled objective for numerical optimization.

    The raw strain values are tiny; multiplying by ``SCALE_FACTOR`` keeps gradients
    well-conditioned for SciPy without changing the location of the maximum.
    """
    return spherical_function(theta_src, phi_src, theta_rot, phi_rot) * SCALE_FACTOR


def scipy_gradient_descent(
    f_scaled,
    init_theta_src: float,
    init_phi_src: float,
    init_theta_rot: float,
    init_phi_rot: float,
) -> tuple[float, float, float, float]:
    """
    Maximize the strain amplitude over the four detector-frame angles.

    **Why this is simpler than the old six-angle version.** Arms are no longer free
    parameters: LIGO arm 1 is ``+x`` and arm 2 is ``+y`` by definition, so the only
    remaining orientation freedom is the source direction and the rotor axis.

    ``SLSQP`` is kept for familiarity, but the heavy orthogonality constraint on the
    arms is gone because orthogonality is now structural.
    """

    def negative_f(vars: np.ndarray) -> float:
        theta_src, phi_src, theta_rot, phi_rot = vars
        return -float(
            f_scaled(
                float(theta_src),
                float(phi_src),
                float(theta_rot),
                float(phi_rot),
            )
        )

    bounds = [
        (0.0, float(np.pi)),
        (0.0, float(2.0 * np.pi)),
        (0.0, float(np.pi)),
        (0.0, float(2.0 * np.pi)),
    ]

    result = minimize(
        negative_f,
        x0 = np.array(
             [init_theta_src, init_phi_src, init_theta_rot, init_phi_rot],
             dtype=float,
        ),
        bounds=bounds,
        method="SLSQP",
        options={"disp": True, "ftol": 1e-6, "eps": 1e-5, "maxiter": 500},
    )

    return float(result.x[0]), float(result.x[1]), float(result.x[2]), float(result.x[3])


if __name__ == "__main__":
    # Cold start: pick a sky location near the old default (source mostly along +z)
    # and a rotor axis mostly in the +z direction
    initial_theta_src = 0.1
    initial_phi_src = 0.0
    initial_theta_rot = 1.0
    initial_phi_rot = 0.0

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _BEST_POSITION_FILE.open("w", encoding="utf-8") as log_file:
        log_file.write(
            "# Detector frame: vertex at origin; arm1 +x; arm2 +y; +z completes RHS.\n"
            "# (theta_src, phi_src): unit vector from detector toward the source.\n"
            "# (theta_rot, phi_rot): rotor symmetry axis (body +z) in detector frame.\n"
            f"# Initial guess [rad]: theta_src={initial_theta_src:.8f}, phi_src={initial_phi_src:.8f}, "
            f"theta_rot={initial_theta_rot:.8f}, phi_rot={initial_phi_rot:.8f}\n"
        )

        best_theta_src, best_phi_src, best_theta_rot, best_phi_rot = scipy_gradient_descent(
            scaled_spherical_function,
            initial_theta_src,
            initial_phi_src,
            initial_theta_rot,
            initial_phi_rot,
        )

        true_max_value = spherical_function(
            best_theta_src,
            best_phi_src,
            best_theta_rot,
            best_phi_rot,
        )

        # Single machine-readable summary line (no duplicate "Location" block).
        log_file.write(
            f"BEST_POSITION: {best_theta_src:.8f}, {best_phi_src:.8f}, "
            f"{best_theta_rot:.8f}, {best_phi_rot:.8f}\n"
        )
        log_file.write(f"max_signal_amplitude: {true_max_value:.12e}\n")
