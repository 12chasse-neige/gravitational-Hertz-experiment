import numpy as np
from metricCalculate import main, omega
from scipy.optimize import minimize

frequency = float(omega)
period = 2 * np.pi / frequency

def get_the_signal_amplitude(theta_arm, phi_arm, theta_det, phi_det):
    sampling = 100
    time_interval = period / sampling
    total_amplitude = 0
    for i in range(sampling):
        total_amplitude += main(i * time_interval, theta_arm, phi_arm, theta_det, phi_det) ** 2
    amplitude = np.sqrt(total_amplitude / sampling)
    return amplitude

def spherical_function(theta_arm, phi_arm, theta_det, phi_det):
    val = get_the_signal_amplitude(theta_arm, phi_arm, theta_det, phi_det)
    return float(val)

# 2. A SCALED version of your function for the optimizer
SCALE_FACTOR = 1e38

def scaled_spherical_function(theta_arm, phi_arm, theta_det, phi_det):
    """
    Returns values roughly around 1.0 to 10.0 so the 
    Gradient Descent math actually works.
    """
    return spherical_function(theta_arm, phi_arm, theta_det, phi_det) * SCALE_FACTOR


# Optimization (Using SciPy)
def scipy_gradient_descent(f_scaled, init_theta_arm, init_phi_arm, init_theta_det, init_phi_det):
    # Minimize the negative of the SCALED function
    def negative_f(vars):
        theta_arm, phi_arm, theta_det, phi_det = vars
        return -f_scaled(theta_arm, phi_arm, theta_det, phi_det)
    
    bounds =[(0, np.pi), (0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi)]
    
    result = minimize(
        negative_f, 
        x0=[init_theta_arm, init_phi_arm, init_theta_det, init_phi_det], 
        bounds=bounds, 
        method='L-BFGS-B',
        # Optional: tighten the tolerances if needed
        options={'ftol': 1e-9, 'gtol': 1e-9} 
    )
    
    return result.x[0], result.x[1],result.x[2], result.x[3]


# Run and Test
if __name__ == "__main__":
    initial_theta_arm = 1
    initial_phi_arm = 1
    initial_theta_det = 1
    initial_phi_det = 1

    # Open the log file for writing
    with open("bestPosition.log", "w") as log_file:
        log_file.write(f"Initial Guess: theta arm = {initial_theta_arm:.4f}, phi arm = {initial_phi_arm:.4f}, theta detector = {initial_theta_det:.4f}, phi detector = {initial_phi_det:.4f}\n")

        # 1. Run the optimizer on the SCALED function
        best_theta_arm, best_phi_arm, best_theta_det, best_phi_det = scipy_gradient_descent(scaled_spherical_function, initial_theta_arm, initial_phi_arm, initial_theta_det, initial_phi_det)

        # 2. Evaluate the true max value using the ORIGINAL function
        true_max_value = spherical_function(best_theta_arm, best_phi_arm, best_theta_det, best_phi_det)

        log_file.write("-" * 50 + "\n")
        log_file.write("SciPy Optimization Results:\n")
        log_file.write(f"Location: theta arm = {best_theta_arm:.4f}, phi arm = {best_phi_arm:.4f}, theta det = {best_theta_det:.4f}, phi det = {best_phi_det:.4f}\n")
        # Use scientific notation (:.6e) to print the 10^-38 number properly!
        log_file.write(f"Max Value: {true_max_value:.6e}\n")