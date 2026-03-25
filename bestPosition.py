import numpy as np
from metricCalculate import calculate_metric_response, ExperimentConfig
from scipy.optimize import minimize

frequency = ExperimentConfig.omega
period = 2 * np.pi / frequency

def get_the_signal_amplitude(theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det):
    """
    Get the amplitude of the signal
    """
    t1 = 0.0
    t2 = period / 8.0 
    
    val1 = calculate_metric_response(t1, theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det)
    val2 = calculate_metric_response(t2, theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det)
    
    amplitude = np.sqrt(val1**2 + val2**2)
    return amplitude

def spherical_function(theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det):
    val = get_the_signal_amplitude(theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det)
    return float(val)

SCALE_FACTOR = 1e38

def scaled_spherical_function(theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det):
    """
    Returns values roughly around 1.0 to 10.0 so the 
    Gradient Descent math actually works.
    """
    return spherical_function(theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det) * SCALE_FACTOR

def constraint_orthogonality(x):
    """
    The constraint function, making sure that the two arms are orthogonal.
    """
    th1, ph1, th2, ph2, thd, phd = x
    a1 = np.array([np.sin(th1)*np.cos(ph1), np.sin(th1)*np.sin(ph1), np.cos(th1)])
    a2 = np.array([np.sin(th2)*np.cos(ph2), np.sin(th2)*np.sin(ph2), np.cos(th2)])
    return np.dot(a1, a2)


def scipy_gradient_descent(f_scaled, init_theta_arm1, init_phi_arm1, init_theta_arm2, init_phi_arm2, init_theta_det, init_phi_det):
    # Minimize the negative of the SCALED function
    def negative_f(vars):
        theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det = vars
        return -f_scaled(theta_arm1, phi_arm1, theta_arm2, phi_arm2, theta_det, phi_det)
    
    bounds =[(0, np.pi), (0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi)]

    constraints = {'type': 'eq', 'fun': constraint_orthogonality}
    
    result = minimize(
        negative_f, 
        x0 = [init_theta_arm1, init_phi_arm1, init_theta_arm2, init_phi_arm2, init_theta_det, init_phi_det], 
        bounds = bounds, 
        method = 'SLSQP',
        constraints = constraints,
        options={'disp': True, 'ftol': 1e-15, 'eps': 1e-3, 'maxiter': 200}
    )
    
    return result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], result.x[5]


if __name__ == "__main__":
    # initial guess
    initial_theta_arm1 = np.pi/2
    initial_phi_arm1 = 0.0
    initial_theta_arm2 = np.pi/2
    initial_phi_arm2 = np.pi/2
    initial_theta_det = 1
    initial_phi_det = 1

    # Open the log file for writing
    with open("Data/bestPosition.log", "w") as log_file:
        log_file.write(f"Initial Guess: theta arm 1 = {initial_theta_arm1:.4f}, phi arm 1 = {initial_phi_arm1:.4f}, theta arm 2 = {initial_theta_arm2:.4f}, phi arm 2 = {initial_phi_arm2:.4f}, theta detector = {initial_theta_det:.4f}, phi detector = {initial_phi_det:.4f}\n")

        best_theta_arm1, best_phi_arm1, best_theta_arm2, best_phi_arm2, best_theta_det, best_phi_det = scipy_gradient_descent(scaled_spherical_function, initial_theta_arm1, initial_phi_arm1, initial_theta_arm2, initial_phi_arm2, initial_theta_det, initial_phi_det)

        true_max_value = spherical_function(best_theta_arm1, best_phi_arm1, best_theta_arm2, best_phi_arm2, best_theta_det, best_phi_det)

        log_file.write("-" * 50 + "\n")
        log_file.write("SciPy Optimization Results:\n")
        log_file.write(f"Location: theta arm 1 = {best_theta_arm1:.4f}, phi arm 1 = {best_phi_arm1:.4f}, theta arm 2 = {best_theta_arm2:.4f}, phi arm 2 = {best_phi_arm2:.4f}, theta det = {best_theta_det:.4f}, phi det = {best_phi_det:.4f}\n")
        log_file.write(f"Max Value: {true_max_value:.6e}\n")