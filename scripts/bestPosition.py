from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import DATA_DIR, BEST_POSITION_FILE, ExperimentConfig
from ghe.optimization import (
    BestGeometry,
    get_signal_amplitude as get_the_signal_amplitude,
    save_best_geometry,
    scaled_spherical_function,
    scipy_gradient_descent,
    spherical_function,
)

angular_frequency = ExperimentConfig.omega
period = 2 * np.pi / angular_frequency


if __name__ == "__main__":
    initial_theta_src = 1.0
    initial_phi_src = 0.0
    initial_theta_rot = 1.0
    initial_phi_rot = 0.0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    save_best_geometry(
        BestGeometry(
            best_theta_src,
            best_phi_src,
            best_theta_rot,
            best_phi_rot,
            true_max_value,
        ),
        path=BEST_POSITION_FILE,
    )
