"""Time-domain hole sum, independent of the production phasor formula."""

import numpy as np
from ghe.config import SourceConfig


def get_hole_coordinate(k: int, t: float, config: SourceConfig) -> tuple[float, float]:
    """
    Return the body-frame position of one hole center at time ``t``.

    The holes rotate in the body ``x-y`` plane with angular velocity
    ``config.omega``.  ``k`` indexes equally spaced holes around the rotor.
    """

    x_k = config.s * np.cos(config.omega * t + k * (2 * np.pi / config.num))
    y_k = config.s * np.sin(config.omega * t + k * (2 * np.pi / config.num))
    return float(x_k), float(y_k)


def calculate_whole_tensor(t: float, config: SourceConfig) -> np.ndarray:
    """
    Build the traceless quadrupole tensor ``I_ij`` in the source body frame.

    Holes are represented as missing mass, hence the negative mass contribution.
    The tensor is still in the frame where the rotor symmetry axis is ``+z``.
    """

    tensor = np.zeros((3, 3))
    volume = np.pi * config.d**2 / 4.0 * config.H
    mass = -config.rho * volume

    for k in range(config.num):
        x, y = get_hole_coordinate(k, t, config)
        coords = np.array([x, y, 0.0])
        r_squared = x**2 + y**2
        tensor += mass * (
            np.outer(coords, coords) - (1.0 / 3.0) * np.eye(3) * r_squared
        )

    return tensor
