"""Independent finite-hole Newtonian force reference (not a GR metric)."""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss


def _cylinder_quadrature(
    radius: float,
    height: float,
    center_x: float,
    center_y: float,
    radial_order: int,
    azimuthal_order: int,
    axial_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-frame points and volume weights for one cylinder."""

    radial_nodes, radial_weights = leggauss(radial_order)
    axial_nodes, axial_weights = leggauss(axial_order)

    radii = 0.5 * radius * (radial_nodes + 1.0)
    radial_weights = 0.5 * radius * radial_weights * radii
    z_values = 0.5 * height * axial_nodes
    axial_weights = 0.5 * height * axial_weights
    azimuths = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    azimuth_weight = 2.0 * np.pi / azimuthal_order

    rr, pp, zz = np.meshgrid(radii, azimuths, z_values, indexing="ij")
    points = np.column_stack(
        (
            center_x + (rr * np.cos(pp)).ravel(),
            center_y + (rr * np.sin(pp)).ravel(),
            zz.ravel(),
        )
    )
    weights = (
        radial_weights[:, None, None]
        * np.full((1, azimuthal_order, 1), azimuth_weight)
        * axial_weights[None, None, :]
    ).ravel()
    return points, weights
