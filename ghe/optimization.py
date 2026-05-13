"""
Best-geometry optimization for a single source.

The optimizer searches over detector-frame source direction and rotor-axis
direction.  The objective is not a new physics model; it simply evaluates the
metric response at two quarter-phase samples and combines them into an amplitude
estimate.  Source-array generation later reuses this best geometry as the array
center reference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .config import SourceConfig
from .geometry import spherical_unit_vector
from .metric import calculate_metric_response
from .paths import BEST_POSITION_FILE, BEST_POSITION_JSON_FILE

SCALE_FACTOR = 1e38
FALLBACK_BEST_POSITION: tuple[float, float, float, float] = (0.1, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class BestGeometry:
    """
    Optimized detector-frame geometry and its signal amplitude.

    ``theta_src`` and ``phi_src`` point from the detector to the source.
    ``theta_rot`` and ``phi_rot`` point along the source rotor symmetry axis.
    """

    theta_src: float
    phi_src: float
    theta_rot: float
    phi_rot: float
    signal_amplitude: float

    @property
    def angles(self) -> tuple[float, float, float, float]:
        """Return the four angles in the order expected by metric functions."""

        return self.theta_src, self.phi_src, self.theta_rot, self.phi_rot

    @property
    def n_src_to_det_vec(self) -> np.ndarray:
        """Unit vector from detector vertex toward the source."""

        return spherical_unit_vector(self.theta_src, self.phi_src)

    @property
    def u_src_to_detector_center_vec(self) -> np.ndarray:
        """Unit vector from the reference source back toward the detector."""

        return -self.n_src_to_det_vec

    @property
    def rot_axis_vec(self) -> np.ndarray:
        """Rotor symmetry-axis unit vector in detector coordinates."""

        return spherical_unit_vector(self.theta_rot, self.phi_rot)


def get_signal_amplitude(
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
) -> float:
    """
    Estimate strain amplitude for one geometry.

    The signal is approximately sinusoidal at the quadrupole frequency.  Sampling
    at ``t=0`` and one eighth of the mechanical period gives two quadrature-like
    values whose Euclidean norm is the amplitude objective.
    """

    active_config = config or SourceConfig()
    period = 2.0 * np.pi / active_config.omega
    val1 = calculate_metric_response(
        0.0,
        theta_src,
        phi_src,
        theta_rot,
        phi_rot,
        config=active_config,
    )
    val2 = calculate_metric_response(
        period / 8.0,
        theta_src,
        phi_src,
        theta_rot,
        phi_rot,
        config=active_config,
    )
    return float(np.sqrt(val1**2 + val2**2))


def spherical_function(
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
) -> float:
    """Unscaled optimization objective in physical strain units."""

    return get_signal_amplitude(theta_src, phi_src, theta_rot, phi_rot, config=config)


def scaled_spherical_function(
    theta_src: float,
    phi_src: float,
    theta_rot: float,
    phi_rot: float,
    *,
    config: SourceConfig | None = None,
) -> float:
    """
    Scaled objective used by SciPy.

    Raw strains are extremely small, so scaling improves optimizer conditioning
    without moving the maximum.
    """

    return spherical_function(theta_src, phi_src, theta_rot, phi_rot, config=config) * SCALE_FACTOR


def scipy_gradient_descent(
    f_scaled: Callable[[float, float, float, float], float],
    init_theta_src: float,
    init_phi_src: float,
    init_theta_rot: float,
    init_phi_rot: float,
    *,
    fix_source_angles: bool = False,
) -> tuple[float, float, float, float]:
    """
    Maximize the scaled signal-amplitude objective over detector-frame angles.

    With ``fix_source_angles=True``, only rotor-axis angles are optimized.  That
    mode is used for source-array rows whose sky direction is fixed by lattice
    position.
    """

    if fix_source_angles:

        def negative_f(vars: np.ndarray) -> float:
            theta_rot, phi_rot = vars
            return -float(
                f_scaled(
                    float(init_theta_src),
                    float(init_phi_src),
                    float(theta_rot),
                    float(phi_rot),
                )
            )

        x0 = np.array([init_theta_rot, init_phi_rot], dtype=float)
        bounds = [(0.0, float(np.pi)), (0.0, float(2.0 * np.pi))]
    else:

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

        x0 = np.array([init_theta_src, init_phi_src, init_theta_rot, init_phi_rot], dtype=float)
        bounds = [
            (0.0, float(np.pi)),
            (0.0, float(2.0 * np.pi)),
            (0.0, float(np.pi)),
            (0.0, float(2.0 * np.pi)),
        ]

    result = minimize(
        negative_f,
        x0=x0,
        bounds=bounds,
        method="SLSQP",
        options={"disp": True, "ftol": 1e-6, "eps": 1e-5, "maxiter": 500},
    )

    if fix_source_angles:
        return (
            float(init_theta_src),
            float(init_phi_src),
            float(result.x[0]),
            float(result.x[1]),
        )

    return float(result.x[0]), float(result.x[1]), float(result.x[2]), float(result.x[3])


def parse_best_position_text(text: str) -> tuple[float, float, float, float] | None:
    """Parse the machine-readable ``BEST_POSITION`` line from a cache file."""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("BEST_POSITION:"):
            values = [x.strip() for x in stripped.split(":", 1)[1].split(",")]
            if len(values) == 4:
                return tuple(float(value) for value in values)  # type: ignore[return-value]
    return None


def load_best_geometry(path: str | Path = BEST_POSITION_FILE) -> BestGeometry | None:
    """Load cached best geometry and recompute its current amplitude."""

    input_path = Path(path)
    if not input_path.is_file():
        return None
    angles = parse_best_position_text(input_path.read_text(encoding="utf-8"))
    if angles is None:
        return None
    amplitude = spherical_function(*angles)
    return BestGeometry(*map(float, angles), signal_amplitude=float(amplitude))


def save_best_geometry(
    geometry: BestGeometry,
    path: str | Path = BEST_POSITION_FILE,
    json_path: str | Path | None = BEST_POSITION_JSON_FILE,
) -> None:
    """
    Save optimized geometry in both legacy text and optional JSON formats.

    The text format remains compatible with older scripts; JSON is easier for
    reproducible run directories and external tooling.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Detector frame: vertex at origin; arm1 +x; arm2 +y; +z completes RHS.\n"
        "# (theta_src, phi_src): unit vector from detector toward the source.\n"
        "# (theta_rot, phi_rot): rotor symmetry axis (body +z) in detector frame.\n"
        f"BEST_POSITION: {geometry.theta_src:.8f}, {geometry.phi_src:.8f}, "
        f"{geometry.theta_rot:.8f}, {geometry.phi_rot:.8f}\n"
        f"max_signal_amplitude: {geometry.signal_amplitude:.12e}\n",
        encoding="utf-8",
    )
    if json_path is not None:
        json_output_path = Path(json_path)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(asdict(geometry), indent=2), encoding="utf-8")


def optimize_best_geometry(
    initial_angles: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.0),
) -> BestGeometry:
    """Run the full four-angle optimization from a cold-start guess."""

    angles = scipy_gradient_descent(scaled_spherical_function, *initial_angles)
    amplitude = spherical_function(*angles)
    return BestGeometry(*map(float, angles), signal_amplitude=float(amplitude))


def solve_best_geometry(recompute: bool = False, path: str | Path = BEST_POSITION_FILE) -> BestGeometry:
    """
    Return cached geometry unless recomputation is requested or cache is missing.

    This is the package entry point used by source-array generation.
    """

    if not recompute:
        cached = load_best_geometry(path)
        if cached is not None:
            return cached

    geometry = optimize_best_geometry()
    save_best_geometry(geometry, path=path)
    return geometry
