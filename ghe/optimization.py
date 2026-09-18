"""
Best-geometry optimization for a single source.

The optimizer searches source and rotor directions to maximize the complete
complex detector response magnitude. Source-array generation later reuses the
best geometry as its array-center reference.
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
from .metric import calculate_response_phasor
from .artifacts import model_metadata, validate_metadata, read_metadata, write_metadata
from .paths import BEST_POSITION_FILE


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
    Return the peak dimensionless amplitude abs(H) for the supplied geometry.
    """

    return float(
        abs(
            calculate_response_phasor(
                theta_src, phi_src, theta_rot, phi_rot, config=config
            )
        )
    )


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

    cfg = config or SourceConfig()
    # A dimensional near-zone scale keeps objectives near order unity. This is
    # conditioning only: it does not change which physical amplitude is largest.
    hole_mass = cfg.rho * np.pi * cfg.d**2 * cfg.H / 4
    scale = cfg.G * hole_mass * cfg.s**2 / (cfg.gw_angular_frequency**2 * cfg.R**5)
    return (
        spherical_function(theta_src, phi_src, theta_rot, phi_rot, config=cfg) / scale
    )


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

        x0 = np.array(
            [init_theta_src, init_phi_src, init_theta_rot, init_phi_rot], dtype=float
        )
        bounds = [
            (0.0, float(np.pi)),
            (0.0, float(2.0 * np.pi)),
            (0.0, float(np.pi)),
            (0.0, float(2.0 * np.pi)),
        ]

    starts = [x0]
    if fix_source_angles:
        # A rotor-axis pole can be a stationary point in spherical coordinates.
        # Seed distinct axes so a vanishing angular gradient does not trap every
        # exact/anchor optimization at the reference orientation.
        starts += [
            np.array([0.0, 0.0]),
            np.array([np.pi / 2, 0.0]),
            np.array([np.pi / 2, np.pi / 2]),
        ]
    results = [
        minimize(
            negative_f,
            x0=start,
            bounds=bounds,
            method="SLSQP",
            options={"disp": False, "ftol": 1e-10, "eps": 1e-7, "maxiter": 500},
        )
        for start in starts
    ]
    successful = [
        result for result in results if result.success and np.isfinite(result.fun)
    ]
    if not successful:
        raise RuntimeError(
            "Geometry optimization failed: "
            + "; ".join(str(r.message) for r in results)
        )
    result = min(successful, key=lambda result: result.fun)
    best_seed = min(starts, key=negative_f)
    if negative_f(best_seed) < result.fun:
        result.x = best_seed

    if fix_source_angles:
        return (
            float(init_theta_src),
            float(init_phi_src),
            float(result.x[0]),
            float(result.x[1]),
        )

    return (
        float(result.x[0]),
        float(result.x[1]),
        float(result.x[2]),
        float(result.x[3]),
    )


def parse_best_position_text(text: str) -> tuple[float, float, float, float] | None:
    """Parse the machine-readable ``BEST_POSITION`` line from a cache file."""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("BEST_POSITION:"):
            values = [x.strip() for x in stripped.split(":", 1)[1].split(",")]
            if len(values) == 4:
                return tuple(float(value) for value in values)  # type: ignore[return-value]
    return None


def load_best_geometry(
    path: str | Path = BEST_POSITION_FILE, *, config: SourceConfig | None = None
) -> BestGeometry | None:
    """Load cached best geometry and recompute its current amplitude."""

    input_path = Path(path)
    if not input_path.is_file():
        return None
    try:
        validate_metadata(read_metadata(input_path), config)
    except ValueError:
        return None  # solve_best_geometry will recompute, never reinterpret old angles.
    angles = parse_best_position_text(input_path.read_text(encoding="utf-8"))
    if angles is None:
        return None
    amplitude = spherical_function(*angles, config=config)
    return BestGeometry(*map(float, angles), signal_amplitude=float(amplitude))


def save_best_geometry(
    geometry: BestGeometry,
    path: str | Path = BEST_POSITION_FILE,
    json_path: str | Path | None = None,
    *,
    config: SourceConfig | None = None,
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
        f"BEST_POSITION: {geometry.theta_src:.17g}, {geometry.phi_src:.17g}, "
        f"{geometry.theta_rot:.17g}, {geometry.phi_rot:.17g}\n"
        f"max_signal_amplitude: {geometry.signal_amplitude:.12e}\n",
        encoding="utf-8",
    )
    write_metadata(output_path, model_metadata(config))
    # Derive the JSON destination from the requested cache path; custom runs
    # must not overwrite the repository's default cache as a side effect.
    json_path = json_path or output_path.with_suffix(".json")
    if json_path is not None:
        json_output_path = Path(json_path)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(
            json.dumps(
                {**asdict(geometry), "metadata": model_metadata(config)}, indent=2
            ),
            encoding="utf-8",
        )


def optimize_best_geometry(
    initial_angles: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.0),
    *,
    config: SourceConfig | None = None,
) -> BestGeometry:
    """Deterministic multistart maximum of the complete response amplitude.

    Include both arm-extension geometries and an off-axis start so an obsolete
    radiative optimum cannot dictate the new search. Singular/inside-source
    trial points are infeasible, rather than attractive optimization targets.
    """
    cfg = config or SourceConfig()

    def objective(ts, ps, tr, pr):
        try:
            return scaled_spherical_function(ts, ps, tr, pr, config=cfg)
        except ValueError:
            return -1e100

    starts = [
        initial_angles,
        (np.pi / 2, 0.0, 0.0, 0.0),
        (np.pi / 2, np.pi / 2, 0.0, 0.0),
        (0.6, 0.8, 0.4, 0.8),
    ]
    candidates = []
    for start in starts:
        if objective(*start) < 0:
            continue
        # Keep the starting point too: finite-difference stopping criteria must
        # never make the returned amplitude worse than a feasible seed.
        candidates.append(start)
        try:
            candidates.append(scipy_gradient_descent(objective, *start))
        except RuntimeError:
            continue
    if not candidates:
        raise RuntimeError("No feasible source geometry found for this configuration")
    angles = max(candidates, key=lambda angles: objective(*angles))
    return BestGeometry(
        *map(float, angles), signal_amplitude=get_signal_amplitude(*angles, config=cfg)
    )


def solve_best_geometry(
    recompute: bool = False,
    path: str | Path = BEST_POSITION_FILE,
    *,
    config: SourceConfig | None = None,
) -> BestGeometry:
    """Reuse a compatible cache or optimize and save the active configuration."""
    if not recompute:
        cached = load_best_geometry(path, config=config)
        if cached is not None:
            return cached
    geometry = optimize_best_geometry(config=config)
    save_best_geometry(geometry, path=path, config=config)
    return geometry
