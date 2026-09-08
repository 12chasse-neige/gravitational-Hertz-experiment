"""Independently verify the oscillatory Newtonian rotor response.

This diagnostic compares an analytic mass-quadrupole force with direct force
integration over point holes and finite cylindrical holes.  It deliberately
does not import ``ghe.finite_distance``.  The observable is a free-mass,
instantaneous differential displacement divided by arm length, not a complete
optical readout or a retarded relativistic solution.

Run from the repository root in the gravitational-Hertz-experiment environment:

    python scr/verifyNearFieldNewtonian.py
    python scr/verifyNearFieldNewtonian.py --quick --output /tmp/check.json

The default run checks both the paper baseline and live YAML configuration,
at the cached legacy geometry and a source along +y with rotor axis +z.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import SourceConfig
from ghe.geometry import rotation_body_to_detector, spherical_unit_vector
from ghe.near_field import _cylinder_quadrature
from ghe.paths import BEST_POSITION_JSON_FILE, REPO_ROOT


ANGLE_NAMES = ("theta_src", "phi_src", "theta_rot", "phi_rot")
# A documented fallback makes this diagnostic reproducible without ignored data/.
LEGACY_ANGLES_20260908 = (
    0.6074123620484425,
    0.7914472879881271,
    0.3594716358447116,
    0.7941001629975755,
)


def complex_record(value: complex) -> dict[str, float]:
    """JSON representation of a peak phasor under exp(-i omega t)."""
    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "amplitude": float(abs(value)),
        "phase_rad": float(np.angle(value)),
    }


def endpoints(length: float) -> np.ndarray:
    """Vertex and the two end masses, in that order, in detector coordinates."""
    return np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0]])


def quadrupole_response(
    config: SourceConfig, source_position: np.ndarray, rotor_angles: np.ndarray
) -> tuple[complex, complex, np.ndarray, np.ndarray]:
    """Return strain, differential acceleration, all accelerations, and Q.

    For two diametrically opposed holes of signed mass m_h, the oscillatory
    STF moment is m_h s^2 [[1,i,0],[i,-1,0],[0,0,0]] in the body frame.
    The force follows by differentiating U_Q = 3G Q_ij r_i r_j / (2 r^5).
    """
    rotation = rotation_body_to_detector(*rotor_angles)
    hole_mass = -config.rho * np.pi * config.d**2 * config.H / 4.0
    q_body = hole_mass * config.s**2 * np.array(
        [[1.0, 1j, 0.0], [1j, -1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    q_detector = rotation @ q_body @ rotation.T
    relative_positions = endpoints(config.L) - source_position
    distance = np.linalg.norm(relative_positions, axis=1)
    qr = relative_positions @ q_detector.T
    rqr = np.einsum("ij,ij->i", relative_positions, qr)
    acceleration = (
        3.0 * config.G * qr / distance[:, None] ** 5
        - 7.5 * config.G * rqr[:, None] * relative_positions / distance[:, None] ** 7
    )
    differential_acceleration = (
        acceleration[1, 0] - acceleration[0, 0]
        - acceleration[2, 1] + acceleration[0, 1]
    )
    strain = -differential_acceleration / (config.gw_angular_frequency**2 * config.L)
    return strain, differential_acceleration, acceleration, q_detector


def direct_newtonian_response(
    config: SourceConfig,
    source_position: np.ndarray,
    rotor_angles: np.ndarray,
    *,
    finite_cylinders: bool,
    phase_samples: int,
    quadrature: tuple[int, int, int],
) -> tuple[complex, int]:
    """Integrate exact Newtonian forces and extract the 2*spin harmonic.

    The full uniform cylinder has stationary Eulerian density, so its 600-Hz
    force is zero.  Only the signed missing mass in the two holes is needed.
    Force sums and DC subtraction use 50-digit Decimal arithmetic to avoid
    subtractive loss: the quadrupole force is about 1e-7 of the hole monopole.
    Geometry/trigonometry/quadrature inputs and final Fourier sums are float64.
    """
    rotation = rotation_body_to_detector(*rotor_angles)
    if finite_cylinders:
        points, weights = _cylinder_quadrature(
            config.d / 2.0, config.H, config.s, 0.0, *quadrature
        )
        points = np.concatenate((points, -points), axis=0)
        weights = np.tile(-config.rho * weights, 2)
    else:
        points = np.array([[config.s, 0.0, 0.0], [-config.s, 0.0, 0.0]])
        hole_mass = -config.rho * np.pi * config.d**2 * config.H / 4.0
        weights = np.full(2, hole_mass)

    phase = 2.0 * np.pi * np.arange(phase_samples) / phase_samples
    with localcontext() as context:
        context.prec = 50
        decimal_weights = [Decimal(float(weight)) for weight in weights]
        positions = [
            [Decimal(float(value)) for value in point]
            for point in endpoints(config.L) - source_position
        ]
        gravitational_constant = Decimal(config.G)
        differential_samples: list[Decimal] = []
        for angle in phase:
            cosine, sine = np.cos(angle), np.sin(angle)
            spin = np.array(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            rotated_points = [
                [Decimal(float(value)) for value in point]
                for point in points @ spin.T @ rotation.T
            ]
            component_accelerations = []
            for endpoint_index, component in ((1, 0), (0, 0), (2, 1), (0, 1)):
                acceleration = Decimal(0)
                for weight, point in zip(decimal_weights, rotated_points):
                    separation = [
                        x - y for x, y in zip(positions[endpoint_index], point)
                    ]
                    radius_squared = sum(value * value for value in separation)
                    acceleration -= (
                        gravitational_constant * weight * separation[component]
                        / (radius_squared * radius_squared.sqrt())
                    )
                component_accelerations.append(acceleration)
            gx_end, gx_vertex, gy_end, gy_vertex = component_accelerations
            differential_samples.append(gx_end - gx_vertex - gy_end + gy_vertex)

        # Removing a constant cannot change a nonzero Fourier harmonic.
        samples_without_dc = np.array(
            [float(value - differential_samples[0]) for value in differential_samples]
        )
    differential_phasor = 2.0 * np.mean(samples_without_dc * np.exp(2j * phase))
    strain = -differential_phasor / (config.gw_angular_frequency**2 * config.L)
    return strain, len(weights)


def evaluate_case(
    name: str,
    geometry_name: str,
    config: SourceConfig,
    radius: float,
    angles: np.ndarray,
    settings: list[tuple[int, tuple[int, int, int]]],
) -> dict[str, object]:
    """Compare direct force integrals with the independent analytic result."""
    # Explicit position prevents SourceConfig.__post_init__ from changing R.
    source_position = radius * spherical_unit_vector(*angles[:2])
    h_quad, differential_acceleration, acceleration, q = quadrupole_response(
        config, source_position, angles[2:]
    )
    comparisons = []
    point_phases = settings[-1][0]
    jobs = [(False, point_phases, settings[0][1])]
    jobs.extend((True, phases, order) for phases, order in settings)
    previous_cylinder: complex | None = None
    for finite_cylinders, phases, order in jobs:
        direct_h, point_count = direct_newtonian_response(
            config, source_position, angles[2:], finite_cylinders=finite_cylinders,
            phase_samples=phases, quadrature=order,
        )
        relative_difference = (direct_h - h_quad) / h_quad
        record: dict[str, object] = {
            "density_model": "finite_cylindrical_holes" if finite_cylinders else "point_holes",
            "phase_samples_per_spin_period": phases,
            "quadrature_orders_radial_azimuthal_axial": list(order) if finite_cylinders else None,
            "number_of_mass_points": point_count,
            "strain_peak_phasor": complex_record(direct_h),
            "relative_complex_difference_from_quadrupole": complex_record(relative_difference),
        }
        if finite_cylinders:
            record["relative_change_from_previous_cylinder_setting"] = (
                float(abs((direct_h - previous_cylinder) / direct_h))
                if previous_cylinder is not None else None
            )
            previous_cylinder = direct_h
        comparisons.append(record)

    return {
        "scenario": name,
        "geometry": geometry_name,
        "source_config": asdict(config),
        "applied_arm_length_m": config.L,
        "applied_source_vertex_distance_m": radius,
        "source_position_m": source_position.tolist(),
        "angles_rad": dict(zip(ANGLE_NAMES, angles.tolist())),
        "source_to_vertex_xend_yend_distances_m": np.linalg.norm(
            endpoints(config.L) - source_position, axis=1
        ).tolist(),
        "hole_signed_mass_kg": -config.rho * np.pi * config.d**2 * config.H / 4.0,
        "quadrupole_peak_phasor_kg_m2": {"real": q.real.tolist(), "imag": q.imag.tolist()},
        "quadrupole_acceleration_phasors_vertex_xend_yend_m_s2": {
            "real": acceleration.real.tolist(), "imag": acceleration.imag.tolist()
        },
        "quadrupole_differential_acceleration_phasor_m_s2": complex_record(differential_acceleration),
        "quadrupole_strain_peak_phasor": complex_record(h_quad),
        "direct_integral_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/near-field-newtonian-check.json")
    parser.add_argument("--geometry-file", type=Path, default=BEST_POSITION_JSON_FILE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="one cylinder setting: (3,8,3), 32 phases")
    mode.add_argument("--extended", action="store_true", help="also use (6,24,6), 128 phases")
    args = parser.parse_args()

    if args.geometry_file.is_file():
        geometry_data = json.loads(args.geometry_file.read_text(encoding="utf-8"))
        cached_angles = np.array([float(geometry_data[key]) for key in ANGLE_NAMES])
        geometry_origin = str(args.geometry_file.resolve())
    else:
        cached_angles = np.array(LEGACY_ANGLES_20260908)
        geometry_origin = "embedded 2026-09-08 cached-geometry snapshot (cache file absent)"
    if not np.all(np.isfinite(cached_angles)):
        raise ValueError("Cached geometry contains non-finite angles")

    live_config = SourceConfig()
    if live_config.num != 2:
        raise ValueError("This independent check requires the configured two-hole rotor")
    settings = [(32, (3, 8, 3))]
    if not args.quick:
        settings.append((64, (4, 12, 4)))
    if args.extended:
        settings.append((128, (6, 24, 6)))

    cases = []
    for name, config, radius in (
        ("paper_baseline", SourceConfig(L=4000.0), 6000.0),
        ("live_yaml", live_config, live_config.R),
    ):
        for geometry_name, angles in (
            ("cached_legacy_geometry", cached_angles),
            ("source_plus_y_rotor_axis_plus_z", np.array([np.pi / 2, np.pi / 2, 0.0, 0.0])),
        ):
            result = evaluate_case(name, geometry_name, config, radius, angles, settings)
            cases.append(result)
            amplitude = result["quadrupole_strain_peak_phasor"]["amplitude"]
            print(f"{name}, {geometry_name}: |h_Newtonian| = {amplitude:.12e}")

    payload = {
        "definition": "h_N = -[(g_x(L,0,0)-g_x(0))-(g_y(0,L,0)-g_y(0))]/(omega_GW^2 L)",
        "phasor_convention": "peak complex amplitude; real signal = Re[phasor exp(-i omega_GW t)]; omega_GW=2 omega_rot",
        "units": {"strain": "dimensionless", "acceleration": "m/s^2", "quadrupole": "kg m^2"},
        "assumptions": [
            "Newtonian instantaneous force and free-mass endpoint response; no relativistic propagation or optical transfer",
            "vertex at origin, arms along +x and +y, with one shared freely moving vertex reference",
            "uniform axisymmetric rotor cylinder contributes only static gravity; holes carry the oscillatory density",
            "paper baseline fixes L=4000 m and explicit R=6000 m; other source properties use live YAML",
            "live scenario uses current SourceConfig L and R; the numerical geometry is recorded in each case",
            "rotor material stresses and environmental forces are outside this Newtonian validation",
        ],
        "analytic_method": "Q=m_h s^2 R[[1,i,0],[i,-1,0],[0,0,0]]R^T; g_Q=3G Qr/r^5-(15G/2)(r.Q.r)r/r^7",
        "direct_method": "Gauss-Legendre radius/axial plus periodic azimuthal mass quadrature; g=-G sum(m r/r^3); 2/N sum(g exp(2i spin_phase))",
        "precision": "Decimal 50 digits for force evaluation and DC subtraction; float64 geometry, quadrature, trigonometry and final Fourier sum",
        "convergence_mode": "extended" if args.extended else "quick" if args.quick else "default",
        "cached_geometry_origin": geometry_origin,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
