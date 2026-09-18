"""Compute one rotor's direct near-field metric tensor at the detector vertex."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghe.config import SourceConfig
from ghe.near_field import (
    NearFieldMetricResult,
    calculate_near_field_metric,
    calculate_quadrupole_gw_metric,
)
from ghe.optimization import BestGeometry, solve_best_geometry
from ghe.paths import BEST_POSITION_JSON_FILE, SINGLE_SOURCE_METRIC_FILE


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Directly integrate the moving-mass energy-momentum tensor of one "
            "perforated rotor at its cached best position."
        )
    )
    parser.add_argument("--time", type=float, default=0.0, help="detector time in seconds")
    parser.add_argument(
        "--output",
        type=Path,
        default=SINGLE_SOURCE_METRIC_FILE,
        help="JSON output path (default: data/single_source_metric.json)",
    )
    parser.add_argument("--radial-order", type=int, default=16)
    parser.add_argument("--azimuthal-order", type=int, default=64)
    parser.add_argument("--axial-order", type=int, default=16)
    return parser.parse_args()


def _load_best_geometry() -> BestGeometry:
    """Prefer the full-precision JSON cache, with the legacy solver as fallback."""

    if BEST_POSITION_JSON_FILE.is_file():
        try:
            values = json.loads(BEST_POSITION_JSON_FILE.read_text(encoding="utf-8"))
            geometry = BestGeometry(
                theta_src=float(values["theta_src"]),
                phi_src=float(values["phi_src"]),
                theta_rot=float(values["theta_rot"]),
                phi_rot=float(values["phi_rot"]),
                signal_amplitude=float(values["signal_amplitude"]),
            )
            if np.all(np.isfinite((*geometry.angles, geometry.signal_amplitude))):
                return geometry
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass
    return solve_best_geometry(recompute=False)


def _json_ready_result(
    result: NearFieldMetricResult,
    h_quadrupole_gw: np.ndarray,
    geometry: BestGeometry,
    config: SourceConfig,
    *,
    radial_order: int,
    azimuthal_order: int,
    axial_order: int,
) -> dict[str, object]:
    result_dict = asdict(result)
    for key, value in tuple(result_dict.items()):
        if isinstance(value, np.ndarray):
            result_dict[key] = value.tolist()
    h_covariant = result_dict.pop("h_covariant")
    direct_spatial_norm = float(np.linalg.norm(result.h_covariant[1:, 1:]))
    quadrupole_spatial_norm = float(np.linalg.norm(h_quadrupole_gw[1:, 1:]))

    return {
        "field_definition": (
            "h_mu_nu(t,x) = (4 G / c^4) integral_V "
            "T_mu_nu(t - |x|/c, y) / |x-y| d^3y"
        ),
        "quadrupole_gw_definition": (
            "h_ij^GW(t,r) = (2 G / c^4 r) d^2 I_ij(t-r/c) / dt^2; "
            "h_0mu^GW = h_mu0^GW = 0"
        ),
        "coordinate_convention": (
            "detector frame (ct,x,y,z), signature (-,+,+,+); vertex at origin; "
            "arm 1 +x; arm 2 +y"
        ),
        "tensor_model": (
            "moving rest-mass contribution for a solid cylinder minus its "
            "cylindrical holes; elastic/internal stresses are not included"
        ),
        "energy_momentum_model": (
            "T^mu_nu dV = rho gamma (c,v)^mu (c,v)^nu dV for rest-volume "
            "elements, lowered with eta_mu_nu = diag(-1,1,1,1)"
        ),
        "tensor_units": "dimensionless",
        "gauge_projection": "none for either tensor; both formulas are stored before TT projection",
        "best_position": {
            "theta_src_rad": geometry.theta_src,
            "phi_src_rad": geometry.phi_src,
            "theta_rot_rad": geometry.theta_rot,
            "phi_rot_rad": geometry.phi_rot,
        },
        "source_config": asdict(config),
        "quadrature": {
            "scheme": "Gauss-Legendre in radius/z and periodic trapezoid in azimuth",
            "radial_order": radial_order,
            "azimuthal_order": azimuthal_order,
            "axial_order": axial_order,
        },
        **result_dict,
        "h_mu_nu": h_covariant,
        "h_quadrupole_gw_mu_nu": h_quadrupole_gw.tolist(),
        "comparison": {
            "direct_spatial_frobenius_norm": direct_spatial_norm,
            "quadrupole_gw_spatial_frobenius_norm": quadrupole_spatial_norm,
            "quadrupole_to_direct_spatial_norm_ratio": (
                quadrupole_spatial_norm / direct_spatial_norm
                if direct_spatial_norm > 0.0
                else None
            ),
        },
    }


def main() -> None:
    args = parse_arguments()
    config = SourceConfig()
    geometry = _load_best_geometry()
    result = calculate_near_field_metric(
        args.time,
        *geometry.angles,
        config=config,
        radial_order=args.radial_order,
        azimuthal_order=args.azimuthal_order,
        axial_order=args.axial_order,
    )
    h_quadrupole_gw = calculate_quadrupole_gw_metric(
        args.time,
        geometry.theta_rot,
        geometry.phi_rot,
        config=config,
    )
    payload = _json_ready_result(
        result,
        h_quadrupole_gw,
        geometry,
        config,
        radial_order=args.radial_order,
        azimuthal_order=args.azimuthal_order,
        axial_order=args.axial_order,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("h_mu_nu at the detector vertex:")
    print(np.array2string(result.h_covariant, precision=12, suppress_small=False))
    print("Quadrupole-only h_mu_nu^GW for comparison:")
    print(np.array2string(h_quadrupole_gw, precision=12, suppress_small=False))
    print(f"Saved metric tensor to {args.output}")


if __name__ == "__main__":
    main()
