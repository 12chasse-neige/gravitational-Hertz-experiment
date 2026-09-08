"""Reproduce the finite-distance quadrupole diagnostics; no legacy caches updated.

Run from the repository root in the gravitational-Hertz-experiment environment.
The SNR values are conditional ideal-quantum-noise proxies, not complete detector
forecasts. Source coordinates are passed explicitly, avoiding the legacy R reset.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import gwinc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghe.config import DetectorConfig, NoiseConfig, SourceConfig
from ghe.finite_distance import michelson_response, quadrupole_field, rotor_quadrupole_phasor
from ghe.geometry import spherical_unit_vector
from ghe.metric import calculate_metric_response
from ghe.noise import get_noise_psd
from ghe.paths import YEAR_SECONDS


def phasor_record(value):
    return {"real": float(value.real), "imag": float(value.imag),
            "abs": float(abs(value)), "phase_rad": float(np.angle(value))}


def one_case(name, cfg, angles):
    center = cfg.R*spherical_unit_vector(*angles[:2])
    Q = rotor_quadrupole_phasor(*angles[2:], config=cfg)
    kw = dict(arm_length=cfg.L, G=cfg.G, c=cfg.c)
    response = michelson_response(center, Q, cfg.gw_angular_frequency, **kw)
    refined = michelson_response(center, Q, cfg.gw_angular_frequency,
                                 quadrature_order=96, **kw)
    old = (calculate_metric_response(0, *angles, config=cfg)
           + 1j*calculate_metric_response(np.pi/(2*cfg.gw_angular_frequency),
                                         *angles, config=cfg))
    detector = DetectorConfig().with_source(cfg)
    noise = NoiseConfig()
    asd = float(np.sqrt(get_noise_psd(np.array([cfg.gw_frequency_hz]),
                                    noise_config=noise, detector_config=detector)[0]))
    totals = {key: phasor_record(val) for key, val in response.items()}
    totals["legacy_radiative_local_TT"] = phasor_record(old)
    snrs = {key: abs(value)*np.sqrt(YEAR_SECONDS)/asd
            for key, value in [("full", response["total"]),
                               ("newtonian", response["newtonian_instantaneous"]),
                               ("legacy_radiative", old)]}
    mass = cfg.rho*np.pi*cfg.H*((cfg.D/2)**2-cfg.num*(cfg.d/2)**2)
    field_vertex = quadrupole_field(-center, Q, cfg.gw_angular_frequency,
                                    G=cfg.G, c=cfg.c)
    return {
        "name": name, "source_config": asdict(cfg), "detector_config": asdict(detector),
        "noise_config": asdict(noise), "angles_rad": list(angles),
        "source_position_m": center.tolist(),
        "mirror_distances_m": [float(np.linalg.norm(p-center))
                               for p in [np.zeros(3), [cfg.L,0,0], [0,cfg.L,0]]],
        "kR": cfg.gw_angular_frequency*cfg.R/cfg.c,
        "kL": cfg.gw_angular_frequency*cfg.L/cfg.c,
        "rotor_mass_kg": mass, "hole_mass_magnitude_kg": cfg.rho*np.pi*cfg.d**2*cfg.H/4,
        "quadrupole_scale_kg_m2": cfg.rho*np.pi*cfg.d**2*cfg.H/4*cfg.s**2,
        "physical_static_monopole_h00_at_vertex": 2*cfg.G*mass/(cfg.c**2*cfg.R),
        "oscillatory_quadrupole_h00_at_vertex": phasor_record(field_vertex["metric"][0,0]),
        "responses": totals, "effective_DARM_peak_m": abs(response["total"])*cfg.L,
        "full_over_legacy_amplitude": abs(response["total"])/abs(old),
        "full_over_newtonian_amplitude_minus_one": abs(response["total"])/abs(response["newtonian_instantaneous"])-1,
        "full_over_newtonian_phase_rad": float(np.angle(response["total"]/response["newtonian_instantaneous"])),
        "ideal_quantum_asd_per_sqrt_Hz": asd,
        "ideal_quantum_noise_proxy_snr_one_year": snrs,
        "ideal_equal_response_source_count_for_snr5": 5/snrs["full"],
        "dual_gauge_relative_error": abs(response["total"]-response["curvature_total"])/abs(response["total"]),
        "quadrature_48_to_96_relative_error": abs(refined["total"]-response["total"])/abs(response["total"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"docs/near-field-analysis-results.json")
    parser.add_argument("--geometry-file", type=Path, default=ROOT/"data/bestPosition.json")
    args = parser.parse_args()
    if args.geometry_file.is_file():
        cached = json.loads(args.geometry_file.read_text())
        angles = tuple(cached[key] for key in ("theta_src", "phi_src", "theta_rot", "phi_rot"))
        geometry_origin = str(args.geometry_file.resolve())
    else:
        angles = (0.6074123620484425, 0.7914472879881271,
                  0.3594716358447116, 0.7941001629975755)
        geometry_origin = "embedded 2026-09-08 geometry snapshot (cache absent)"
    paper = SourceConfig(L=4000)
    cases = [one_case("paper_baseline_cached_angles", paper, angles),
             one_case("live_defaults_cached_angles", SourceConfig(), angles),
             one_case("paper_baseline_source_plus_y_axis_plus_z", paper,
                      (np.pi/2, np.pi/2, 0, 0))]
    # Isolate propagation/near-zone phase behavior, keeping source size fixed.
    Q = rotor_quadrupole_phasor(*angles[2:], config=paper)
    distances = []
    for R in [6000, 60000, 600000, 6000000]:
        r = michelson_response(R*spherical_unit_vector(*angles[:2]), Q,
                              paper.gw_angular_frequency, arm_length=paper.L,
                              G=paper.G, c=paper.c)
        distances.append({"R_m": R, "kR": R*paper.gw_angular_frequency/paper.c,
                          "full": phasor_record(r["total"]),
                          "newtonian": phasor_record(r["newtonian_instantaneous"]),
                          "dual_gauge_relative_error": abs(r["total"]-r["curvature_total"])/abs(r["total"])})
    reference_budget = gwinc.load_budget(str(ROOT/"configs/aLIGO.yaml"),
                                        freq=np.array([paper.gw_frequency_hz]))
    reference_asd = float(np.sqrt(reference_budget.run().psd[0]))
    result = {
        "model": "leading conserved mass quadrupole; exact kr; free point masses; equal-arm single-roundtrip Michelson",
        "phasor_convention": "X(t)=Re[X exp(-i Omega t)]; peak, not RMS",
        "snr_status": "conditional proxy using ideal strain-referred quantum PSD and common DARM calibration; no full cavity/control/source-noise model",
        "year_seconds": YEAR_SECONDS,
        "geometry_origin": geometry_origin,
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT/"configs/source.yaml", ROOT/"configs/detector.yaml"]},
        "cases": cases, "fixed_arm_distance_sweep": distances,
        "separate_aLIGO_noise_reference": {
            "status": "different detector, for context only; not used in the main SNR table",
            "gwinc_version": gwinc.__version__,
            "config": "configs/aLIGO.yaml", "frequency_Hz": paper.gw_frequency_hz,
            "length_m": reference_budget.ifo.Infrastructure.Length,
            "test_mass_kg": reference_budget.ifo.Suspension.Stage[0].Mass,
            "total_asd_per_sqrt_Hz": reference_asd,
            "single_source_snr_one_year_for_same_signal": {
                row["name"]: row["responses"]["total"]["abs"]*np.sqrt(YEAR_SECONDS)/reference_asd
                for row in cases},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    for row in cases:
        print(row["name"], "|H_full|=", row["responses"]["total"]["abs"],
              "ideal SNR/year=", row["ideal_quantum_noise_proxy_snr_one_year"]["full"],
              "gauge check=", row["dual_gauge_relative_error"])
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
