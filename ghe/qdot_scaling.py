"""Arm-length optimization of one rotor's explicit Qdot contribution.

The statistic is conditional on exact subtraction of the other derivative
terms. Noise is analytic detuned quantum + GWINC thermal + GWINC residual gas,
added as one-sided PSDs exactly once. It remains an ideal free-mass strain proxy.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .artifacts import file_digest, model_metadata
from .config import DetectorConfig, NoiseConfig, SourceConfig, _read_yaml
from .detector_response import curvature_response_components, validate_clearance
from .finite_distance import rotor_quadrupole_phasor, validate_rotor
from .geometry import spherical_unit_vector
from .noise import get_noise_psd
from .paths import DETECTOR_CONFIG_FILE, SOURCE_CONFIG_FILE, YEAR_SECONDS
from .thermal_gas import _json_config, calculate_thermal_gas, resolve_config_path


class QdotArmLengthStudy:
    """Reusable evaluator with fixed R/L, mirror curvatures and rotor properties.

    With angles=None, maximize |H_Qdot| at each L using deterministic multistart
    optimization. Noise is geometry-independent, so this also maximizes the
    selected Qdot SNR. Explicit angles instead select a fixed-geometry study.
    All configuration overrides are in memory; shared YAML/cache files remain
    untouched. The source frequency must match the active source YAML, which
    supplies the thermal/gas detuning and material constants.
    """

    def __init__(
        self,
        *,
        source_config=None,
        detector_config=DETECTOR_CONFIG_FILE,
        angles=None,
        observation_time_s=YEAR_SECONDS,
    ):
        self.source = source_config or SourceConfig()
        validate_rotor(self.source)
        if not np.isfinite(observation_time_s) or observation_time_s <= 0:
            raise ValueError("Observation time must be finite and positive")
        self.observation_time_s = float(observation_time_s)
        self.distance_arm_ratio = self.source.R / self.source.L
        self.detector_path = resolve_config_path(detector_config)
        self.document = _read_yaml(self.detector_path)
        self.detector = DetectorConfig.from_yaml(self.detector_path)
        self.angles = None if angles is None else np.asarray(angles, dtype=float)
        if self.angles is not None and (
            self.angles.shape != (4,) or not np.all(np.isfinite(self.angles))
        ):
            raise ValueError("Require four finite source/rotor angles in radians")
        source_document = _read_yaml(SOURCE_CONFIG_FILE)
        frequency = float(source_document["Source"]["Rotor"]["AngularVelocity"]) / np.pi
        if not np.isclose(self.source.gw_frequency_hz, frequency, rtol=1e-12, atol=0):
            raise ValueError(
                "Source frequency must match the active source YAML for consistent thermal/gas tuning"
            )
        self.cache = {}
        self.warm_angles = None

    def cavity_properties(self, length):
        """Match GWINC arm_cavity geometry and check the fixed optical loss budget.

        A beam's circular-aperture clipping fraction is exp(-2 a²/w²). Allowing
        the entire configured mirror loss to cover clipping is a necessary,
        generous consistency check; it leaves no allowance for other losses at
        that limit. Reject the exact confocal degeneracy as GWINC does.
        """
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Arm length must be finite and positive")
        curvature = self.document["Optics"]["Curvature"]
        radii = np.array([float(curvature["ITM"]), float(curvature["ETM"])])
        g = 1 - length / radii
        product = float(np.prod(g))
        if not 0 < product < 1:
            raise ValueError("Cavity is unstable or degenerate: require 0 < g1*g2 < 1")
        # Same spot-radius formula as gwinc.ifo.noises.arm_cavity.
        gcav = np.sqrt(product * (1 - product))
        wavelength = float(self.document["Laser"]["Wavelength"])
        spots = np.sqrt(length * wavelength / np.pi * np.abs(g[::-1]) / gcav)
        aperture = float(self.document["Materials"]["MassRadius"])
        clipping = 1e6 * np.exp(-2 * (aperture / spots) ** 2)
        loss_ppm = 1e6 * float(self.document["Optics"]["Loss"])
        if np.any(clipping > loss_ppm):
            raise ValueError(
                f"Beam clipping exceeds the fixed {loss_ppm:g} ppm mirror-loss budget"
            )
        return {
            "g1": float(g[0]),
            "g2": float(g[1]),
            "g1_g2": product,
            "itm_beam_radius_m": float(spots[0]),
            "etm_beam_radius_m": float(spots[1]),
            "itm_clipping_ppm": float(clipping[0]),
            "etm_clipping_ppm": float(clipping[1]),
            "mirror_loss_budget_ppm": loss_ppm,
        }

    @staticmethod
    def _response(angles, source, component="Qdot"):
        moment = rotor_quadrupole_phasor(*angles[2:], config=source)
        return curvature_response_components(
            source.R * spherical_unit_vector(*angles[:2]),
            moment,
            source.gw_angular_frequency,
            arm_length=source.L,
            G=source.G,
            c=source.c,
            source_radius=np.hypot(source.D / 2, source.H / 2),
        )[component]

    def _best_geometry(self, source):
        if self.angles is not None:
            return self.angles, self._response(self.angles, source), {"mode": "fixed"}
        starts = [
            (np.pi / 2, 0, 0, 0),
            (np.pi / 2, np.pi / 2, 0, 0),
            (0.6, 0.8, 0.4, 0.8),
            (2.5, 4.1, 1.2, 2.3),
        ]
        if self.warm_angles is not None:
            starts.append(tuple(self.warm_angles))
        evaluations = {}

        def response(angles):
            key = tuple(map(float, angles))
            if key not in evaluations:
                evaluations[key] = self._response(angles, source)
            return evaluations[key]

        seeds = []
        for start in starts:
            try:
                seeds.append((np.array(start), response(start)))
            except (ValueError, ArithmeticError):
                continue
        if not seeds:
            raise ValueError("No feasible source geometry at this arm length")
        scale = max(abs(result.phasor) for _, result in seeds)
        if scale == 0:
            raise ArithmeticError("All Qdot optimization seeds have zero response")

        def objective(angles):
            try:
                return -abs(response(angles).phasor) / scale
            except (ValueError, ArithmeticError):
                return 1e100

        candidates = list(seeds)
        successes = 0
        bounds = [(0, np.pi), (0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi)]
        for start, _ in seeds:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                options={"ftol": 1e-10, "eps": 1e-6, "maxiter": 150},
            )
            if result.success and np.isfinite(result.fun) and result.fun < 0:
                successes += 1
                candidates.append((result.x, response(result.x)))
        if successes == 0:
            raise ArithmeticError(
                "Qdot geometry optimization did not converge from any seed"
            )
        angles, result = max(candidates, key=lambda pair: abs(pair[1].phasor))
        self.warm_angles = angles.copy()
        return (
            angles,
            result,
            {
                "mode": "optimize_qdot",
                "successful_starts": successes,
                "num_starts": len(seeds),
                "response_evaluations": len(evaluations),
            },
        )

    def evaluate(self, arm_length_m):
        """Calculate one valid length, including all three independent noise PSDs."""
        length = float(arm_length_m)
        if length in self.cache:
            return self.cache[length]
        cavity = self.cavity_properties(length)
        source = replace(self.source, L=length, R=length * self.distance_arm_ratio)
        validate_rotor(source)
        detector = self.detector.with_source(source)
        noise = NoiseConfig(model="detuned_signal_recycling")
        frequency = source.gw_frequency_hz
        quantum = float(
            get_noise_psd(
                np.array([frequency]), detector_config=detector, noise_config=noise
            )[0]
        )
        thermal_gas = calculate_thermal_gas(
            detector_config=self.detector_path,
            arm_length_m=length,
            frequency_hz=[frequency],
        )
        index = int(np.flatnonzero(thermal_gas.frequency_hz == frequency)[0])
        thermal = float(thermal_gas.spectra_psd["ThermalTotal"][index])
        gas = float(thermal_gas.spectra_psd["ResidualGas"][index])
        total = quantum + thermal + gas
        if (
            not np.all(np.isfinite([quantum, thermal, gas, total]))
            or min(quantum, thermal, gas) < 0
            or total <= 0
        ):
            raise ArithmeticError("Noise calculation returned invalid PSDs")
        angles, response, optimization = self._best_geometry(source)
        # Record closest source-to-arm distance to assess the compact-source limit.
        position = source.R * spherical_unit_vector(*angles[:2])
        validate_clearance(position, length, np.hypot(source.D / 2, source.H / 2))
        minimum_distance = min(
            np.linalg.norm(position - np.clip(position @ axis, 0, length) * axis)
            for axis in np.eye(3)[:2]
        )
        q_peak = abs(self._response(angles, source, component="Q").phasor)
        row = {
            "arm_length_m": length,
            "source_distance_m": source.R,
            "frequency_hz": frequency,
            "qdot_phasor_real": float(response.phasor.real),
            "qdot_phasor_imag": float(response.phasor.imag),
            "qdot_peak_strain": float(abs(response.phasor)),
            "q_peak_strain": float(q_peak),
            "qdot_over_q_amplitude": float(abs(response.phasor) / q_peak)
            if q_peak
            else None,
            "quantum_psd": quantum,
            "thermal_psd": thermal,
            "gas_psd": gas,
            "total_psd": total,
            "total_asd": float(np.sqrt(total)),
            "rho_qdot": float(
                abs(response.phasor) * np.sqrt(self.observation_time_s / total)
            ),
            "angles_rad": list(map(float, angles)),
            "optimization": optimization,
            "integration_order": response.order,
            "integration_absolute_error": response.absolute_error,
            "cavity": cavity,
            "minimum_source_arm_distance_m": float(minimum_distance),
            "source_radius_over_minimum_distance": float(
                np.hypot(source.D / 2, source.H / 2) / minimum_distance
            ),
            "analytic_detuning_rad": detector.phi_SR,
            "gwinc_version": thermal_gas.metadata["gwinc_version"],
        }
        self.cache[length] = row
        return row


def sweep_qdot_arm_lengths(
    lengths_m,
    *,
    source_config=None,
    detector_config=DETECTOR_CONFIG_FILE,
    angles=None,
    observation_time_s=YEAR_SECONDS,
    refine=True,
    progress=None,
):
    """Sweep L and refine each interior sampled peak in log L.

    A lower-bound maximum is reported explicitly and never presented as an
    unconstrained optimal arm length. Invalid optical geometries are recorded,
    while unexpected numerical/optimization failures abort rather than vanish.
    """
    lengths = np.asarray(lengths_m, dtype=float)
    if (
        lengths.ndim != 1
        or lengths.size < 2
        or not np.all(np.isfinite(lengths) & (lengths > 0))
    ):
        raise ValueError("Require at least two finite positive arm lengths")
    lengths = np.unique(lengths)
    if lengths.size < 2:
        raise ValueError("Require at least two distinct arm lengths")
    study = QdotArmLengthStudy(
        source_config=source_config,
        detector_config=detector_config,
        angles=angles,
        observation_time_s=observation_time_s,
    )
    grid, rejected = [], []
    for length in lengths:
        try:
            # Only expected optical infeasibility is skipped; computation errors propagate.
            study.cavity_properties(float(length))
        except ValueError as exc:
            rejected.append({"arm_length_m": float(length), "reason": str(exc)})
            continue
        row = study.evaluate(length)
        grid.append(row)
        if progress is not None:
            progress(row)
    if not grid:
        raise ValueError("No optically feasible arm lengths in the requested sweep")
    refinements = []
    if refine:
        for previous, center, following in zip(grid, grid[1:], grid[2:]):
            if center["rho_qdot"] > max(previous["rho_qdot"], following["rho_qdot"]):
                scale = center["rho_qdot"]

                def objective(log_length):
                    try:
                        return -study.evaluate(np.exp(log_length))["rho_qdot"] / scale
                    except ValueError:
                        return 1e100

                result = minimize_scalar(
                    objective,
                    method="bounded",
                    bounds=np.log(
                        [previous["arm_length_m"], following["arm_length_m"]]
                    ),
                    options={"xatol": 1e-6},
                )
                if not result.success:
                    raise ArithmeticError("Arm-length peak refinement did not converge")
                refinements.append(study.evaluate(np.exp(result.x)))
    best = max([*grid, *refinements], key=lambda row: row["rho_qdot"])
    location = (
        "lower_bound"
        if best["arm_length_m"] == lengths[0]
        else "upper_bound"
        if best["arm_length_m"] == lengths[-1]
        else "interior"
    )
    if location == "interior" and best in (grid[0], grid[-1]):
        location = "feasibility_boundary"
    return {
        "metadata": model_metadata(study.source),
        "detector_config_path": str(study.detector_path),
        "fixed_detector_configuration": _json_config(dict(study.document)),
        "input_sha256": {
            str(path): file_digest(path)
            for path in (study.detector_path, SOURCE_CONFIG_FILE)
        },
        "observation_time_s": study.observation_time_s,
        "distance_arm_ratio": study.distance_arm_ratio,
        "mirror_curvatures_m": dict(study.document["Optics"]["Curvature"]),
        "geometry_mode": "optimize_qdot" if angles is None else "fixed",
        "search_bounds_m": [float(lengths[0]), float(lengths[-1])],
        "best_location": location,
        "best": best,
        "grid": grid,
        "refined_peaks": refinements,
        "rejected_lengths": rejected,
        "assumptions": [
            "One rotor; peak exp(-i Omega t) phasors and known phase.",
            "R/L fixed; mirror curvatures, rotor, powers, masses, squeezing, temperatures and pressures fixed.",
            "Retune the analytic SR detuning at the source frequency for each arm length.",
            "Analytic detuned quantum + GWINC thermal + GWINC residual gas PSDs summed once.",
            "Exact subtraction of all other curvature terms; ideal free-mass strain-noise proxy.",
            "Small source-arm clearance increases sensitivity to omitted source multipoles and mechanical coupling; no subtraction-error allowance is included.",
            "Multistart angular optimization finds the best candidate, not a certified global maximum.",
            "Exclude clipping above the configured mirror-loss budget and GWINC cavity degeneracies.",
        ],
    }
