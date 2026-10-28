"""Reproducible production-model benchmarks and small pipeline exercises.

Fixed benchmarks make historical comparisons independent of YAML edits; separate
checks exercise the active Qdot projection and noise budget. All geometry-cache
writes stay in the requested output folder. This module orchestrates public APIs;
it contains no alternative field equations. Historical finite-hole Newtonian
checks remain in docs/near-field-newtonian-check.json.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np

from .artifacts import model_metadata
from .config import SourceConfig, DetectorConfig, NoiseConfig, SamplingConfig
from .detector_response import (
    curvature_response, curvature_response_components, reference_michelson_response,
)
from .finite_distance import rotor_quadrupole_phasor
from .geometry import spherical_unit_vector
from .optimization import solve_best_geometry
from .quadrature import project_quadratures, project_sine
from .noise import get_noise_psd, get_noise_budget_psd
from .thermal_gas import calculate_thermal_gas
from .signal import compute_phasor_sum, synthesize_signal
from .snr import calculate_snr_from_arrays, calculate_snr_from_phasor
from .source_array.generation import build_array_context, build_chunk
from .source_array.io import write_source_array_npz_file
from .spectrum import calculate_spectrum, save_spectrum_npz

PAPER_ANGLES = (
    0.6074123620484425,
    0.7914472879881271,
    0.3594716358447116,
    0.7941001629975755,
)
ARM_EXTENSION_ANGLES = (np.pi / 2, np.pi / 2, 0.0, 0.0)


def benchmark_config():
    """Freeze the report's SI source parameters; never inherit live defaults."""
    return SourceConfig(
        num=2,
        H=2.0,
        D=5.0,
        d=1.0,
        s=1.5,
        R=6000.0,
        rho=1750.0,
        G=6.674e-11,
        c=2.998e8,
        omega=600 * np.pi,
        L=4000.0,
    )


def phasor_record(value):
    """JSON-safe representation of a complex peak strain phasor."""
    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "amplitude": float(abs(value)),
        "phase_rad": float(np.angle(value)),
    }


def benchmark_case(angles, config):
    """Compare production curvature integration with the harmonic observable."""
    source = config.R * spherical_unit_vector(*angles[:2])
    Q = rotor_quadrupole_phasor(*angles[2:], config=config)
    parameters = dict(arm_length=config.L, G=config.G, c=config.c)
    production = curvature_response(
        source,
        Q,
        config.gw_angular_frequency,
        source_radius=np.hypot(config.D / 2, config.H / 2),
        **parameters,
    )
    independent = reference_michelson_response(
        source, Q, config.gw_angular_frequency, quadrature_order=96, **parameters
    )
    error = abs(production.phasor - independent["total"]) / abs(production.phasor)
    if error > 1e-9:
        raise AssertionError(f"Harmonic/curvature discrepancy: {error}")
    components = curvature_response_components(
        source, Q, config.gw_angular_frequency,
        source_radius=np.hypot(config.D / 2, config.H / 2), **parameters,
    )
    recombination = abs(sum(item.phasor for item in components.values()) - production.phasor)
    if recombination > 1e-10 * production.integration_scale:
        raise AssertionError("Derivative components do not recombine into the full response")
    return {
        "angles_rad": list(angles),
        "response": phasor_record(production.phasor),
        "harmonic_response": phasor_record(independent["total"]),
        "dual_response_relative_error": error,
        "quadrature_order": production.order,
        "refinement_absolute_error": production.absolute_error,
        "integration_scale": production.integration_scale,
        "components": {name: phasor_record(item.phasor) for name, item in components.items()},
        "component_recombination_absolute_error": float(recombination),
    }


def qdot_projection_case(config, angles):
    """Check phase recovery on a shifted, non-whole-period reception-time grid.

    Compare the measured sine coefficient to the full complex response projected
    analytically onto the Qdot phase. It need not equal the Qdot-only amplitude.
    """
    case = benchmark_case(angles, config)
    delay = np.exp(0.4j)  # mechanical rotor delay delta=0.2 rad
    total = complex(case["response"]["real"], case["response"]["imag"]) * delay
    qdot = case["components"]["Qdot"]
    qdot = complex(qdot["real"], qdot["imag"]) * delay
    phase = float(np.angle(qdot))
    times = 0.12345 + np.arange(1300) / 120000.0
    values = synthesize_signal(total, times, config)
    fit = project_quadratures(times, values, config.gw_angular_frequency)
    projection = project_sine(
        times, values, config.gw_angular_frequency, phasor_phase_rad=phase
    )
    expected = float(np.real(total * np.exp(-1j * phase)))
    scale = abs(total)
    recovery_error = abs(fit.phasor - total) / scale
    projection_error = abs(projection.coefficient - expected) / scale
    if max(recovery_error, projection_error) > 1e-10:
        raise AssertionError("Reception-time quadrature or Qdot-phase recovery failed")
    return {
        "source_config": asdict(config),
        "angles_rad": list(map(float, angles)),
        "total_phasor": phasor_record(total),
        "qdot_phasor": phasor_record(qdot),
        "qdot_phase_rad": phase,
        "signed_sine_coefficient": projection.coefficient,
        "analytic_full_signal_projection": expected,
        "phasor_recovery_relative_error": float(recovery_error),
        "projection_error_over_total_amplitude": float(projection_error),
        "fit_method": projection.fit.method,
        "interpretation": "Full-signal Qdot-phase projection; other derivative terms can overlap.",
    }


def active_noise_check(config):
    """Verify the shared total PSD includes quantum, thermal and gas once each."""
    detector = DetectorConfig().with_source(config)
    frequencies = np.array([config.gw_frequency_hz])
    quantum = get_noise_psd(frequencies, detector_config=detector)
    thermal_gas = calculate_thermal_gas(frequency_hz=frequencies, runtime_detector=detector)
    index = int(np.flatnonzero(thermal_gas.frequency_hz == frequencies[0])[0])
    thermal = float(thermal_gas.spectra_psd["ThermalTotal"][index])
    gas = float(thermal_gas.spectra_psd["ResidualGas"][index])
    total = get_noise_budget_psd(frequencies, detector_config=detector)
    expected = float(quantum[0]) + thermal + gas
    if not np.isfinite(expected) or expected <= 0 or not np.isclose(total[0], expected, rtol=1e-13, atol=0):
        raise AssertionError("Invalid or inconsistent quantum + thermal + gas PSD")
    return {
        "detector_config": asdict(detector),
        "frequency_hz": float(frequencies[0]),
        "components_psd_per_hz": {"quantum": float(quantum[0]), "thermal": thermal, "residual_gas": gas},
        "total_psd_per_hz": float(total[0]),
    }


def run_validation(output_dir):
    """Write fixed benchmarks and timed 1/10/100-source artifacts to output_dir.

    Noise settings are recorded explicitly. These small timings are observations,
    not a performance guarantee for a ten-million-source calculation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = benchmark_config()
    # A local cache prevents validation from changing the user's data/ geometry.
    geometry = solve_best_geometry(path=output_dir / "bestPosition.txt", config=config)
    sampling = SamplingConfig(duration_s=0.01, sample_rate_hz=120000.0)
    detector = DetectorConfig().with_source(config)
    noise = NoiseConfig()
    payload = {
        **model_metadata(config),
        "sampling": asdict(sampling),
        "detector": asdict(detector),
        "noise": asdict(noise),
        "interpretation": "ideal free-mass response; conditional strain-noise calibration",
        "benchmarks": [
            benchmark_case(a, config) for a in (PAPER_ANGLES, ARM_EXTENSION_ANGLES)
        ],
        "qdot_projections": [
            qdot_projection_case(config, PAPER_ANGLES),
            qdot_projection_case(SourceConfig(), PAPER_ANGLES),
        ],
        "active_noise_budget": active_noise_check(SourceConfig()),
        "small_arrays": [],
    }
    for count in (1, 10, 100):
        start = perf_counter()
        context = build_array_context(
            num_sources=count,
            config=config,
            reference_geometry=geometry,
            chunk_center_approximation=True,
            approximation_chunk_size=10,
            optimize_each_source=True,
        )
        rows = build_chunk(context, 0, count)
        H = compute_phasor_sum(rows, config=config, chunk_size=7)
        elapsed = perf_counter() - start
        spectrum = calculate_spectrum(
            synthesize_signal(H, sampling.time_axis(), config), sampling=sampling
        )
        mono = calculate_snr_from_phasor(
            H, config.gw_frequency_hz, detector_config=detector, noise_config=noise
        )
        fft = calculate_snr_from_arrays(
            spectrum.magnitude,
            spectrum.freqs,
            detector_config=detector,
            noise_config=noise,
        )
        if not np.isclose(mono, fft, rtol=1e-10):
            raise AssertionError("FFT and phasor SNR disagree")
        combined_mono = calculate_snr_from_phasor(
            H, config.gw_frequency_hz, detector_config=detector,
            noise_config=noise, noise_budget="quantum-thermal-gas",
        )
        combined_fft = calculate_snr_from_arrays(
            spectrum.magnitude, spectrum.freqs, detector_config=detector,
            noise_config=noise, noise_budget="quantum-thermal-gas",
        )
        if not np.isclose(combined_mono, combined_fft, rtol=1e-10):
            raise AssertionError("Combined-budget FFT and phasor SNR disagree")
        write_source_array_npz_file(
            output_dir / f"array_{count}.npz", rows, context.metadata()
        )
        save_spectrum_npz(spectrum, output_dir / f"spectrum_{count}.npz", config=config)
        payload["small_arrays"].append(
            {
                "num_sources": count,
                "response": phasor_record(H),
                "generation_and_sum_seconds": elapsed,
                "snr_phasor": mono,
                "snr_fft": fft,
                "snr_noise_budget": "quantum",
                "snr_combined_phasor": combined_mono,
                "snr_combined_fft": combined_fft,
                "generation_strategy": "chunk_anchor",
                "approximation_chunk_size": 10,
            }
        )
    destination = output_dir / "validation.json"
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return destination
