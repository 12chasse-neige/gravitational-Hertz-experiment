"""Reproducible production-model benchmarks and small pipeline exercises.

Fixed physical inputs make historical numerical comparisons independent of live
YAML edits and mutable optimized-geometry caches. This module orchestrates public
APIs; it contains no alternative field equations. Tests additionally carry an
independent finite-hole Newtonian reference.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np

from .artifacts import model_metadata
from .config import SourceConfig, DetectorConfig, NoiseConfig, SamplingConfig
from .detector_response import curvature_response, reference_michelson_response
from .finite_distance import rotor_quadrupole_phasor
from .geometry import spherical_unit_vector
from .optimization import solve_best_geometry
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
    return {
        "angles_rad": list(angles),
        "response": phasor_record(production.phasor),
        "harmonic_response": phasor_record(independent["total"]),
        "dual_response_relative_error": error,
        "quadrature_order": production.order,
        "refinement_absolute_error": production.absolute_error,
        "integration_scale": production.integration_scale,
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
                "generation_strategy": "chunk_anchor",
                "approximation_chunk_size": 10,
            }
        )
    destination = output_dir / "validation.json"
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return destination
