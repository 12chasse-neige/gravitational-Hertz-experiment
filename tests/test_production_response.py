"""Scientific acceptance tests for the production observable and its conventions.

Assertions use zero absolute tolerance for tiny physical strains: numpy's
ordinary default absolute tolerance would incorrectly let a zero signal pass.
"""

from dataclasses import replace
import numpy as np
import pytest

from ghe.config import SourceConfig, NoiseConfig, DetectorConfig, SamplingConfig
from ghe.detector_response import IntegrationSettings, curvature_response
from ghe.finite_distance import quadrupole_field, rotor_quadrupole_phasor
from ghe.geometry import spherical_unit_vector
from ghe.metric import calculate_response_phasor, calculate_metric_response
from ghe.optimization import optimize_best_geometry
from ghe.signal import compute_phasor_sum, calculate_source_array_signal
from ghe.source_array.generation import build_array_context, build_chunk
from ghe.source_array.schema import SOURCE_ARRAY_DTYPE
from ghe.source_array.phase import get_signal_amplitude_and_phase
from ghe.spectrum import calculate_spectrum
from ghe.snr import calculate_snr_from_arrays, calculate_snr_from_phasor
from ghe.validation import (
    benchmark_config,
    PAPER_ANGLES,
    ARM_EXTENSION_ANGLES,
    benchmark_case,
)


@pytest.mark.parametrize(
    "angles,expected", [(PAPER_ANGLES, 1.936e-32), (ARM_EXTENSION_ANGLES, 2.026e-30)]
)
def test_fixed_benchmarks_and_dual_response(angles, expected):
    """Numbers are frozen from the pre-integration report, not the new kernel."""
    result = benchmark_case(angles, benchmark_config())
    assert result["response"]["amplitude"] == pytest.approx(expected, rel=4e-4, abs=0)
    assert result["dual_response_relative_error"] < 1e-11


def test_batch_field_matches_scalar_at_multiple_distances():
    cfg = benchmark_config()
    Q = rotor_quadrupole_phasor(0.3, 0.7, config=cfg)
    points = np.array(
        [[6000.0, 2000.0, 1000.0], [300.0, 500.0, 600.0], [1e6, 2e6, -3e6]]
    )
    batch = quadrupole_field(points, Q, cfg.gw_angular_frequency, G=cfg.G, c=cfg.c)
    for i, point in enumerate(points):
        scalar = quadrupole_field(point, Q, cfg.gw_angular_frequency, G=cfg.G, c=cfg.c)
        for key in batch:
            np.testing.assert_allclose(batch[key][i], scalar[key], rtol=2e-14, atol=0)


def test_explicit_distance_and_replacement_semantics():
    cfg = SourceConfig(L=2400, R=123.0)
    assert cfg.R == 123.0
    assert replace(cfg, R=7000.0).R == 7000.0
    assert replace(cfg, L=4000.0).R == 123.0
    assert replace(cfg, L=4000.0, R=None).R == 6000.0
    overridden = calculate_response_phasor(*PAPER_ANGLES, config=cfg, R=7000.0)
    explicit = calculate_response_phasor(*PAPER_ANGLES, config=replace(cfg, R=7000.0))
    assert overridden == explicit


@pytest.mark.parametrize(
    "changes", [{"num": 3}, {"omega": 0}, {"rho": -1}, {"d": 4}, {"s": float("nan")}]
)
def test_invalid_rotor_is_rejected(changes):
    with pytest.raises(ValueError):
        calculate_response_phasor(
            *PAPER_ANGLES, config=replace(benchmark_config(), **changes)
        )


def test_clearance_checks_whole_arm_not_only_nodes():
    # The rotor lies midway along +x. It would evade an endpoint-only check.
    with pytest.raises(ValueError, match="light path"):
        calculate_response_phasor(
            np.pi / 2, 0, 0, 0, config=replace(benchmark_config(), R=2000)
        )


def test_integration_refines_and_fails_explicitly():
    cfg = benchmark_config()
    Q = rotor_quadrupole_phasor(0, 0, config=cfg)
    kwargs = dict(arm_length=cfg.L, G=cfg.G, c=cfg.c)
    source = np.array([10.0, 2000.0, 20.0])
    with pytest.raises(ArithmeticError, match="converge"):
        curvature_response(
            source,
            Q,
            cfg.gw_angular_frequency,
            **kwargs,
            settings=IntegrationSettings(
                initial_order=4, max_order=8, relative_tolerance=1e-12
            ),
        )
    # Ordinary benchmark geometry converges under two different starting rules.
    source = cfg.R * spherical_unit_vector(*PAPER_ANGLES[:2])
    standard = curvature_response(source, Q, cfg.gw_angular_frequency, **kwargs)
    refined = curvature_response(
        source,
        Q,
        cfg.gw_angular_frequency,
        **kwargs,
        settings=IntegrationSettings(
            initial_order=96, max_order=384, relative_tolerance=1e-11
        ),
    )
    np.testing.assert_allclose(standard.phasor, refined.phasor, rtol=1e-10, atol=0)


def make_row(cfg):
    rows = np.zeros(1, dtype=SOURCE_ARRAY_DTYPE)
    for name, angle in zip(
        ("theta_src", "phi_src", "theta_rot", "phi_rot"), PAPER_ANGLES
    ):
        rows[name] = angle
    rows["distance_to_detector_m"] = cfg.R
    return rows


def test_phase_delay_matches_independent_time_translation():
    cfg = benchmark_config()
    rows = make_row(cfg)
    delta = 0.731
    rows["rotor_phase_offset_rad"] = delta
    rows["gw_phase_offset_rad"] = 2 * delta
    times = np.array([0, 0.00012, 0.00037, 0.0011])
    # Compare against directly evaluating the public response at t-delta/omega.
    expected = [
        calculate_metric_response(t - delta / cfg.omega, *PAPER_ANGLES, config=cfg)
        for t in times
    ]
    actual = calculate_source_array_signal(times, rows, config=cfg)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)
    amplitude, phase = get_signal_amplitude_and_phase(*PAPER_ANGLES, cfg.R, config=cfg)
    direct = [calculate_metric_response(t, *PAPER_ANGLES, config=cfg) for t in times]
    np.testing.assert_allclose(
        amplitude * np.cos(cfg.gw_angular_frequency * times + phase),
        direct,
        rtol=2e-13,
        atol=0,
    )


def test_coherence_cancellation_and_chunk_independence():
    cfg = benchmark_config()
    one = make_row(cfg)
    rows = np.repeat(one, 10)
    H = compute_phasor_sum(one, config=cfg)
    total = compute_phasor_sum(rows, config=cfg, chunk_size=3)
    np.testing.assert_allclose(total, 10 * H, rtol=1e-14, atol=0)
    assert total == compute_phasor_sum(rows, config=cfg, chunk_size=7)
    rows["rotor_phase_offset_rad"][5:] = np.pi / 2
    rows["gw_phase_offset_rad"][5:] = np.pi
    assert abs(compute_phasor_sum(rows, config=cfg)) < abs(H) * 1e-14


@pytest.mark.parametrize("strategy", ["exact", "rigid", "chunk_anchor"])
def test_all_strategies_align_full_response_and_use_actual_config(strategy):
    cfg = replace(benchmark_config(), R=7000.0, omega=500 * np.pi)
    geometry = optimize_best_geometry(config=cfg)
    context = build_array_context(
        num_sources=4,
        config=cfg,
        reference_geometry=geometry,
        optimize_each_source=strategy == "exact",
        chunk_center_approximation=strategy == "chunk_anchor",
        approximation_chunk_size=2,
    )
    rows = build_chunk(context, 0, 4)
    raw_amplitudes = [
        abs(
            calculate_response_phasor(
                *(
                    float(row[name])
                    for name in ("theta_src", "phi_src", "theta_rot", "phi_rot")
                ),
                config=cfg,
                R=float(row["distance_to_detector_m"]),
            )
        )
        for row in rows
    ]
    np.testing.assert_allclose(
        abs(compute_phasor_sum(rows, config=cfg)),
        sum(raw_amplitudes),
        rtol=2e-13,
        atol=0,
    )
    split = np.concatenate([build_chunk(context, 0, 1), build_chunk(context, 1, 4)])
    for name in rows.dtype.names:
        np.testing.assert_allclose(rows[name], split[name], rtol=1e-12, atol=1e-12)


def test_optimizer_never_worse_than_documented_seed():
    cfg = benchmark_config()
    best = optimize_best_geometry(config=cfg)
    assert best.signal_amplitude >= abs(
        calculate_response_phasor(*ARM_EXTENSION_ANGLES, config=cfg)
    ) * (1 - 1e-12)


def test_fft_and_phasor_snr_agree_and_scale():
    cfg = benchmark_config()
    detector = DetectorConfig().with_source(cfg)
    noise = NoiseConfig()
    sampling = SamplingConfig(duration_s=0.01, sample_rate_hz=120000.0)
    H = calculate_response_phasor(*PAPER_ANGLES, config=cfg)
    signal = np.real(H * np.exp(-1j * cfg.gw_angular_frequency * sampling.time_axis()))
    spectrum = calculate_spectrum(signal, sampling=sampling)
    fft = calculate_snr_from_arrays(
        spectrum.magnitude, spectrum.freqs, detector_config=detector, noise_config=noise
    )
    mono = calculate_snr_from_phasor(
        H, cfg.gw_frequency_hz, detector_config=detector, noise_config=noise
    )
    np.testing.assert_allclose(fft, mono, rtol=1e-12, atol=0)
    ten = calculate_snr_from_phasor(
        10 * H, cfg.gw_frequency_hz, detector_config=detector, noise_config=noise
    )
    np.testing.assert_allclose([ten / mono, ten**2 / mono**2], [10, 100], rtol=1e-13)


def test_independent_finite_holes_recover_newtonian_limit():
    from tests.support.verify_newtonian import (
        direct_newtonian_response,
        quadrupole_response,
    )

    cfg = benchmark_config()
    source = cfg.R * spherical_unit_vector(*PAPER_ANGLES[:2])
    independent, *_ = quadrupole_response(cfg, source, np.array(PAPER_ANGLES[2:]))
    direct, _ = direct_newtonian_response(
        cfg,
        source,
        np.array(PAPER_ANGLES[2:]),
        finite_cylinders=True,
        phase_samples=16,
        quadrature=(2, 8, 2),
    )
    # Finite holes contain higher multipoles: equality is approximate, governed
    # by (source size / distance)^2, not the optical quadrature tolerance.
    np.testing.assert_allclose(direct, independent, rtol=2e-6, atol=0)
    slow_propagation = replace(cfg, c=cfg.c * 1e5)
    complete = calculate_response_phasor(*PAPER_ANGLES, config=slow_propagation)
    np.testing.assert_allclose(complete, independent, rtol=1e-6, atol=0)


def test_explicit_array_center_uses_its_own_reference_rotor():
    """Rigid transport must originate at the chosen center, not a cached sky point."""
    cfg = benchmark_config()
    from ghe.optimization import BestGeometry

    original = BestGeometry(
        *PAPER_ANGLES, abs(calculate_response_phasor(*PAPER_ANGLES, config=cfg))
    )
    context = build_array_context(
        num_sources=1,
        theta_array=1.2,
        phi_array=2.1,
        config=cfg,
        reference_geometry=original,
        optimize_each_source=False,
    )
    row = build_chunk(context, 0, 1)[0]
    np.testing.assert_allclose(
        spherical_unit_vector(row["theta_rot"], row["phi_rot"]),
        spherical_unit_vector(
            context.reference_rotor_theta, context.reference_rotor_phi
        ),
        atol=1e-12,
    )


def test_cylinder_quadrature_volume():
    """Independent reference mass integration must preserve the cylinder volume."""
    from tests.support.newtonian import _cylinder_quadrature

    points, weights = _cylinder_quadrature(0.5, 2.0, 1.5, 0.0, 3, 8, 3)
    np.testing.assert_allclose(weights.sum(), np.pi * 0.5**2 * 2, rtol=1e-14)
    np.testing.assert_allclose(
        np.average(points, weights=weights, axis=0), [1.5, 0.0, 0.0], atol=1e-14
    )
