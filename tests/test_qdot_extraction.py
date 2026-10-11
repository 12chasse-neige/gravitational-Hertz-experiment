"""Scientific checks for derivative decomposition and phase-sensitive projection."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ghe.config import SourceConfig
from ghe.detector_response import IntegrationSettings, curvature_response_components
from ghe.finite_distance import (
    TIDAL_COMPONENT_NAMES,
    quadrupole_field,
    quadrupole_tidal_components,
    rotor_quadrupole_phasor,
)
from ghe.metric import calculate_response_phasor
from ghe.quadrature import project_quadratures
from ghe.validation import ARM_EXTENSION_ANGLES, PAPER_ANGLES, benchmark_config
from scripts.extractQdot import analyze_single_source


def test_component_fields_recombine_and_are_symmetric_trace_free():
    cfg = benchmark_config()
    Q = rotor_quadrupole_phasor(0.3, 0.7, config=cfg)
    points = np.array(
        [[6000.0, 2000.0, 1000.0], [300.0, 500.0, 600.0], [1e7, 2e7, -3e7]]
    )
    components = quadrupole_tidal_components(
        points, Q, cfg.gw_angular_frequency, G=cfg.G, c=cfg.c
    )
    assert tuple(components) == TIDAL_COMPONENT_NAMES
    total = quadrupole_field(points, Q, cfg.gw_angular_frequency, G=cfg.G, c=cfg.c)[
        "tidal"
    ]
    np.testing.assert_allclose(sum(components.values()), total, rtol=3e-13, atol=0)
    for component in components.values():
        assert component.shape == (3, 3, 3)
        np.testing.assert_allclose(
            component, np.swapaxes(component, -1, -2), rtol=1e-14, atol=0
        )
        scales = np.linalg.norm(component, axis=(-2, -1))
        assert np.all(np.abs(np.trace(component, axis1=-2, axis2=-1)) <= 2e-14 * scales)
    scalar = quadrupole_tidal_components(
        points[0], Q, cfg.gw_angular_frequency, G=cfg.G, c=cfg.c
    )
    for name in components:
        np.testing.assert_allclose(
            scalar[name], components[name][0], rtol=1e-14, atol=0
        )


def test_qdot_matches_explicit_retarded_time_derivative_and_static_limit():
    cfg = benchmark_config()
    Q = rotor_quadrupole_phasor(0.8, 1.2, config=cfg)
    x = np.array([6000.0, 2000.0, -1000.0])
    r = np.linalg.norm(x)
    n = x / r
    dotQ = -1j * cfg.gw_angular_frequency * Q
    q = dotQ @ n
    S = n @ q
    expected = (
        cfg.G
        / (cfg.c * r**4)
        * np.exp(1j * cfg.gw_angular_frequency * r / cfg.c)
        * (
            -3 * dotQ
            + 15 * (np.outer(n, q) + np.outer(q, n))
            + 7.5 * np.eye(3) * S
            - 52.5 * np.outer(n, n) * S
        )
    )
    components = quadrupole_tidal_components(
        x, Q, cfg.gw_angular_frequency, G=cfg.G, c=cfg.c
    )
    np.testing.assert_allclose(components["Qdot"], expected, rtol=1e-14, atol=0)
    static = quadrupole_tidal_components(x, Q, 0, G=cfg.G, c=cfg.c)
    for name in TIDAL_COMPONENT_NAMES[1:]:
        assert np.count_nonzero(static[name]) == 0
    np.testing.assert_allclose(
        static["Q"],
        quadrupole_field(x, Q, 0, G=cfg.G, c=cfg.c)["tidal"],
        rtol=1e-14,
        atol=0,
    )


@pytest.mark.parametrize(
    "angles", [PAPER_ANGLES, ARM_EXTENSION_ANGLES, (0.7, 0.9, 1.1, 0.4)]
)
def test_detector_components_recombine_and_rotor_delay_is_applied_to_every_term(angles):
    cfg = benchmark_config()
    times = 0.0123 + np.arange(1200) / 120000
    plain, arrays = analyze_single_source(times, angles=angles, config=cfg)
    delayed, shifted = analyze_single_source(
        times, angles=angles, config=cfg, rotor_delay_rad=0.23
    )
    component_sum = sum(
        complex(record["modeled_phasor"]["real"], record["modeled_phasor"]["imag"])
        for record in plain["components"].values()
    )
    np.testing.assert_allclose(
        component_sum,
        calculate_response_phasor(*angles, config=cfg),
        rtol=2e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        arrays["qdot_model"] + arrays["remaining_model"],
        arrays["model_total"],
        rtol=0,
        atol=1e-12 * abs(component_sum),
    )
    for name in TIDAL_COMPONENT_NAMES:
        value = plain["components"][name]["modeled_phasor"]
        actual = delayed["components"][name]["modeled_phasor"]
        H = complex(value["real"], value["imag"])
        np.testing.assert_allclose(
            complex(actual["real"], actual["imag"]),
            H * np.exp(0.46j),
            rtol=1e-14,
            atol=0,
        )
        recovered = delayed["components"][name]["projection"]["phasor"]
        np.testing.assert_allclose(
            complex(recovered["real"], recovered["imag"]),
            H * np.exp(0.46j),
            rtol=2e-13,
            atol=0,
        )
    expected = np.real(
        component_sum * np.exp(0.46j) * np.exp(-1j * cfg.gw_angular_frequency * times)
    )
    np.testing.assert_allclose(
        shifted["signal"], expected, rtol=0, atol=1e-12 * abs(component_sum)
    )


def test_component_convergence_and_source_clearance_fail_explicitly():
    cfg = benchmark_config()
    Q = rotor_quadrupole_phasor(0, 0, config=cfg)
    with pytest.raises(ArithmeticError, match="did not converge"):
        curvature_response_components(
            [10, 2000, 20],
            Q,
            cfg.gw_angular_frequency,
            arm_length=cfg.L,
            G=cfg.G,
            c=cfg.c,
            settings=IntegrationSettings(
                initial_order=4, max_order=8, relative_tolerance=1e-12
            ),
        )
    with pytest.raises(ValueError, match="light path"):
        analyze_single_source(
            np.arange(1200) / 120000,
            angles=(np.pi / 2, 0, 0, 0),
            config=replace(cfg, R=2000),
        )


@pytest.mark.parametrize(
    "count,method", [(1200, "orthogonal_projection"), (1273, "least_squares")]
)
@pytest.mark.parametrize("scale", [1.0, 1e-32])
def test_projection_recovers_peak_coefficients_and_dc_at_original_time_zero(
    count, method, scale
):
    times = 0.1234 + np.arange(count) / 120000
    Omega = 1200 * np.pi
    signal = scale * (1.7 * np.cos(Omega * times) - 0.8 * np.sin(Omega * times) + 0.3)
    result = project_quadratures(times, signal, Omega)
    assert result.method == method
    np.testing.assert_allclose(
        [result.cosine, result.sine, result.offset],
        scale * np.array([1.7, -0.8, 0.3]),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(result.phasor, scale * (1.7 - 0.8j), rtol=1e-12, atol=0)
    assert result.residual_rms < 1e-12 * scale


def test_nonuniform_projection_and_zero_signal():
    times = np.arange(1273) / 120000
    times[1:-1] += np.sin(np.arange(1271)) * 1e-6
    Omega = 1200 * np.pi
    signal = 1e-32 * (np.cos(Omega * times) + 2 * np.sin(Omega * times))
    result = project_quadratures(times, signal, Omega)
    assert result.method == "least_squares"
    np.testing.assert_allclose(result.phasor, 1e-32 * (1 + 2j), rtol=1e-13, atol=0)
    zero = project_quadratures(times, np.zeros_like(times), Omega)
    assert zero.phasor == 0j and zero.offset == 0 and zero.residual_rms == 0


def test_same_frequency_contamination_is_not_identified_as_qdot():
    times = np.arange(1200) / 120000
    sine = np.sin(1200 * np.pi * times)
    # Orthogonality observes the sum, even with a perfectly known sine phase.
    qdot = 2e-32 * sine
    other = 7e-32 * sine
    fit = project_quadratures(times, qdot + other, 1200 * np.pi)
    assert fit.sine == pytest.approx(9e-32, rel=1e-13, abs=0)
    assert fit.sine != pytest.approx(2e-32, rel=1e-13, abs=0)


@pytest.mark.parametrize(
    "times,signal,Omega,match",
    [
        ([0, 1], [0, 0], 1, ">=3"),
        ([0, 1, 2], [0, 0], 1, "equal length"),
        ([[0, 1, 2]], [[0, 0, 0]], 1, "one-dimensional"),
        ([0, 1, 2], [0, np.nan, 0], 1, "finite"),
        ([0, 1, np.inf], [0, 0, 0], 1, "finite"),
        ([0, 1, 1], [0, 0, 0], 1, "increasing"),
        ([2, 1, 0], [0, 0, 0], 1, "increasing"),
        ([0, 1, 2], [0, 0, 0], 0, "positive"),
        ([0, 1, 2], [0, 0, 0], np.nan, "positive"),
        ([0, 1, 2], [0, 0, 0], np.pi, "Nyquist"),
        ([0, 1, 3, 6], [0, 0, 0, 0], 2 * np.pi, "Rank-deficient"),
        ([0, 1, 2], [0j, 1j, 0j], 1, "real"),
    ],
)
def test_invalid_projection_inputs(times, signal, Omega, match):
    with pytest.raises(ValueError, match=match):
        project_quadratures(times, signal, Omega)


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, success=True):
    completed = subprocess.run(
        [sys.executable, "scripts/extractQdot.py", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (completed.returncode == 0) == success, completed.stdout + completed.stderr
    return completed


def test_cli_generation_loading_and_artifacts(tmp_path):
    generated, loaded = tmp_path / "generated", tmp_path / "loaded"
    run_cli(
        "--angles", *PAPER_ANGLES, "--rotor-delay-rad", 0.23, "--output-dir", generated
    )
    for name in ("report.json", "signals.npz", "signals.csv", "comparison.png"):
        assert (generated / name).stat().st_size > 0
    first = json.loads((generated / "report.json").read_text())
    assert first["signal_origin"] == "generated"
    assert "cannot be separated" in first["interpretation"]["signal_projection"]
    run_cli(
        "--angles",
        *PAPER_ANGLES,
        "--rotor-delay-rad",
        0.23,
        "--signal-input",
        generated / "signals.npz",
        "--output-dir",
        loaded,
    )
    second = json.loads((loaded / "report.json").read_text())
    assert second["signal_origin"] == "loaded"
    assert second["signal_projection"] == first["signal_projection"]
    with np.load(loaded / "signals.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(
            arrays["qdot_model"] + arrays["remaining_model"],
            arrays["model_total"],
            rtol=0,
            atol=abs(
                complex(
                    first["modeled_total_phasor"]["real"],
                    first["modeled_total_phasor"]["imag"],
                )
            )
            * 1e-12,
        )
        assert "sine_quadrature" in arrays and "metadata" in arrays
    csv = np.genfromtxt(loaded / "signals.csv", delimiter=",", names=True)
    assert csv.dtype.names[:2] == ("time_s", "signal")
    assert csv.size == first["sampling"]["num_samples"]
    mismatch = run_cli(
        "--angles",
        *PAPER_ANGLES,
        "--signal-input",
        generated / "signals.npz",
        "--output-dir",
        loaded,
        success=False,
    )
    assert "geometry/rotor delay" in mismatch.stderr


def test_cli_raw_loaded_signal_preserves_time_origin_and_reports_mismatch(tmp_path):
    times = 0.1234 + np.arange(1273) / 120000
    Omega = SourceConfig().gw_angular_frequency
    signal = 1e-32 * (
        2 * np.cos(Omega * times)
        + 3 * np.sin(Omega * times)
        + 0.4
        + 0.1 * np.sin(2 * Omega * times)
    )
    path = tmp_path / "input.npz"
    np.savez(path, time_s=times, signal=signal)
    run_cli(
        "--angles",
        *PAPER_ANGLES,
        "--signal-input",
        path,
        "--output-dir",
        tmp_path / "analysis",
    )
    report = json.loads((tmp_path / "analysis" / "report.json").read_text())
    expected = project_quadratures(times, signal, Omega)
    assert report["signal_projection"]["cosine"] == expected.cosine
    assert report["signal_projection"]["method"] == "least_squares"
    assert report["sampling"]["first_time_s"] == times[0]
    assert report["input_model_agreement"]["residual_rms"] > 0
    with np.load(tmp_path / "analysis" / "signals.npz") as arrays:
        np.testing.assert_array_equal(arrays["time_s"], times)
    missing = run_cli("--signal-input", path, success=False)
    assert "requires explicit --angles" in missing.stderr


@pytest.mark.parametrize(
    "arguments,match",
    [
        (("--duration", -1), "finite and positive"),
        (("--sample-rate", 0), "finite and positive"),
        (("--sample-rate", 600), "Nyquist"),
        (("--distance", -1), "finite and positive"),
        (("--rotor-delay-rad", "nan"), "finite"),
    ],
)
def test_cli_rejects_invalid_model_and_sampling(tmp_path, arguments, match):
    result = run_cli(
        "--angles", *PAPER_ANGLES, *arguments, "--output-dir", tmp_path, success=False
    )
    assert match in result.stderr
    assert not (tmp_path / "report.json").exists()
