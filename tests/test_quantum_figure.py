"""The figure compares analytic and GWINC curves with and without SR."""

from dataclasses import replace
import sys

import numpy as np
import pytest
import yaml
from scipy.optimize import minimize_scalar

from ghe.config import DETECTOR_CONFIG_FILE, DetectorConfig
from ghe.noise import (
    get_detuned_signal_recycling_noise_psd,
    squeeze_quantum_noise_with_varying_angle,
)
from scripts.quantumNoise import (
    CE2_SILICON_CONFIG_FILE,
    _gwinc_quantum_psd,
    get_gwinc_detuned_quantum_asd,
    get_gwinc_quantum_asd,
    plot_noise_curve_with_detuned_interferometer,
    tune_gwinc_srm_phase,
)


def test_direct_quantum_calculation_matches_gwinc_budget() -> None:
    gwinc = pytest.importorskip("gwinc")
    frequency = np.array([100.0, 600.0, 1000.0])
    budget = gwinc.load_budget(str(CE2_SILICON_CONFIG_FILE))

    direct_psd = _gwinc_quantum_psd(frequency, budget.ifo)
    budget_psd = budget.run(freq=frequency)["Quantum"].psd
    np.testing.assert_allclose(direct_psd, budget_psd, rtol=1e-12)


def test_blue_curve_opens_srm_and_uses_exact_ideal_db_scaling(monkeypatch) -> None:
    pytest.importorskip("gwinc")
    import scripts.quantumNoise as figure_module

    frequency = np.array([100.0, 600.0, 1000.0])
    seen = {}

    def record_ifo(freq, ifo):
        seen["srm"] = ifo.Optics.SRM.Transmittance
        seen["phase"] = ifo.Optics.SRM.Tunephase
        seen["squeeze_db"] = ifo.Squeezer.AmplitudedB
        seen["squeeze_type"] = ifo.Squeezer.Type
        seen["arm_length"] = ifo.Infrastructure.Length
        return np.ones_like(freq)

    monkeypatch.setattr(figure_module, "_gwinc_quantum_psd", record_ifo)
    actual = get_gwinc_quantum_asd(frequency, squeeze_db=15.0)
    np.testing.assert_allclose(actual, 10.0 ** (-15.0 / 20.0))
    assert seen == {
        "srm": 1.0,
        "phase": 0.0,
        "squeeze_db": 0.0,
        "squeeze_type": "None",
        "arm_length": 40000,
    }


def test_blue_curve_is_independent_of_configured_srm(tmp_path) -> None:
    pytest.importorskip("gwinc")
    frequency = np.array([100.0, 600.0, 1000.0])
    ce_yaml = yaml.safe_load(CE2_SILICON_CONFIG_FILE.read_text())
    ce_yaml["Optics"]["SRM"]["Transmittance"] = 0.8
    ce_yaml["Optics"]["SRM"]["Tunephase"] = 2.0
    edited_path = tmp_path / "ce2.yaml"
    edited_path.write_text(yaml.safe_dump(ce_yaml))

    default_curve = get_gwinc_quantum_asd(frequency, squeeze_db=15.0)
    edited_curve = get_gwinc_quantum_asd(frequency, squeeze_db=15.0, ifo=edited_path)
    np.testing.assert_allclose(edited_curve, default_curve, rtol=1e-12)
    assert default_curve[1] == pytest.approx(1.1138328385473814e-24, rel=1e-5)

    # Other optical fields still come from the selected GWINC configuration.
    ce_yaml["Laser"]["Power"] /= 2
    edited_path.write_text(yaml.safe_dump(ce_yaml))
    lower_power_curve = get_gwinc_quantum_asd(
        frequency, squeeze_db=15.0, ifo=edited_path
    )
    assert np.max(np.abs(lower_power_curve / default_curve - 1.0)) > 0.01


def test_detuned_gwinc_minimum_is_at_source_frequency() -> None:
    gwinc = pytest.importorskip("gwinc")
    detector = DetectorConfig.from_yaml(DETECTOR_CONFIG_FILE)
    asd, tunephase = get_gwinc_detuned_quantum_asd(
        np.array([10.0, 600.0]), detector_config=detector
    )
    assert tunephase == pytest.approx(0.01142715189167273, rel=1e-6)
    assert asd[1] == pytest.approx(1.8695542025700354e-25, rel=1e-5)
    # The b2 signal response is weak off resonance in both models. Reading
    # GWINC's b1 instead produces a spurious factor-of-50 gap near 10 Hz.
    analytic_low = np.sqrt(
        get_detuned_signal_recycling_noise_psd(np.array([10.0]), config=detector)[0]
    )
    assert 1.0 < asd[0] / analytic_low < 1.5

    # Find the frequency minimum independently of the slope-root tuner.
    ifo = gwinc.load_budget(str(CE2_SILICON_CONFIG_FILE)).ifo
    ifo.Optics.SRM.Transmittance = detector.T_SRM
    ifo.Optics.SRM.CavityLength = detector.length_SR
    ifo.Optics.SRM.Tunephase = tunephase
    ifo.Optics.Quadrature.dc = 0.0
    ifo.Squeezer.AmplitudedB = 0.0
    ifo.Squeezer.Type = "None"
    minimum = minimize_scalar(
        lambda f: float(_gwinc_quantum_psd(np.array([f]), ifo)[0]),
        bounds=(590.0, 610.0),
        method="bounded",
        options={"xatol": 1e-7},
    )
    assert minimum.success
    assert abs(minimum.x - 600.0) < 0.01


def test_detuned_gwinc_retunes_when_source_frequency_changes(tmp_path) -> None:
    pytest.importorskip("gwinc")
    detector = DetectorConfig.from_yaml(DETECTOR_CONFIG_FILE)
    retargeted = replace(detector, resonance_frequency_hz=700.0, phi_SR=None)
    # An explicit readout in an external YAML must also be changed to b2.
    ce_yaml = yaml.safe_load(CE2_SILICON_CONFIG_FILE.read_text())
    ce_yaml["Squeezer"]["Readout"] = {"Type": "Homodyne", "Angle": np.pi / 2}
    edited_path = tmp_path / "ce2-readout.yaml"
    edited_path.write_text(yaml.safe_dump(ce_yaml))
    asd, tunephase = get_gwinc_detuned_quantum_asd(
        np.array([699.0, 700.0, 701.0]), detector_config=retargeted, ifo=edited_path
    )
    assert tunephase == pytest.approx(0.008986015652592263, rel=1e-5)
    assert asd[1] < asd[0]
    assert asd[1] < asd[2]


def test_figure_prints_all_four_target_values(tmp_path, capsys) -> None:
    pytest.importorskip("gwinc")
    output = tmp_path / "quantum.png"
    analytic_target = plot_noise_curve_with_detuned_interferometer(
        output_path=output,
        points=24,
    )
    output_text = capsys.readouterr().out
    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert "GWINC detuned readout: b_2 (Optics.Quadrature.dc = 0 rad)" in output_text
    assert "GWINC tuned SRM Tunephase at 600 Hz" in output_text
    assert "GWINC CE2 no-SR ideal ASD at 600 Hz" in output_text
    assert "GWINC CE2 tuned-SR ideal ASD at 600 Hz" in output_text
    assert "Analytic no-SR ideal ASD at 600 Hz" in output_text
    assert "Analytic detuned-SR ideal ASD at 600 Hz" in output_text
    assert analytic_target == pytest.approx(3.4896826024322485e-27, rel=1e-6)


def test_detector_squeezer_edit_changes_all_four_curves(tmp_path) -> None:
    pytest.importorskip("gwinc")
    detector_yaml = yaml.safe_load(DETECTOR_CONFIG_FILE.read_text())
    detector_yaml["Squeezer"]["AmplitudedB"] = 12.0
    edited_path = tmp_path / "detector.yaml"
    edited_path.write_text(yaml.safe_dump(detector_yaml))

    original = DetectorConfig.from_yaml(DETECTOR_CONFIG_FILE)
    edited = DetectorConfig.from_yaml(edited_path)
    frequency = np.array([600.0])
    expected_asd_ratio = 10.0 ** (3.0 / 20.0)

    original_simple = squeeze_quantum_noise_with_varying_angle(frequency, config=original)
    edited_simple = squeeze_quantum_noise_with_varying_angle(frequency, config=edited)
    np.testing.assert_allclose(np.sqrt(edited_simple / original_simple), expected_asd_ratio)

    original_detuned = get_detuned_signal_recycling_noise_psd(frequency, config=original)
    edited_detuned = get_detuned_signal_recycling_noise_psd(frequency, config=edited)
    np.testing.assert_allclose(np.sqrt(edited_detuned / original_detuned), expected_asd_ratio)

    original_gwinc = get_gwinc_quantum_asd(frequency, squeeze_db=original.squeeze_db)
    edited_gwinc = get_gwinc_quantum_asd(frequency, squeeze_db=edited.squeeze_db)
    np.testing.assert_allclose(edited_gwinc / original_gwinc, expected_asd_ratio)

    original_tuned, original_phase = get_gwinc_detuned_quantum_asd(
        frequency, detector_config=original
    )
    edited_tuned, edited_phase = get_gwinc_detuned_quantum_asd(
        frequency, detector_config=edited
    )
    np.testing.assert_allclose(edited_tuned / original_tuned, expected_asd_ratio)
    assert edited_phase == pytest.approx(original_phase, rel=1e-10)

    # The hybrid's SRM choice belongs to both detuned curves, while blue stays
    # an independent no-SR reference.
    changed_sr = replace(original, T_SRM=1e-4, phi_SR=None)
    changed_detuned = get_detuned_signal_recycling_noise_psd(frequency, config=changed_sr)
    assert not np.isclose(changed_detuned[0], original_detuned[0], rtol=1e-3, atol=0)
    changed_tuned, changed_phase = get_gwinc_detuned_quantum_asd(
        frequency, detector_config=changed_sr
    )
    assert not np.isclose(changed_tuned[0], original_tuned[0], rtol=1e-3, atol=0)
    assert changed_phase != pytest.approx(original_phase, rel=0, abs=1e-9)
    np.testing.assert_allclose(
        get_gwinc_quantum_asd(frequency, squeeze_db=changed_sr.squeeze_db),
        original_gwinc,
        rtol=1e-12,
    )


def test_tuner_rejects_fully_transmitting_srm() -> None:
    gwinc = pytest.importorskip("gwinc")
    ifo = gwinc.load_budget(str(CE2_SILICON_CONFIG_FILE)).ifo
    ifo.Optics.SRM.Transmittance = 1.0
    with pytest.raises(ValueError, match="requires 0 < SRM transmittance < 1"):
        tune_gwinc_srm_phase(ifo, 600.0)


def test_broad_srm_uses_matched_detuning_when_target_minimum_is_unavailable() -> None:
    pytest.importorskip("gwinc")
    detector = replace(
        DetectorConfig.from_yaml(DETECTOR_CONFIG_FILE), T_SRM=0.5, phi_SR=None
    )
    frequency = np.array([10.0, 100.0, 600.0])
    with pytest.raises(ValueError, match="no positive-detuning quantum-noise dip"):
        get_gwinc_detuned_quantum_asd(frequency, detector_config=detector)

    asd, tunephase = get_gwinc_detuned_quantum_asd(
        frequency, detector_config=detector, sr_tuning="analytic-phase"
    )
    assert tunephase == pytest.approx(np.pi - 2 * detector.phi_SR, abs=1e-13)
    analytic = np.sqrt(get_detuned_signal_recycling_noise_psd(frequency, config=detector))
    assert np.all(np.isfinite(asd))
    # Matching the physical detuning and b2 readout leaves small corrections
    # from GWINC's complete optics at this broad SRM transmittance.
    np.testing.assert_allclose(asd, analytic, rtol=0.05, atol=0)


def test_cli_can_plot_broad_srm_with_matched_detuning(tmp_path, capsys, monkeypatch) -> None:
    pytest.importorskip("gwinc")
    from scripts.quantumNoise import main

    output = tmp_path / "broad-srm.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantumNoise.py",
            "--t-srm", "0.5",
            "--gwinc-sr-tuning", "analytic-phase",
            "--points", "24",
            "--output", str(output),
        ],
    )
    main()
    output_text = capsys.readouterr().out
    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert "GWINC SR tuning: analytic-phase" in output_text
    assert "GWINC SRM Tunephase (analytic detuning)" in output_text
    assert "T_SRM=0.5" in output_text
    assert "GWINC CE2 matched-detuning SR ideal ASD" in output_text


def test_cli_can_retarget_both_detuned_curves(tmp_path, capsys, monkeypatch) -> None:
    pytest.importorskip("gwinc")
    from scripts.quantumNoise import main

    output = tmp_path / "retargeted.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantumNoise.py",
            "--target-frequency-hz",
            "700",
            "--points",
            "24",
            "--output",
            str(output),
        ],
    )
    main()
    output_text = capsys.readouterr().out
    assert output.is_file()
    assert "GWINC tuned SRM Tunephase at 700 Hz" in output_text
    assert "Analytic detuned-SR ideal ASD at 700 Hz" in output_text
