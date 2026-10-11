"""Physical scaling, noise accounting and bounded-search acceptance checks."""

from dataclasses import replace
import json
import numpy as np
import pytest

from ghe.config import SourceConfig
from ghe.paths import DETECTOR_CONFIG_FILE, SOURCE_CONFIG_FILE, YEAR_SECONDS
from ghe.qdot_scaling import QdotArmLengthStudy, sweep_qdot_arm_lengths
from ghe.thermal_gas import calculate_thermal_gas
from scripts.sweepQdotArmLength import main


ANGLES = (np.pi / 2, np.pi / 2, 0.0, 0.0)


def test_target_only_noise_matches_full_grid_and_does_not_mutate_inputs():
    before = [path.read_bytes() for path in (DETECTOR_CONFIG_FILE, SOURCE_CONFIG_FILE)]
    full = calculate_thermal_gas()
    small = calculate_thermal_gas(arm_length_m=40000, frequency_hz=[600])
    assert small.frequency_hz.tolist() == [600]
    index = int(np.flatnonzero(full.frequency_hz == 600)[0])
    for name in ("ThermalTotal", "ResidualGas", "ThermalPlusGas"):
        np.testing.assert_allclose(
            small.spectra_psd[name][0],
            full.spectra_psd[name][index],
            rtol=1e-13,
            atol=0,
        )
    shorter = calculate_thermal_gas(arm_length_m=100, frequency_hz=[600])
    assert shorter.metadata["effective_detector"]["Infrastructure"]["Length"] == 100
    assert (
        shorter.metadata["effective_detector"]["Optics"]["Curvature"]
        == full.metadata["effective_detector"]["Optics"]["Curvature"]
    )
    assert shorter.spectra_psd["ThermalTotal"][0] > small.spectra_psd["ThermalTotal"][0]
    assert before == [
        path.read_bytes() for path in (DETECTOR_CONFIG_FILE, SOURCE_CONFIG_FILE)
    ]


def test_snr_uses_psd_sum_once_and_coherent_duration_scaling():
    source = SourceConfig()
    study = QdotArmLengthStudy(angles=ANGLES, source_config=source)
    row = study.evaluate(40000)
    assert row["source_distance_m"] == 60000
    total = row["quantum_psd"] + row["thermal_psd"] + row["gas_psd"]
    assert row["total_psd"] == total
    assert row["rho_qdot"] == pytest.approx(
        row["qdot_peak_strain"] * np.sqrt(YEAR_SECONDS / total), rel=1e-14, abs=0
    )
    np.testing.assert_allclose(
        [row["quantum_psd"], row["thermal_psd"], row["gas_psd"]],
        [1.2177884665718313e-53, 1.5273788842555156e-52, 5.439225004672221e-52],
        rtol=1e-7,
        atol=0,
    )
    longer = QdotArmLengthStudy(
        angles=ANGLES, observation_time_s=4 * YEAR_SECONDS
    ).evaluate(40000)
    assert longer["rho_qdot"] == pytest.approx(2 * row["rho_qdot"], rel=1e-13, abs=0)
    assert study.evaluate(40000) is row


def test_optimized_qdot_is_at_least_as_good_as_fixed_reference():
    fixed = QdotArmLengthStudy(angles=ANGLES).evaluate(100)
    optimized = QdotArmLengthStudy().evaluate(100)
    assert optimized["rho_qdot"] >= fixed["rho_qdot"] * (1 - 1e-12)
    assert optimized["optimization"]["successful_starts"] > 0
    assert optimized["rho_qdot"] == pytest.approx(22.0816790549, rel=2e-6, abs=0)
    assert optimized["source_distance_m"] == 150
    assert optimized["minimum_source_arm_distance_m"] > 50 - 1e-9


def test_sweep_reports_boundary_and_excludes_inconsistent_optics():
    study = sweep_qdot_arm_lengths(
        [100, 1000, 30000, 40000, 59000, 61000], angles=ANGLES
    )
    assert study["best_location"] == "lower_bound"
    assert study["best"]["arm_length_m"] == 100
    assert [row["arm_length_m"] for row in study["grid"]] == [100, 1000, 40000]
    rejected = {row["arm_length_m"]: row["reason"] for row in study["rejected_lengths"]}
    assert "degenerate" in rejected[30000]
    assert "clipping" in rejected[59000]
    assert "unstable" in rejected[61000]
    for row in study["grid"]:
        assert row["source_distance_m"] == 1.5 * row["arm_length_m"]
        assert 0 < row["cavity"]["g1_g2"] < 1
    assert study["mirror_curvatures_m"] == {"ITM": 30000, "ETM": 30000}


def test_refinement_finds_an_interior_peak(monkeypatch):
    # An independently defined smooth objective exercises search refinement;
    # real field/noise behavior is covered by the numerical acceptance checks.
    def evaluate(self, length):
        return {
            "arm_length_m": float(length),
            "rho_qdot": float(np.exp(-(np.log(length / 700) ** 2))),
        }

    monkeypatch.setattr(QdotArmLengthStudy, "evaluate", evaluate)
    result = sweep_qdot_arm_lengths([100, 500, 2000], angles=ANGLES)
    assert result["best_location"] == "interior"
    assert result["best"]["arm_length_m"] == pytest.approx(700, rel=2e-6)
    assert len(result["refined_peaks"]) == 1


def test_cli_artifacts(tmp_path):
    report = main(
        [
            "--min-length",
            "100",
            "--max-length",
            "2000",
            "--points",
            "3",
            "--angles",
            *map(str, ANGLES),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert report["best_location"] == "lower_bound"
    saved = json.loads((tmp_path / "qdot_arm_length.json").read_text())
    assert saved["best"] == report["best"]
    csv = np.genfromtxt(
        tmp_path / "qdot_arm_length.csv",
        names=True,
        delimiter=",",
        dtype=None,
        encoding="utf-8",
    )
    assert csv.size == 3
    np.testing.assert_allclose(csv["theta_src_rad"], ANGLES[0], rtol=1e-14, atol=0)
    assert (tmp_path / "qdot_arm_length.png").stat().st_size > 0


@pytest.mark.parametrize(
    "lengths", [[], [100], [100, 100], [0, 100], [-1, 100], [100, np.nan]]
)
def test_invalid_sweep_lengths(lengths):
    with pytest.raises(ValueError):
        sweep_qdot_arm_lengths(lengths, angles=ANGLES)


def test_invalid_duration_angles_frequency_and_no_feasible_lengths():
    with pytest.raises(ValueError, match="positive"):
        QdotArmLengthStudy(observation_time_s=0)
    with pytest.raises(ValueError, match="four finite"):
        QdotArmLengthStudy(angles=[1, 2, 3])
    with pytest.raises(ValueError, match="frequency must match"):
        QdotArmLengthStudy(source_config=replace(SourceConfig(), omega=1000))
    with pytest.raises(ValueError, match="No optically feasible"):
        sweep_qdot_arm_lengths([59000, 61000])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arm_length_m": -1},
        {"frequency_hz": []},
        {"frequency_hz": [0]},
        {"frequency_hz": [np.nan]},
        {"frequency_hz": [[600]]},
    ],
)
def test_invalid_thermal_gas_overrides(kwargs):
    with pytest.raises(ValueError):
        calculate_thermal_gas(**kwargs)
