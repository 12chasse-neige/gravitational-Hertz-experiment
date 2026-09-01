from __future__ import annotations

import sys

import yaml

import main as main_cli
from ghe.config import (
    DETECTOR_CONFIG_FILE,
    SOURCE_CONFIG_FILE,
    DetectorConfig,
    NoiseConfig,
    SamplingConfig,
    SourceArrayConfig,
    SourceConfig,
)
from scr import quantumNoise, sourceArray


def _read_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_dataclass_defaults_come_from_yaml_files() -> None:
    detector_yaml = _read_yaml(DETECTOR_CONFIG_FILE)
    source_yaml = _read_yaml(SOURCE_CONFIG_FILE)

    detector = DetectorConfig()
    sampling = SamplingConfig()
    noise = NoiseConfig()
    source = SourceConfig()
    source_array = SourceArrayConfig()

    assert detector.testmass == detector_yaml["Suspension"]["Stage"][0]["Mass"]
    assert detector.length == detector_yaml["Infrastructure"]["Length"]
    assert detector.T_ITM == detector_yaml["Optics"]["ITM"]["Transmittance"]
    assert detector.T_SRM == detector_yaml["Optics"]["SRM"]["Transmittance"]
    assert detector.length_SR == detector_yaml["Optics"]["SRM"]["CavityLength"]
    assert detector.loss_mirror_ppm == 1.0e6 * detector_yaml["Optics"]["Loss"]
    assert detector.loss_BS_ppm == 1.0e6 * detector_yaml["Optics"]["BSLoss"]
    assert sampling.duration_s == source_yaml["Sampling"]["Duration"]
    assert noise.model == source_yaml["Noise"]["Model"]

    assert source.num == source_yaml["Source"]["Rotor"]["HoleCount"]
    assert source.D == source_yaml["Source"]["Rotor"]["Diameter"]
    assert source.omega == source_yaml["Source"]["Rotor"]["AngularVelocity"]
    distance_ratio = source_yaml["Source"]["Placement"]["DetectorDistanceArmLengths"]
    assert source.R == distance_ratio * source.L
    assert source_array.num_sources == source_yaml["SourceArray"]["NumberOfSources"]
    assert source_array.chunk_size == source_yaml["SourceArray"]["ChunkSize"]
    assert source_array.chunk_center_approximation is source_yaml["SourceArray"][
        "ChunkCenterApproximation"
    ]
    assert source_array.main_renewal_chunk_center_approximation is source_yaml[
        "SourceArray"
    ]["MainRenewalChunkCenterApproximation"]


def test_detector_config_loads_standard_gwinc_yaml(tmp_path) -> None:
    gwinc_path = tmp_path / "ifo.yaml"
    gwinc_path.write_text(
        """
Infrastructure:
  Length: 3995
Suspension:
  Stage:
    - Mass: 39.6
Laser:
  Wavelength: 1.064e-6
  Power: 125
Optics:
  Loss: 40e-6
  BSLoss: 0.5e-3
  ITM:
    Transmittance: 0.014
  ETM:
    Transmittance: 5e-6
  PRM:
    Transmittance: 0.03
  SRM:
    Transmittance: 0.325
    CavityLength: 55
    Tunephase: 0.1
Seismic:
  Site: LHO
""".strip(),
        encoding="utf-8",
    )

    detector = DetectorConfig.from_yaml(gwinc_path)

    assert detector.length == 3995.0
    assert detector.testmass == 39.6
    assert detector.T_SRM == 0.325
    assert detector.length_SR == 55.0
    assert detector.phi_SR == 0.1
    assert detector.loss_mirror_ppm == 40.0
    assert detector.loss_BS_ppm == 500.0


def test_explicit_config_and_environment_overrides_still_win(monkeypatch) -> None:
    assert SourceConfig(D=9.0).D == 9.0
    assert DetectorConfig(T_SRM=0.2).T_SRM == 0.2

    monkeypatch.setenv("LIGO_ARM_LENGTH", "1234")
    monkeypatch.setenv("GHE_SAMPLE_RATE_HZ", "4321")
    assert SourceConfig().L == 1234.0
    assert DetectorConfig().length == 1234.0
    assert SamplingConfig().sample_rate_hz == 4321.0


def test_existing_cli_flags_override_yaml_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--source-array-num-sources", "17", "--source-array-chunk-size", "3"],
    )
    main_args = main_cli.parse_arguments()
    assert main_args.source_array_num_sources == 17
    assert main_args.source_array_chunk_size == 3
    assert main_args.source_array_chunk_center_approximation is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["sourceArray.py", "--num-sources", "19", "--spacing", "8.5"],
    )
    source_args = sourceArray.parse_arguments()
    assert source_args.num_sources == 19
    assert source_args.spacing == 8.5

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantumNoise.py",
            "--detector-config",
            "comparison.yaml",
            "--length-sr",
            "70",
            "--t-srm",
            "0.2",
        ],
    )
    detector_args = quantumNoise.parse_arguments()
    assert detector_args.length_sr == 70.0
    assert detector_args.t_srm == 0.2
    assert detector_args.detector_config.name == "comparison.yaml"
