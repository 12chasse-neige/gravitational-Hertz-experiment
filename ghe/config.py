"""Typed configuration loaded from the project YAML files in ``configs/``."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .paths import (
    BEST_POSITION_FILE,
    BEST_POSITION_JSON_FILE,
    CONFIG_DIR,
    DATA_DIR,
    DETECTOR_CONFIG_FILE,
    FREQS_FILE,
    IMG_DIR,
    IMAGES_DIR,
    MAGNITUDE_FILE,
    PAPER_DIR,
    PAPER_FIGURES_DIR,
    REPO_ROOT,
    SCR_DIR,
    SCRIPTS_DIR,
    SOURCE_ARRAY_DISTRIBUTION_FILE,
    SOURCE_ARRAY_NPZ_FILE,
    SOURCE_CONFIG_FILE,
    TOTAL_FREQS_FILE,
    TOTAL_MAGNITUDE_FILE,
    YEAR_SECONDS,
)


class ConfigFileError(ValueError):
    """Raised when a project YAML configuration is missing or malformed."""


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ConfigFileError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"Could not parse YAML configuration {path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ConfigFileError(f"Configuration root must be a mapping: {path}")
    return document


def _yaml_value(document: Mapping[str, Any], path: Path, *keys: str) -> Any:
    value: Any = document
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(value, Mapping) or key not in value:
            dotted_key = ".".join(traversed)
            raise ConfigFileError(f"Missing required key {dotted_key!r} in {path}")
        value = value[key]
    return value


def _float_value(document: Mapping[str, Any], path: Path, *keys: str) -> float:
    value = _yaml_value(document, path, *keys)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        dotted_key = ".".join(keys)
        raise ConfigFileError(f"{dotted_key!r} in {path} must be a number") from exc


def _optional_float_value(
    document: Mapping[str, Any], path: Path, *keys: str
) -> float | None:
    value = _yaml_value(document, path, *keys)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        dotted_key = ".".join(keys)
        raise ConfigFileError(f"{dotted_key!r} in {path} must be a number or null") from exc


def _int_value(document: Mapping[str, Any], path: Path, *keys: str) -> int:
    value = _yaml_value(document, path, *keys)
    if isinstance(value, bool):
        raise ConfigFileError(f"{'.'.join(keys)!r} in {path} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigFileError(f"{'.'.join(keys)!r} in {path} must be an integer") from exc
    if converted != value:
        raise ConfigFileError(f"{'.'.join(keys)!r} in {path} must be an integer")
    return converted


def _bool_value(document: Mapping[str, Any], path: Path, *keys: str) -> bool:
    value = _yaml_value(document, path, *keys)
    if not isinstance(value, bool):
        raise ConfigFileError(f"{'.'.join(keys)!r} in {path} must be true or false")
    return value


_SOURCE_YAML = _read_yaml(SOURCE_CONFIG_FILE)
_DETECTOR_YAML = _read_yaml(DETECTOR_CONFIG_FILE)

_CONSTANT_DEFAULTS = {
    "G": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Constants", "GravitationalConstant"),
    "hbar": _float_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "Constants", "ReducedPlanckConstant"
    ),
    "c": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Constants", "SpeedOfLight"),
}

_SAMPLING_DEFAULTS = {
    "duration_s": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Sampling", "Duration"),
    "sample_rate_hz": _float_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "Sampling", "SampleRate"
    ),
}

_NOISE_DEFAULTS = {
    "model": str(_yaml_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Noise", "Model")),
    "squeeze_db": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Noise", "SqueezingDB"),
    "min_frequency_hz": _float_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "Noise", "FrequencyBand", "Minimum"
    ),
    "max_frequency_hz": _float_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "Noise", "FrequencyBand", "Maximum"
    ),
}

_SOURCE_DEFAULTS = {
    "num": _int_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Source", "Rotor", "HoleCount"),
    "H": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Source", "Rotor", "Length"),
    "D": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Source", "Rotor", "Diameter"),
    "d": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Source", "Rotor", "HoleDiameter"),
    "s": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Source", "Rotor", "HoleOffset"),
    "rho": _float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "Source", "Material", "Density"),
    "G": _CONSTANT_DEFAULTS["G"],
    "c": _CONSTANT_DEFAULTS["c"],
    "omega": _float_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "Source", "Rotor", "AngularVelocity"
    ),
}
_SOURCE_DISTANCE_ARM_LENGTHS = _float_value(
    _SOURCE_YAML,
    SOURCE_CONFIG_FILE,
    "Source",
    "Placement",
    "DetectorDistanceArmLengths",
)


def _detector_defaults_from_yaml(
    document: Mapping[str, Any], path: Path
) -> dict[str, float | None]:
    """Translate standard GWINC IFO keys into ``DetectorConfig`` field names."""

    stages = _yaml_value(document, path, "Suspension", "Stage")
    if not isinstance(stages, list) or not stages or not isinstance(stages[0], Mapping):
        raise ConfigFileError(f"'Suspension.Stage' in {path} must be a non-empty list")
    try:
        testmass = float(stages[0]["Mass"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigFileError(f"'Suspension.Stage[0].Mass' in {path} must be a number") from exc

    # GWINC stores optical losses as dimensionless fractions. The internal
    # detector model historically stores these two values in ppm.
    loss_mirror_ppm = 1.0e6 * _float_value(document, path, "Optics", "Loss")
    loss_bs_ppm = 1.0e6 * _float_value(document, path, "Optics", "BSLoss")
    return {
        "testmass": testmass,
        "length": _float_value(document, path, "Infrastructure", "Length"),
        "length_SR": _float_value(document, path, "Optics", "SRM", "CavityLength"),
        "phi_SR": _optional_float_value(document, path, "Optics", "SRM", "Tunephase"),
        "wavelength": _float_value(document, path, "Laser", "Wavelength"),
        "power": _float_value(document, path, "Laser", "Power"),
        "T_PRM": _float_value(document, path, "Optics", "PRM", "Transmittance"),
        "T_ITM": _float_value(document, path, "Optics", "ITM", "Transmittance"),
        "T_ETM": _float_value(document, path, "Optics", "ETM", "Transmittance"),
        "T_SRM": _float_value(document, path, "Optics", "SRM", "Transmittance"),
        "loss_mirror_ppm": loss_mirror_ppm,
        "loss_BS_ppm": loss_bs_ppm,
    }


_DETECTOR_DEFAULTS = _detector_defaults_from_yaml(_DETECTOR_YAML, DETECTOR_CONFIG_FILE)

_SOURCE_ARRAY_DEFAULTS = {
    "num_sources": _int_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "NumberOfSources"
    ),
    "chunk_size": _int_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "ChunkSize"),
    "spacing": _optional_float_value(_SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "Spacing"),
    "theta_array": _optional_float_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "CenterDirection", "Theta"
    ),
    "phi_array": _optional_float_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "CenterDirection", "Phi"
    ),
    "optimize_each_source": _bool_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "OptimizeEachSource"
    ),
    "chunk_center_approximation": _bool_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "ChunkCenterApproximation"
    ),
    "main_renewal_chunk_center_approximation": _bool_value(
        _SOURCE_YAML,
        SOURCE_CONFIG_FILE,
        "SourceArray",
        "MainRenewalChunkCenterApproximation",
    ),
    "approximation_chunk_size": _int_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "ApproximationChunkSize"
    ),
    "recompute_best_position": _bool_value(
        _SOURCE_YAML, SOURCE_CONFIG_FILE, "SourceArray", "RecomputeBestPosition"
    ),
}


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_optional_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _default_arm_length() -> float:
    return _env_float("LIGO_ARM_LENGTH", _DETECTOR_DEFAULTS["length"])


@dataclass(frozen=True)
class SamplingConfig:
    """
    Shared time-domain sampling configuration.

    ``duration_s`` is the simulated integration window.  ``sample_rate_hz`` sets
    the time grid used before FFT.  Both can be overridden through environment
    variables for compatibility with old workflows.
    """

    duration_s: float = field(
        default_factory=lambda: _env_float("GHE_INT_TIME", _SAMPLING_DEFAULTS["duration_s"])
    )
    sample_rate_hz: float = field(
        default_factory=lambda: _env_float(
            "GHE_SAMPLE_RATE_HZ", _SAMPLING_DEFAULTS["sample_rate_hz"]
        )
    )

    @property
    def num_samples(self) -> int:
        """Number of discrete samples in the time-domain signal."""

        return int(round(self.duration_s * self.sample_rate_hz))

    def time_axis(self) -> np.ndarray:
        """Evenly sampled time axis on ``[0, duration_s)``."""

        return np.linspace(0.0, self.duration_s, self.num_samples, endpoint=False)


@dataclass(frozen=True)
class SourceConfig:
    """
    Physical constants and source-side parameters for one rotating source.

    The dominant quadrupole radiation occurs at ``2*omega``. ``R`` is derived
    from the interferometer arm length using the distance ratio configured in
    ``configs/source.yaml``.
    """

    num: int = _SOURCE_DEFAULTS["num"]  # number of holes on one rotor
    H: float = _SOURCE_DEFAULTS["H"]  # rotor length (m)
    D: float = _SOURCE_DEFAULTS["D"]  # rotor diameter (m)
    d: float = _SOURCE_DEFAULTS["d"]  # hole diameter (m)
    s: float = _SOURCE_DEFAULTS["s"]  # rotor center to hole center (m)
    R: float = field(
        default_factory=lambda: _SOURCE_DISTANCE_ARM_LENGTHS * _default_arm_length()
    )
    rho: float = _SOURCE_DEFAULTS["rho"]  # rotor density (kg/m^3)
    G: float = _SOURCE_DEFAULTS["G"]  # gravitational constant (m^3 kg^-1 s^-2)
    c: float = _SOURCE_DEFAULTS["c"]  # speed of light (m/s)
    omega: float = _SOURCE_DEFAULTS["omega"]  # rotor angular velocity (rad/s)
    L: float = field(default_factory=_default_arm_length)  # arm length used for metric response (m)

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _SOURCE_DISTANCE_ARM_LENGTHS * self.L)

    @property
    def gw_angular_frequency(self) -> float:
        """Dominant gravitational-wave angular frequency emitted by the source."""

        return 2.0 * self.omega

    @property
    def gw_frequency_hz(self) -> float:
        """Dominant gravitational-wave frequency emitted by the source."""

        return self.gw_angular_frequency / (2.0 * np.pi)


@dataclass(frozen=True)
class DetectorConfig:
    """
    Detector and quantum-noise parameters.

    ``phi_SR`` defaults to the detuning phase whose signal-recycling resonance is
    at the source gravitational-wave frequency.
    """

    testmass: float = field(
        default_factory=lambda: _env_float("LIGO_TEST_MASS", _DETECTOR_DEFAULTS["testmass"])
    )
    length: float = field(default_factory=_default_arm_length)                                                   # arm length (m)
    length_SR: float = field(
        default_factory=lambda: _env_float(
            "LIGO_SIGNAL_RECYCLE_ARM_LENGTH", _DETECTOR_DEFAULTS["length_SR"]
        )
    )
    phi_SR: float | None = field(
        default_factory=lambda: _env_optional_float(
            "LIGO_SIGNAL_RECYCLE_DETUNE_PHASE", _DETECTOR_DEFAULTS["phi_SR"]
        )
    )
    resonance_frequency_hz: float | None = None
    hbar: float = _CONSTANT_DEFAULTS["hbar"]
    wavelength: float = _DETECTOR_DEFAULTS["wavelength"]
    c: float = _CONSTANT_DEFAULTS["c"]
    power: float = _DETECTOR_DEFAULTS["power"]
    T_PRM: float = _DETECTOR_DEFAULTS["T_PRM"]
    T_ITM: float = _DETECTOR_DEFAULTS["T_ITM"]
    T_ETM: float = _DETECTOR_DEFAULTS["T_ETM"]
    T_SRM: float = field(
        default_factory=lambda: _env_float("LIGO_SRM_TRANSMITTANCE", _DETECTOR_DEFAULTS["T_SRM"])
    )
    loss_mirror_ppm: float = _DETECTOR_DEFAULTS["loss_mirror_ppm"]
    loss_BS_ppm: float = _DETECTOR_DEFAULTS["loss_BS_ppm"]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectorConfig":
        """Load detector fields from any standard GWINC ``ifo.yaml`` file."""

        input_path = Path(path)
        values = _detector_defaults_from_yaml(_read_yaml(input_path), input_path)
        return cls(**values)

    def __post_init__(self) -> None:
        from .phase import (
            get_resonance_phase_for_detuned_signal_recycling,
            get_source_gw_frequency_hz,
        )

        resonance_frequency_hz = self.resonance_frequency_hz
        if resonance_frequency_hz is None:
            resonance_frequency_hz = get_source_gw_frequency_hz()
            object.__setattr__(self, "resonance_frequency_hz", resonance_frequency_hz)
        if self.phi_SR is None:
            phi_SR = get_resonance_phase_for_detuned_signal_recycling(
                resonance_frequency_hz,
                config=self,
            )
            object.__setattr__(self, "phi_SR", phi_SR)

    @property
    def L(self) -> float:
        """Alias for detector arm length, matching ``SourceConfig.L``."""

        return self.length

    def with_source(self, source: SourceConfig) -> "DetectorConfig":
        """Return a detector config synchronized to a source configuration."""

        return replace(
            self,
            length=source.L,
            resonance_frequency_hz=source.gw_frequency_hz,
            phi_SR=None,
        )


@dataclass(frozen=True)
class SourceArrayConfig:
    """
    Runtime configuration for coherent source-array generation.

    ``chunk_size`` controls streaming/write batches.  ``approximation_chunk_size``
    controls how many sources share one optimized chunk-anchor when that strategy
    is enabled.
    """

    num_sources: int = _SOURCE_ARRAY_DEFAULTS["num_sources"]
    chunk_size: int = _SOURCE_ARRAY_DEFAULTS["chunk_size"]
    spacing: float | None = _SOURCE_ARRAY_DEFAULTS["spacing"]
    theta_array: float | None = _SOURCE_ARRAY_DEFAULTS["theta_array"]
    phi_array: float | None = _SOURCE_ARRAY_DEFAULTS["phi_array"]
    optimize_each_source: bool = _SOURCE_ARRAY_DEFAULTS["optimize_each_source"]
    chunk_center_approximation: bool = _SOURCE_ARRAY_DEFAULTS["chunk_center_approximation"]
    main_renewal_chunk_center_approximation: bool = _SOURCE_ARRAY_DEFAULTS[
        "main_renewal_chunk_center_approximation"
    ]
    approximation_chunk_size: int = _SOURCE_ARRAY_DEFAULTS["approximation_chunk_size"]
    recompute_best_position: bool = _SOURCE_ARRAY_DEFAULTS["recompute_best_position"]


@dataclass(frozen=True)
class NoiseConfig:
    """SNR integration band, squeezing level, and detector-noise model."""

    model: str = _NOISE_DEFAULTS["model"]
    squeeze_db: float = _NOISE_DEFAULTS["squeeze_db"]
    min_frequency_hz: float = _NOISE_DEFAULTS["min_frequency_hz"]
    max_frequency_hz: float = _NOISE_DEFAULTS["max_frequency_hz"]


@dataclass(frozen=True)
class RunConfig:
    """
    Serializable configuration for a full run.

    Run directories write this object to ``config.json`` so a small experiment can
    be reproduced without relying on ambient environment variables.
    """

    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    source_array: SourceArrayConfig = field(default_factory=SourceArrayConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector", self.detector.with_source(self.source))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested dictionary."""

        return asdict(self)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write this configuration to a JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_environment(cls) -> "RunConfig":
        """Build a config using the same environment-variable defaults as scripts."""

        return cls()


# Backward-compatible names used by the original scripts.
TimeSamplingConfig = SamplingConfig
ExperimentConfig = SourceConfig
TIME_SAMPLING = SamplingConfig()
INT_TIME = TIME_SAMPLING.duration_s
NUM = TIME_SAMPLING.num_samples


def build_time_axis() -> np.ndarray:
    """Compatibility helper returning the default global time axis."""

    return TIME_SAMPLING.time_axis()


__all__ = [
    "BEST_POSITION_FILE",
    "BEST_POSITION_JSON_FILE",
    "CONFIG_DIR",
    "ConfigFileError",
    "DATA_DIR",
    "DETECTOR_CONFIG_FILE",
    "DetectorConfig",
    "ExperimentConfig",
    "FREQS_FILE",
    "IMG_DIR",
    "IMAGES_DIR",
    "INT_TIME",
    "MAGNITUDE_FILE",
    "PAPER_DIR",
    "PAPER_FIGURES_DIR",
    "NoiseConfig",
    "NUM",
    "REPO_ROOT",
    "RunConfig",
    "SCR_DIR",
    "SCRIPTS_DIR",
    "SOURCE_ARRAY_DISTRIBUTION_FILE",
    "SOURCE_ARRAY_NPZ_FILE",
    "SOURCE_CONFIG_FILE",
    "SamplingConfig",
    "SourceArrayConfig",
    "SourceConfig",
    "TIME_SAMPLING",
    "TOTAL_FREQS_FILE",
    "TOTAL_MAGNITUDE_FILE",
    "TimeSamplingConfig",
    "YEAR_SECONDS",
    "build_time_axis",
]
