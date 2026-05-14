"""
Typed configuration for physics, detector, sampling, and run settings.

The original scripts mixed constants, environment variables, and file paths.
This module keeps run-critical values serializable while preserving legacy names
such as ``ExperimentConfig``, ``INT_TIME``, and ``NUM`` for compatibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .paths import (
    BEST_POSITION_FILE,
    BEST_POSITION_JSON_FILE,
    DATA_DIR,
    FREQS_FILE,
    IMG_DIR,
    IMAGES_DIR,
    MAGNITUDE_FILE,
    REPO_ROOT,
    SCR_DIR,
    SCRIPTS_DIR,
    SOURCE_ARRAY_DISTRIBUTION_FILE,
    SOURCE_ARRAY_NPZ_FILE,
    TOTAL_FREQS_FILE,
    TOTAL_MAGNITUDE_FILE,
    YEAR_SECONDS,
)


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class SamplingConfig:
    """
    Shared time-domain sampling configuration.

    ``duration_s`` is the simulated integration window.  ``sample_rate_hz`` sets
    the time grid used before FFT.  Both can be overridden through environment
    variables for compatibility with old workflows.
    """

    duration_s: float = field(default_factory=lambda: _env_float("GHE_INT_TIME", 0.01))
    sample_rate_hz: float = field(
        default_factory=lambda: _env_float("GHE_SAMPLE_RATE_HZ", 120000.0)
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
    Physical constants and source-side parameters for one rotating source. The dominant quadrupole radiation occurs at ``2*omega``.
    """

    num: int = 2                        # number of the holes on single rotor
    H: float = 2.0                      # length of the rotor (m)
    D: float = 5.0                      # diameter of the rotor (m)
    d: float = 1.0                      # diameter of the holes (m)
    s: float = 1.5                      # distance from the center of the rotor to the center of the holes (m)
    R: float = field(default_factory=lambda: _env_float("LIGO_ARM_LENGTH", 8000.0))  # distance from source to detector (m)
    rho: float = 1750.0                 # mass density of the rotor material (kg/m^3)
    G: float = 6.674e-11                # gravitational constant (m^3 kg^-1 s^-2)
    c: float = 2.998e8                  # speed of light (m/s)
    omega: float = 300.0 * 2.0 * np.pi  # angular velocity of the rotor (rad/s)
    L: float = field(default_factory=lambda: _env_float("LIGO_ARM_LENGTH", 4000.0))  # arm length used for metric response (m)


@dataclass(frozen=True)
class DetectorConfig:
    """
    Detector and quantum-noise parameters.
    """

    testmass: float = field(default_factory=lambda: _env_float("LIGO_TEST_MASS", 39.6))
    length: float = field(default_factory=lambda: _env_float("LIGO_ARM_LENGTH", 4000.0))
    length_SR: float = field(default_factory=lambda: _env_float("LIGO_SIGNAL_RECYCLE_ARM_LENGTH", 55.0))
    hbar: float = 1.05457e-34
    wavelength: float = 1064e-9
    c: float = SourceConfig.c
    power: float = 125.0
    T_PRM: float = 0.03
    T_ITM: float = 0.014
    T_ETM: float = 5e-6
    T_SRM: float = field(default_factory=lambda: _env_float("LIGO_SRM_TRANSMITTANCE", 0.325))
    loss_mirror_ppm: float = 40.0
    loss_BS_ppm: float = 500.0


@dataclass(frozen=True)
class SourceArrayConfig:
    """
    Runtime configuration for coherent source-array generation.

    ``chunk_size`` controls streaming/write batches.  ``approximation_chunk_size``
    controls how many sources share one optimized chunk-anchor when that strategy
    is enabled.
    """

    num_sources: int = 10_000_000
    chunk_size: int = 100_000
    spacing: float | None = None
    theta_array: float | None = None
    phi_array: float | None = None
    optimize_each_source: bool = True
    chunk_center_approximation: bool = False
    approximation_chunk_size: int = 1_000
    recompute_best_position: bool = False


@dataclass(frozen=True)
class NoiseConfig:
    """SNR integration band, squeezing level, and detector-noise model."""

    # model: str = field(
    #     default_factory=lambda: _env_str(
    #         "GHE_NOISE_MODEL",
    #         "frequency_dependent_squeezed",
    #     )
    # )
    model: str = "detuned_signal_recycling"
    squeeze_db: float = 10.0
    min_frequency_hz: float = 1.0
    max_frequency_hz: float = 5000.0


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
    "DATA_DIR",
    "DetectorConfig",
    "ExperimentConfig",
    "FREQS_FILE",
    "IMG_DIR",
    "IMAGES_DIR",
    "INT_TIME",
    "MAGNITUDE_FILE",
    "NoiseConfig",
    "NUM",
    "REPO_ROOT",
    "RunConfig",
    "SCR_DIR",
    "SCRIPTS_DIR",
    "SOURCE_ARRAY_DISTRIBUTION_FILE",
    "SOURCE_ARRAY_NPZ_FILE",
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
