"""
Signal-to-noise-ratio integration.

Inputs are FFT magnitudes and their corresponding positive frequencies.  The
integration follows the original project convention:

    SNR = sqrt(sum(4 * |h(f)|^2 / S_h(f)) * df)

By default, the sampled integration time is inferred from the spectrum spacing
as ``duration_s = 1 / df``.  This keeps saved spectra self-contained: changing
the current sampling configuration does not change the SNR of an unchanged
``freq``/``magnitude`` pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .config import DetectorConfig, NoiseConfig, SamplingConfig, SourceConfig
from .artifacts import model_metadata, read_metadata, validate_metadata, file_digest
from dataclasses import asdict
from .noise import get_noise_psd
from .paths import FREQS_FILE, MAGNITUDE_FILE, YEAR_SECONDS


def infer_duration_from_frequency_spacing(freq: np.ndarray) -> float:
    """
    Infer the time-domain duration represented by evenly spaced FFT bins.

    ``scipy.fft.rfftfreq`` produces bins separated by ``1 / duration_s``.  The
    zero-frequency bin may already be omitted, so only spacing matters here.
    """

    freq = np.asarray(freq, dtype=float)
    if freq.size < 2:
        raise ValueError("Frequency array must contain at least two bins.")

    diffs = np.diff(freq)
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError("Frequency array must be finite and strictly increasing.")

    df = float(np.median(diffs))
    if not np.allclose(diffs, df, rtol=1e-9, atol=max(abs(df) * 1e-12, 1e-15)):
        raise ValueError("Frequency bins must be evenly spaced to infer duration.")

    return 1.0 / df


def calculate_snr_from_arrays(
    signal_magnitude: np.ndarray,
    freq: np.ndarray,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    noise_psd_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """
    Calculate 1-year SNR from in-memory spectrum arrays.

    ``noise_psd_func`` is injectable for tests and model comparisons.  When it is
    omitted, the package uses the model selected by ``NoiseConfig.model``.
    ``sampling_config`` is accepted for API compatibility; duration is inferred
    from ``freq`` so the spectrum stays self-consistent.
    """

    active_noise = noise_config or NoiseConfig()
    active_detector = detector_config or DetectorConfig()

    signal_magnitude = np.asarray(signal_magnitude, dtype=float)
    freq = np.asarray(freq, dtype=float)
    valid_mask = (freq >= active_noise.min_frequency_hz) & (
        freq <= active_noise.max_frequency_hz
    )
    freq_valid = freq[valid_mask]
    signal_magnitude_valid = signal_magnitude[valid_mask]

    if freq_valid.size < 2:
        raise ValueError(
            "Frequency array has too few points in the configured SNR band. "
            "Check generated spectrum data."
        )

    if noise_psd_func is None:
        total_noise_psd = get_noise_psd(
            freq_valid,
            noise_config=active_noise,
            detector_config=active_detector,
        )
    else:
        total_noise_psd = noise_psd_func(freq_valid)

    # Frequencies are evenly spaced because they come from rfftfreq.  The
    # integral is therefore a rectangular sum over the selected frequency band.
    integrand = (4.0 * signal_magnitude_valid**2) / total_noise_psd
    df = freq_valid[1] - freq_valid[0]
    snr = np.sqrt(np.sum(integrand) * df)
    duration_s = infer_duration_from_frequency_spacing(freq_valid)
    return float(snr * np.sqrt(YEAR_SECONDS / duration_s))


def calculate_snr(
    magnitude_path: str | Path = MAGNITUDE_FILE,
    freq_path: str | Path = FREQS_FILE,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    source_config: SourceConfig | None = None,
) -> float:
    """Load legacy spectrum arrays from disk and calculate 1-year SNR."""

    metadata = read_metadata(magnitude_path)
    validate_metadata(metadata, source_config)
    if read_metadata(freq_path) != metadata:
        raise ValueError("Spectrum files belong to different runs; regenerate the pair")
    if metadata.get("magnitude_sha256") != file_digest(magnitude_path) or metadata.get(
        "frequency_sha256"
    ) != file_digest(freq_path):
        raise ValueError(
            "Spectrum payload changed; regenerate the spectrum and metadata"
        )
    cfg = source_config or SourceConfig()
    detector_config = (detector_config or DetectorConfig()).with_source(cfg)
    signal_magnitude = np.load(magnitude_path)
    freq = np.load(freq_path)
    return calculate_snr_from_arrays(
        signal_magnitude,
        freq,
        noise_config=noise_config,
        detector_config=detector_config,
        sampling_config=sampling_config,
    )


def save_snr_json(
    snr_year: float,
    output_path: str | Path,
    *,
    source_config=None,
    detector_config=None,
    noise_config=None,
) -> None:
    """Persist a small machine-readable SNR summary for run directories."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "snr_year": snr_year,
                **model_metadata(source_config),
                "detector_config": asdict(
                    detector_config
                    or DetectorConfig().with_source(source_config or SourceConfig())
                ),
                "noise_config": asdict(noise_config or NoiseConfig()),
                "interpretation": "conditional ideal free-mass response / strain-noise calibration proxy",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def calculate_snr_from_phasor(
    phasor: complex,
    gw_frequency_hz: float,
    *,
    noise_config: NoiseConfig | None = None,
    detector_config: DetectorConfig | None = None,
) -> float:
    """
    Estimate the 1-year SNR from a monochromatic phasor sum.

    The phasor ``H = Σ A_i exp(j φ_i)`` represents the total strain amplitude at
    the dominant GW frequency ``f_gw``.  For a single-bin signal the integral
    collapses to::

        SNR² = |H|² · T_year / S_h(f_gw)
    """

    active_noise = noise_config or NoiseConfig()
    active_detector = detector_config or DetectorConfig()

    if (
        not np.isfinite(gw_frequency_hz)
        or gw_frequency_hz <= 0
        or not np.isfinite(phasor)
    ):
        raise ValueError("Require a finite phasor and positive finite frequency")
    if (
        not active_noise.min_frequency_hz
        <= gw_frequency_hz
        <= active_noise.max_frequency_hz
    ):
        raise ValueError("Signal frequency is outside the configured SNR band")
    freq = np.array([gw_frequency_hz], dtype=float)
    noise_psd = get_noise_psd(
        freq,
        noise_config=active_noise,
        detector_config=active_detector,
    )

    snr_squared = abs(phasor) ** 2 * YEAR_SECONDS / float(noise_psd[0])
    return float(np.sqrt(snr_squared))
