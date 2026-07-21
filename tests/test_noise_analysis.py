from __future__ import annotations

from unittest.mock import patch

import numpy as np

import scr.noiseAnalysis as noise_analysis
from ghe.config import SamplingConfig
from ghe.spectrum import Spectrum


def test_calculate_snr_regenerates_spectrum_by_default() -> None:
    expected_spectrum = Spectrum(
        signal=np.array([0.0, 1.0, 0.0, -1.0]),
        magnitude=np.array([0.5, 0.0]),
        freqs=np.array([1.0, 2.0]),
    )
    calls: dict[str, object] = {}

    def fake_generate_current_spectrum(*, sampling_config=None):
        calls["sampling_config"] = sampling_config
        return expected_spectrum

    def fake_calculate_snr_from_arrays(magnitude, freqs, **kwargs):
        calls["magnitude"] = magnitude
        calls["freqs"] = freqs
        return 12.5

    sampling = SamplingConfig(duration_s=1.0, sample_rate_hz=4.0)
    with patch.object(
        noise_analysis,
        "generate_current_spectrum",
        fake_generate_current_spectrum,
    ), patch.object(
        noise_analysis,
        "_calculate_snr_from_arrays",
        fake_calculate_snr_from_arrays,
    ):
        result = noise_analysis.calculate_snr(
            sampling_config=sampling,
            verbose=False,
        )

    assert result == 12.5
    assert calls["sampling_config"] is sampling
    np.testing.assert_array_equal(calls["magnitude"], expected_spectrum.magnitude)
    np.testing.assert_array_equal(calls["freqs"], expected_spectrum.freqs)


def test_calculate_snr_with_paths_keeps_saved_spectrum_workflow() -> None:
    calls: dict[str, object] = {}

    def fake_calculate_snr(**kwargs):
        calls.update(kwargs)
        return 3.0

    with patch.object(noise_analysis, "_calculate_snr", fake_calculate_snr):
        result = noise_analysis.calculate_snr(
            "custom_magnitude.npy",
            "custom_freqs.npy",
            verbose=False,
        )

    assert result == 3.0
    assert calls["magnitude_path"] == "custom_magnitude.npy"
    assert calls["freq_path"] == "custom_freqs.npy"
