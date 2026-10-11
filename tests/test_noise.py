from __future__ import annotations

import numpy as np
from dataclasses import replace

from ghe.config import DetectorConfig, NoiseConfig
from ghe.noise import (
    get_detuned_signal_recycling_noise_psd,
    get_noise_psd,
    normalize_noise_model,
    squeeze_quantum_noise_with_varying_angle,
)


def test_detuned_sr_matches_previous_model_for_open_zero_phase_sr() -> None:
    freqs = np.array([20.0, 100.0, 600.0, 1000.0])
    detector_config = DetectorConfig(T_SRM=1.0, length_SR=0.0, phi_SR=0.0)
    squeeze_db = 10.0

    previous = squeeze_quantum_noise_with_varying_angle(
        freqs,
        squeeze_db=squeeze_db,
        config=detector_config,
    )
    detuned = get_detuned_signal_recycling_noise_psd(
        freqs,
        squeeze_db=squeeze_db,
        config=detector_config,
    )

    assert np.allclose(detuned, previous, rtol=1e-12, atol=0.0)


def test_noise_selector_accepts_detuned_alias() -> None:
    freqs = np.array([100.0, 600.0, 1000.0])
    detector_config = DetectorConfig(T_SRM=0.325, length_SR=55.0)
    noise_config = NoiseConfig(model="detuned", squeeze_db=8.0)

    selected = get_noise_psd(
        freqs,
        noise_config=noise_config,
        detector_config=detector_config,
    )
    direct = get_detuned_signal_recycling_noise_psd(
        freqs,
        squeeze_db=noise_config.squeeze_db,
        config=detector_config,
    )

    assert normalize_noise_model("detuned") == "detuned_signal_recycling"
    assert np.allclose(selected, direct)


def test_detector_squeezer_controls_default_analytic_noise_and_snr() -> None:
    from ghe.snr import calculate_snr_from_phasor

    detector = DetectorConfig()
    quieter_detector = replace(detector, squeeze_db=detector.squeeze_db + 5.0)
    frequency = np.array([600.0])
    baseline_psd = get_noise_psd(frequency, detector_config=detector)
    quieter_psd = get_noise_psd(frequency, detector_config=quieter_detector)

    # Ideal analytic squeezing multiplies the PSD by 10^(-dB/10).
    np.testing.assert_allclose(quieter_psd / baseline_psd, 10.0 ** (-5.0 / 10.0))
    np.testing.assert_allclose(
        get_detuned_signal_recycling_noise_psd(frequency, config=quieter_detector),
        quieter_psd,
    )
    baseline_snr = calculate_snr_from_phasor(1e-23, 600.0, detector_config=detector)
    quieter_snr = calculate_snr_from_phasor(1e-23, 600.0, detector_config=quieter_detector)
    np.testing.assert_allclose(quieter_snr / baseline_snr, 10.0 ** (5.0 / 20.0))

    # A one-run override remains possible without editing the detector YAML.
    override_psd = get_noise_psd(
        frequency, detector_config=detector, noise_config=NoiseConfig(squeeze_db=8.0)
    )
    np.testing.assert_allclose(
        override_psd,
        get_detuned_signal_recycling_noise_psd(frequency, squeeze_db=8.0, config=detector),
    )
