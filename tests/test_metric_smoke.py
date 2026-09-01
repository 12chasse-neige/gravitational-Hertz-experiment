from __future__ import annotations

import numpy as np

from ghe.metric import calculate_metric_response
from ghe.optimization import load_best_geometry
from scr.metricCalculate import calculate_metric_response as legacy_calculate_metric_response


def test_explicit_metric_matches_legacy_default() -> None:
    best = load_best_geometry()
    assert best is not None

    package_value = calculate_metric_response(0.0, *best.angles)
    legacy_value = legacy_calculate_metric_response(0.0)
    assert np.isclose(package_value, legacy_value, rtol=1e-12, atol=0.0)


def test_legacy_metric_defaults_to_zero_time_and_best_position() -> None:
    best = load_best_geometry()
    assert best is not None

    expected = calculate_metric_response(0.0, *best.angles)
    actual = legacy_calculate_metric_response()
    assert np.isclose(actual, expected, rtol=1e-12, atol=0.0)
