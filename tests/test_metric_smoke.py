"""The script facade and package must describe the same observable."""

import numpy as np
from ghe.metric import calculate_metric_response
from ghe.optimization import BestGeometry, get_signal_amplitude
from scripts.metricCalculate import calculate_metric_response as script_response


def test_script_default_geometry_and_time(monkeypatch):
    angles = (0.6, 0.8, 0.4, 0.8)
    geometry = BestGeometry(*angles, get_signal_amplitude(*angles))
    monkeypatch.setattr(
        "ghe.optimization.solve_best_geometry", lambda **kwargs: geometry
    )
    np.testing.assert_allclose(
        script_response(), calculate_metric_response(0, *angles), rtol=1e-12, atol=0
    )


def test_script_explicit_geometry():
    angles = (0.7, 0.9, 0.3, 0.2)
    np.testing.assert_allclose(
        script_response(0.001, *angles, R=7000),
        calculate_metric_response(0.001, *angles, R=7000),
        rtol=1e-12,
        atol=0,
    )


def test_script_distance_override_reaches_default_geometry_solver(monkeypatch):
    angles = (0.6, 0.8, 0.4, 0.8)
    seen = []

    def solver(*, config):
        seen.append(config.R)
        return BestGeometry(*angles, get_signal_amplitude(*angles, config=config))

    monkeypatch.setattr("ghe.optimization.solve_best_geometry", solver)
    script_response(R=7000)
    assert seen == [7000]
