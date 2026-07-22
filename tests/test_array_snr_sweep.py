from __future__ import annotations

import csv

import pytest

from scr.sweepArraySNR import parse_array_sizes, sweep_array_snr, write_results


def test_parse_array_sizes_requires_one_as_reference() -> None:
    assert parse_array_sizes("1, 10,100") == [1, 10, 100]
    with pytest.raises(Exception, match="first array size must be 1"):
        parse_array_sizes("10,100")


def test_sweep_uses_full_precision_single_source_reference(monkeypatch) -> None:
    numerical_values = {1: 1.234567890123456e-7, 10: 1.234567890123400e-6}

    def fake_calculate(num_sources, **kwargs):
        return numerical_values[num_sources], float(num_sources)

    monkeypatch.setattr("scr.sweepArraySNR.calculate_array_snr", fake_calculate)
    rows = sweep_array_snr(
        [1, 10],
        strategy="rigid",
        generation_chunk_size=10,
        approximation_chunk_size=10,
        spacing=None,
    )

    assert rows[1]["ideal_snr_year"] == 10 * numerical_values[1]
    expected = (numerical_values[10] - 10 * numerical_values[1]) / (10 * numerical_values[1])
    assert rows[1]["relative_deviation"] == expected


def test_csv_preserves_seventeen_digit_scientific_notation(tmp_path) -> None:
    output_path = tmp_path / "sweep.csv"
    rows = [
        {
            "num_sources": 1,
            "numerical_snr_year": 1.2345678901234567e-7,
            "ideal_snr_year": 1.2345678901234567e-7,
            "relative_deviation": 0.0,
            "phasor_magnitude": 9.876543210987654e-30,
        }
    ]
    write_results(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as input_file:
        saved = next(csv.DictReader(input_file))
    assert saved["numerical_snr_year"] == "1.23456789012345658e-07"
    assert saved["relative_deviation"] == "0.00000000000000000e+00"
