"""Saved results may enter production only with matching model provenance."""

import numpy as np
import pytest

from ghe.artifacts import model_metadata, read_metadata
from ghe.config import SourceConfig
from ghe.optimization import solve_best_geometry, load_best_geometry
from ghe.signal import choose_source_array_input, compute_phasor_sum_from_file
from ghe.source_array.io import (
    write_csv_rows,
    write_source_array_npz_file,
    iter_source_array_file,
)
from ghe.source_array.schema import SOURCE_ARRAY_DTYPE
from ghe.spectrum import calculate_spectrum, save_spectrum_arrays
from ghe.snr import calculate_snr


def test_csv_streaming_and_npz_preserve_schema(tmp_path):
    rows = np.zeros(7, dtype=SOURCE_ARRAY_DTYPE)
    rows["source_id"] = np.arange(7)
    for suffix in ("csv", "npz"):
        path = tmp_path / f"array.{suffix}"
        if suffix == "csv":
            write_csv_rows(path, [rows[:3], rows[3:]], metadata=model_metadata())
        else:
            write_source_array_npz_file(path, rows, metadata=model_metadata())
        chunks = list(iter_source_array_file(path, 3))
        assert [len(chunk) for chunk in chunks] == [3, 3, 1]
        np.testing.assert_array_equal(np.concatenate(chunks), rows)


def test_old_npz_does_not_shadow_new_csv(tmp_path):
    npz, csv = tmp_path / "array.npz", tmp_path / "array.csv"
    rows = np.zeros(0, dtype=SOURCE_ARRAY_DTYPE)
    write_source_array_npz_file(npz, rows)
    write_csv_rows(csv, [rows], metadata=model_metadata())
    assert choose_source_array_input(npz, csv) == csv
    with pytest.raises(ValueError, match="regenerate"):
        compute_phasor_sum_from_file(npz)
    with pytest.raises(ValueError, match="regenerate"):
        list(iter_source_array_file(csv, config=SourceConfig(R=8000)))


def test_unversioned_geometry_is_recomputed_in_local_cache(tmp_path):
    path = tmp_path / "best.txt"
    path.write_text("BEST_POSITION: 1, 0, 1, 0\n")
    assert load_best_geometry(path) is None
    solved = solve_best_geometry(path=path)
    loaded = load_best_geometry(path)
    np.testing.assert_allclose(loaded.angles, solved.angles, rtol=0, atol=1e-8)
    assert read_metadata(path) == model_metadata()
    assert load_best_geometry(path, config=SourceConfig(R=8000)) is None


def test_spectrum_rejects_missing_or_mixed_payloads(tmp_path):
    spectrum = calculate_spectrum(
        np.cos(2 * np.pi * 600 * np.arange(1200) / 120000), sampling_rate=120000
    )
    magnitude, frequency = tmp_path / "m.npy", tmp_path / "f.npy"
    np.save(magnitude, spectrum.magnitude)
    np.save(frequency, spectrum.freqs)
    with pytest.raises(ValueError, match="metadata"):
        calculate_snr(magnitude, frequency)
    save_spectrum_arrays(spectrum, magnitude, frequency)
    assert np.isfinite(calculate_snr(magnitude, frequency))
    np.save(magnitude, spectrum.magnitude * 2)
    with pytest.raises(ValueError, match="payload"):
        calculate_snr(magnitude, frequency)


def test_calculation_reader_rejects_reordered_csv_header(tmp_path):
    from ghe.source_array.io import read_source_array
    from ghe.artifacts import write_metadata

    path = tmp_path / "wrong.csv"
    path.write_text("theta_src,distance_to_detector_m\n0.6,6000\n")
    write_metadata(path, model_metadata())
    with pytest.raises(ValueError, match="schema"):
        read_source_array(path)
