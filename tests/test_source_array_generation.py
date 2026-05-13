from __future__ import annotations

import numpy as np

from ghe.source_array.generation import build_array_context, build_chunk
from ghe.source_array.io import read_source_array_npz, write_source_array_npz_file
from ghe.source_array.layout import choose_lattice_dimensions
from ghe.source_array.schema import SOURCE_ARRAY_DTYPE


def test_lattice_dimensions_are_cube_like_for_perfect_cube() -> None:
    assert choose_lattice_dimensions(27) == (3, 3, 3)


def test_source_array_chunk_schema_smoke() -> None:
    context = build_array_context(num_sources=1, optimize_each_source=False)
    chunk = build_chunk(context, 0, 1)
    assert chunk.dtype == SOURCE_ARRAY_DTYPE
    assert chunk["source_id"][0] == 0
    assert np.isfinite(chunk["distance_to_detector_m"][0])


def test_npz_round_trip(tmp_path) -> None:
    array = np.zeros(2, dtype=SOURCE_ARRAY_DTYPE)
    array["source_id"] = [0, 1]
    output_path = tmp_path / "source_array.npz"
    write_source_array_npz_file(output_path, array, metadata={"generation_strategy": "test"})

    loaded, metadata = read_source_array_npz(output_path)
    assert loaded.dtype == SOURCE_ARRAY_DTYPE
    assert loaded["source_id"].tolist() == [0, 1]
    assert metadata["generation_strategy"] == "test"
