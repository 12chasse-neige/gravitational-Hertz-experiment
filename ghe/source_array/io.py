from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import SOURCE_ARRAY_COLUMNS, SOURCE_ARRAY_DTYPE, empty_source_array


def source_array_metadata_json(metadata: dict[str, Any] | None) -> str:
    return json.dumps(metadata or {}, sort_keys=True)


def write_csv_rows(output_path: str | Path, chunks: Iterable[np.ndarray]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_ARRAY_COLUMNS)
        for chunk in chunks:
            rows = zip(*(chunk[name] for name in SOURCE_ARRAY_COLUMNS))
            writer.writerows(rows)


def read_source_array_csv(input_path: str | Path) -> np.ndarray:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.stat().st_size == 0:
        return empty_source_array()
    data = np.genfromtxt(
        input_path,
        delimiter=",",
        names=True,
        dtype=SOURCE_ARRAY_DTYPE,
        encoding="utf-8",
    )
    return np.atleast_1d(data).astype(SOURCE_ARRAY_DTYPE, copy=False)


def write_source_array_npz_file(
    output_path: str | Path,
    source_array: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_array=source_array.astype(SOURCE_ARRAY_DTYPE, copy=False),
        metadata=np.array(source_array_metadata_json(metadata)),
    )


def read_source_array_npz(input_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    input_path = Path(input_path)
    with np.load(input_path, allow_pickle=False) as payload:
        source_array = payload["source_array"].astype(SOURCE_ARRAY_DTYPE, copy=False)
        raw_metadata = str(payload["metadata"]) if "metadata" in payload else "{}"
    return np.atleast_1d(source_array), json.loads(raw_metadata)


def read_source_array(input_path: str | Path) -> np.ndarray:
    path = Path(input_path)
    if path.suffix.lower() == ".npz":
        return read_source_array_npz(path)[0]
    return read_source_array_csv(path)
