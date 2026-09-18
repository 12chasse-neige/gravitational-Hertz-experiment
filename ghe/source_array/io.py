"""Source-table storage and validated streaming input.

Low-level CSV/NPZ readers support historical inspection. read_source_array and
iter_source_array_file are calculation entry points and require model metadata.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import SOURCE_ARRAY_COLUMNS, SOURCE_ARRAY_DTYPE, empty_source_array


def source_array_metadata_json(metadata: dict[str, Any] | None) -> str:
    return json.dumps(metadata or {}, sort_keys=True)


def write_csv_rows(
    output_path: str | Path, chunks: Iterable[np.ndarray], *, metadata=None
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Invalidate the prior sidecar BEFORE truncating the table. An interrupted
    # regeneration must not leave a partial file carrying an old valid identity.
    from ghe.artifacts import sidecar_path

    sidecar_path(output_path).unlink(missing_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_ARRAY_COLUMNS)
        for chunk in chunks:
            rows = zip(*(chunk[name] for name in SOURCE_ARRAY_COLUMNS))
            writer.writerows(rows)
    if metadata is not None:
        from ghe.artifacts import write_metadata

        write_metadata(output_path, metadata)


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
        source_array = payload["source_array"]
        if source_array.dtype != SOURCE_ARRAY_DTYPE or source_array.ndim != 1:
            raise ValueError("Source-array NPZ schema mismatch; regenerate the array")
        raw_metadata = str(payload["metadata"]) if "metadata" in payload else "{}"
    return np.atleast_1d(source_array), json.loads(raw_metadata)


def read_source_array(input_path: str | Path, *, config=None) -> np.ndarray:
    validate_source_array_file(input_path, config=config)
    path = Path(input_path)
    if path.suffix.lower() == ".npz":
        return read_source_array_npz(path)[0]
    chunks = list(iter_source_array_file(path, config=config))
    return np.concatenate(chunks) if chunks else empty_source_array()


def validate_source_array_file(input_path, *, config=None):
    """Validate provenance without materializing a potentially huge CSV."""
    from ghe.artifacts import read_metadata, validate_metadata

    path = Path(input_path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            metadata = (
                json.loads(str(payload["metadata"])) if "metadata" in payload else {}
            )
    else:
        metadata = read_metadata(path)
    validate_metadata(metadata, config)
    return metadata


def iter_source_array_file(input_path, chunk_size=10_000, *, config=None):
    """Validated bounded-memory CSV reader; NPZ loads its compressed table once.

    CSV column names must match the preserved public schema. Never cast a
    mismatched header positionally, since that could turn a distance into phase.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    validate_source_array_file(input_path, config=config)
    path = Path(input_path)
    if path.suffix.lower() == ".npz":
        rows, _ = read_source_array_npz(path)
        for start in range(0, len(rows), chunk_size):
            yield rows[start : start + chunk_size]
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader, ())) != SOURCE_ARRAY_COLUMNS:
            raise ValueError("Source-array CSV schema mismatch; regenerate the array")
        chunk = []
        for row in reader:
            chunk.append(tuple(row))
            if len(chunk) == chunk_size:
                yield np.array(chunk, dtype=SOURCE_ARRAY_DTYPE)
                chunk = []
        if chunk:
            yield np.array(chunk, dtype=SOURCE_ARRAY_DTYPE)
