from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def choose_lattice_dimensions(num_sources: int) -> tuple[int, int, int]:
    """
    Choose a factorization ``(nx, ny, nz)`` that is as cube-like as possible.
    """

    if num_sources < 1:
        raise ValueError("num_sources must be positive.")

    best_dims = (1, 1, num_sources)
    best_score = (num_sources - 1, num_sources - 1, num_sources)

    max_a = int(round(num_sources ** (1.0 / 3.0))) + 2
    for a in range(1, max_a + 1):
        if num_sources % a != 0:
            continue

        remainder = num_sources // a
        b = int(math.isqrt(remainder))
        while b >= a:
            if remainder % b == 0:
                c = remainder // b
                dims = tuple(sorted((a, b, c)))
                score = (dims[2] - dims[0], dims[1] - dims[0], dims[2])
                if score < best_score:
                    best_dims = dims
                    best_score = score
                break
            b -= 1

    return best_dims


def positions_for_index_range(
    start: int,
    stop: int,
    layout: tuple[int, int, int],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat source IDs into centered Cartesian lattice coordinates.
    """

    nx, ny, _ = layout
    indices = np.arange(start, stop, dtype=np.int64)
    ix = indices % nx
    iy = (indices // nx) % ny
    iz = indices // (nx * ny)

    positions = np.column_stack(
        [
            (ix - (nx - 1) / 2.0) * spacing,
            (iy - (ny - 1) / 2.0) * spacing,
            (iz - ((layout[2] - 1) / 2.0)) * spacing,
        ]
    )
    return indices, positions


def iter_position_chunks(
    num_sources: int,
    layout: tuple[int, int, int],
    spacing: float,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, num_sources, chunk_size):
        stop = min(start + chunk_size, num_sources)
        yield positions_for_index_range(start, stop, layout, spacing)
