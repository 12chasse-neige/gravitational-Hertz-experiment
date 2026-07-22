from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def _all_divisors(n: int) -> list[int]:
    divisors = []
    limit = int(math.isqrt(n))
    for d in range(1, limit + 1):
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
    divisors.sort()
    return divisors


def choose_lattice_dimensions(num_sources: int) -> tuple[int, int, int]:
    """
    Choose a factorization ``(nx, ny, nz)`` that is as cube-like as possible.
    """

    if num_sources < 1:
        raise ValueError("num_sources must be positive.")

    divisors = _all_divisors(num_sources)
    best_dims: tuple[int, int, int] = (1, 1, num_sources)
    best_score: float = float(num_sources)

    cbrt = round(num_sources ** (1.0 / 3.0))

    for a in divisors:
        if a > cbrt * 2:
            break
        remainder_a = num_sources // a
        b_ideal = int(math.isqrt(remainder_a))
        for b in divisors:
            if b < a:
                continue
            if b > remainder_a:
                break
            if remainder_a % b != 0:
                continue
            c = remainder_a // b
            if b > c:
                continue
            score = float(c - a)
            if score < best_score:
                best_dims = (int(a), int(b), int(c))
                best_score = score

    if best_score < float(num_sources):
        return best_dims

    return (1, 1, int(num_sources))


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
