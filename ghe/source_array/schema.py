from __future__ import annotations

import numpy as np

SOURCE_ARRAY_DTYPE = np.dtype(
    [
        ("source_id", np.int64),
        ("x_m", np.float64),
        ("y_m", np.float64),
        ("z_m", np.float64),
        ("distance_to_detector_m", np.float64),
        ("distance_offset_m", np.float64),
        ("propagation_compensation_s", np.float64),
        ("theta_src", np.float64),
        ("phi_src", np.float64),
        ("theta_rot", np.float64),
        ("phi_rot", np.float64),
        ("gw_phase_offset_rad", np.float64),
        ("rotor_phase_offset_rad", np.float64),
    ]
)

SOURCE_ARRAY_COLUMNS = SOURCE_ARRAY_DTYPE.names


def empty_source_array(size: int = 0) -> np.ndarray:
    return np.empty(size, dtype=SOURCE_ARRAY_DTYPE)


def ensure_source_array_dtype(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == SOURCE_ARRAY_DTYPE:
        return array
    return array.astype(SOURCE_ARRAY_DTYPE, copy=False)
