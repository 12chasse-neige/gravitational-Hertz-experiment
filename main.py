"""
This script gives out the total source array's impact on the detector, giving out the
total SNR of the signal (Integrate Time = 1 year).
"""
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scripts.metricCalculate import ExperimentConfig, calculate_metric_response
from scripts.noiseAnalysis import calculate_snr
from scripts.fourier import fourier, INT_TIME, NUM

_DATA_DIR = Path(__file__).resolve().parent / "data"
_SOURCE_ARRAY_DISTRIBUTION = _DATA_DIR / "source_array_distribution.csv"
_TOTAL_MAGNITUDE = _DATA_DIR / "total_magnitude.npy"
_TOTAL_FREQS = _DATA_DIR / "total_freqs.npy"

def read_csv_line(ID: int) -> np.ndarray:
    """
    Get the source with ID's parameters from the source distribution file.
    """
    with open(_SOURCE_ARRAY_DISTRIBUTION, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if ID + 1 >= len(rows):
        raise IndexError(f"Source ID {ID} is out of range ({len(rows) - 1} sources available)")

    return np.array([float(value) for value in rows[ID + 1]], dtype=float)
    

def get_single_source_metric_response(t: float, ID: int) -> float:
    """
    Return a single source's effect on the detector using the parameters from
    "data/source_array_distribution.csv".
    """
    parameter = read_csv_line(ID)

    theta_src = float(parameter[7])
    phi_src = float(parameter[8])
    theta_rot = float(parameter[9])
    phi_rot = float(parameter[10])
    distance = float(parameter[4])

    phase_offset = float(parameter[12])
    time_offset = phase_offset / ExperimentConfig.omega

    response = calculate_metric_response(
        t - time_offset,
        theta_src,
        phi_src,
        theta_rot,
        phi_rot,
        R=distance,
    )
    return response

def get_source_num(input_path: Path = _SOURCE_ARRAY_DISTRIBUTION) -> int:
    """
    Return total source number in the paranmeters file.
    """
    with open(input_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    return len(rows)

def get_total_signal(t: float) -> float:
    """
    Read all sources' parameters from file "data/source_array_distribution.csv" and 
    return the total signal induced by the sources.
    """
    num = get_source_num()
    total_signal = 0.0

    for ID in range(num - 1):
        total_signal += get_single_source_metric_response(t, ID)

    return total_signal

def main():
    t = np.linspace(0, INT_TIME, NUM, endpoint=False)
    h_values = [get_total_signal(ti) for ti in t]
    h_values = np.array(h_values)
    _, fft_magnitude, freqs = fourier(h_values)
    np.save(_TOTAL_FREQS, freqs)
    np.save(_TOTAL_MAGNITUDE, fft_magnitude)

    calculate_snr(_TOTAL_MAGNITUDE, _TOTAL_FREQS)
 

if __name__ == "__main__":
    main()
    
    
