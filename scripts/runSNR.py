if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

from ghe.config import (
    DATA_DIR,
    FREQS_FILE,
    INT_TIME,
    MAGNITUDE_FILE,
    REPO_ROOT,
    SCRIPTS_DIR,
    YEAR_SECONDS,
)


def parse_float_list(raw: str) -> list[float]:
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        parts = [p.strip() for p in s[1:-1].split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("Range format must be [start,stop,step], e.g. [10,100,10]")
        start, stop, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step == 0:
            raise ValueError("Step cannot be zero.")
        if (stop - start) * step < 0:
            raise ValueError("Step direction does not move from start toward stop.")

        values = []
        current = start
        eps = abs(step) * 1e-9 + 1e-12
        if step > 0:
            while current <= stop + eps:
                values.append(float(current))
                current += step
        else:
            while current >= stop - eps:
                values.append(float(current))
                current += step
        return values

    values = []
    for part in s.split(","):
        x = part.strip()
        if not x:
            continue
        values.append(float(x))
    if not values:
        raise ValueError("Empty input. Use comma list (20,39.6,80) or range [10,100,10].")
    return values


def run_python_file(script_path: Path, env: dict[str, str]) -> None:
    subprocess.run([sys.executable, str(script_path)], check=True, env=env, cwd=REPO_ROOT)


def calculate_snr_year_from_saved_data(test_mass: float, arm_length: float) -> float:
    from ghe.config import DetectorConfig
    from ghe.noise import squeeze_quantum_noise_with_varying_angle

    signal_magnitude = np.load(MAGNITUDE_FILE)
    freq = np.load(FREQS_FILE)

    valid_mask = (freq >= 1.0) & (freq <= 5000.0)
    freq_valid = freq[valid_mask]
    signal_magnitude_valid = signal_magnitude[valid_mask]
    detector_config = DetectorConfig(testmass=test_mass, length=arm_length)
    total_noise_psd = squeeze_quantum_noise_with_varying_angle(freq_valid, config=detector_config)

    integrand = (4.0 * signal_magnitude_valid**2) / total_noise_psd
    df = freq_valid[1] - freq_valid[0]
    snr_1s = np.sqrt(np.sum(integrand) * df)
    return float(snr_1s * np.sqrt(YEAR_SECONDS / INT_TIME))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep LIGO test mass and arm length, recompute best angles when arm length changes, "
            "and output an snr_year table."
        )
    )
    parser.add_argument(
        "--masses",
        required=True,
        help="Masses in kg: comma list (20,39.6,80) or range [10,100,10].",
    )
    parser.add_argument(
        "--lengths",
        required=True,
        help="Lengths in m: comma list (1000,2000,4000) or range [1000,4000,1000].",
    )
    parser.add_argument(
        "--output",
        default="data/snr_year_table.csv",
        help="Output CSV path. Default: data/snr_year_table.csv",
    )
    args = parser.parse_args()

    masses = parse_float_list(args.masses)
    lengths = parse_float_list(args.lengths)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for length in lengths:
        env_for_length = dict(os.environ)
        env_for_length["LIGO_ARM_LENGTH"] = str(length)
        print(f"\n[Length {length}] Running bestPosition.py ...")
        run_python_file(SCRIPTS_DIR / "bestPosition.py", env=env_for_length)
        print(f"[Length {length}] Running fourier.py ...")
        run_python_file(SCRIPTS_DIR / "fourier.py", env=env_for_length)

        for mass in masses:
            snr_year = calculate_snr_year_from_saved_data(test_mass=mass, arm_length=length)
            print(f"[Length {length}, Mass {mass}] snr_year = {snr_year:.6e}")
            results.append(
                {
                    "arm_length_m": length,
                    "test_mass_kg": mass,
                    "snr_year": snr_year,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arm_length_m", "test_mass_kg", "snr_year"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved table: {output_path}")


if __name__ == "__main__":
    main()
