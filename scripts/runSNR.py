if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import subprocess
import sys
from pathlib import Path
import numpy as np

from ghe.config import (
    DATA_DIR,
    FREQS_FILE,
    MAGNITUDE_FILE,
    REPO_ROOT,
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
        raise ValueError(
            "Empty input. Use comma list (20,39.6,80) or range [10,100,10]."
        )
    return values


def run_python_file(script_path: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, str(script_path)], check=True, env=env, cwd=REPO_ROOT
    )


def calculate_snr_year_from_saved_data(
    test_mass: float,
    arm_length: float,
    noise_model: str | None = None,
) -> float:
    from ghe.config import DetectorConfig, NoiseConfig, SourceConfig
    from ghe.snr import calculate_snr

    source = SourceConfig(L=arm_length)
    return calculate_snr(
        MAGNITUDE_FILE,
        FREQS_FILE,
        source_config=source,
        detector_config=DetectorConfig(testmass=test_mass).with_source(source),
        noise_config=NoiseConfig(model=noise_model) if noise_model else None,
    )


def main() -> None:
    """Re-optimize for each L, with R derived from the configured R/L ratio.

    This sweep evaluates phasors directly: it needs neither shared temporary
    spectra nor subprocess environment overrides. Test mass changes noise only.
    """
    from dataclasses import asdict
    from ghe.config import SourceConfig, DetectorConfig, NoiseConfig
    from ghe.metric import calculate_response_phasor
    from ghe.optimization import optimize_best_geometry
    from ghe.snr import calculate_snr_from_phasor
    from ghe.artifacts import model_metadata, write_metadata

    parser = argparse.ArgumentParser(
        description="Ideal free-mass SNR sweep over arm length and test mass"
    )
    parser.add_argument("--masses", required=True)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "snr_year_table.csv")
    parser.add_argument("--noise-model", default=None)
    args = parser.parse_args()
    masses, lengths = parse_float_list(args.masses), parse_float_list(args.lengths)
    if not all(np.isfinite(x) and x > 0 for x in masses + lengths):
        raise ValueError("Masses and lengths must be finite and positive")
    noise = NoiseConfig(model=args.noise_model) if args.noise_model else NoiseConfig()
    rows, cases = [], []
    for length in lengths:
        source = SourceConfig(L=length, R=None)
        geometry = optimize_best_geometry(config=source)
        H = calculate_response_phasor(*geometry.angles, config=source)
        for mass in masses:
            detector = DetectorConfig(testmass=mass).with_source(source)
            snr = calculate_snr_from_phasor(
                H, source.gw_frequency_hz, detector_config=detector, noise_config=noise
            )
            rows.append(dict(arm_length_m=length, test_mass_kg=mass, snr_year=snr))
            cases.append(
                {
                    **model_metadata(source),
                    "geometry": asdict(geometry),
                    "detector_config": asdict(detector),
                    "noise_config": asdict(noise),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["arm_length_m", "test_mass_kg", "snr_year"]
        )
        writer.writeheader()
        writer.writerows(rows)
    write_metadata(
        args.output,
        {"cases": cases, "interpretation": "conditional ideal strain-noise proxy"},
    )
    print(f"Saved {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
