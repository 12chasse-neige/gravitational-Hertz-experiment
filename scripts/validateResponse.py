"""Run fixed physics benchmarks and small end-to-end array validations."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghe.validation import run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/validation"))
    args = parser.parse_args()
    print(f"Validated results: {run_validation(args.output_dir)}")


if __name__ == "__main__":
    main()
