"""Optimize and persist a model-versioned single-source geometry."""

from __future__ import annotations
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghe.optimization import solve_best_geometry


def main():
    print(solve_best_geometry(recompute=True))


if __name__ == "__main__":
    main()
