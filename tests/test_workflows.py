"""Exercise real entry points, including file handoff and configuration metadata."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_script(*arguments):
    completed = subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


@pytest.mark.parametrize("storage", ["csv", "npz"])
def test_main_fft_and_direct_phasor_paths(tmp_path, storage):
    fft_dir = tmp_path / "fft"
    run_script(
        "main.py",
        "--renew-source-array",
        "--source-array-num-sources",
        4,
        "--source-array-chunk-size",
        2,
        "--source-array-format",
        storage,
        "--run-dir",
        fft_dir,
    )
    source_path = fft_dir / f"source_array.{storage}"
    assert source_path.is_file()
    mono_dir = tmp_path / "mono"
    run_script(
        "main.py",
        "--source-array-input",
        source_path,
        "--use-mono-approx",
        "--run-dir",
        mono_dir,
    )
    fft = json.loads((fft_dir / "snr.json").read_text())
    mono = json.loads((mono_dir / "snr.json").read_text())
    np.testing.assert_allclose(fft["snr_year"], mono["snr_year"], rtol=1e-12, atol=0)
    assert fft["model_version"] == mono["model_version"]
    assert "metadata" in json.loads((fft_dir / "config.json").read_text())


def test_array_sweep_and_restored_mass_length_sweep(tmp_path):
    run_script(
        "scripts/sweepArraySNR.py",
        "--array-sizes",
        "1,10",
        "--strategy",
        "rigid",
        "--output",
        tmp_path / "arrays.csv",
    )
    run_script(
        "scripts/runSNR.py",
        "--masses",
        "100,200",
        "--lengths",
        "4000",
        "--output",
        tmp_path / "lengths.csv",
    )
    for name in ("arrays", "lengths"):
        assert (tmp_path / f"{name}.csv.metadata.json").is_file()


def test_validation_command(tmp_path):
    run_script("scripts/validateResponse.py", "--output-dir", tmp_path)
    report = json.loads((tmp_path / "validation.json").read_text())
    assert [row["num_sources"] for row in report["small_arrays"]] == [1, 10, 100]
    assert (
        max(row["dual_response_relative_error"] for row in report["benchmarks"]) < 1e-11
    )
