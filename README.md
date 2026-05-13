# Gravitational Hertz Experiment

This project models a gravitational-wave source based on a rotating hole array and evaluates its detectability with a LIGO-like interferometer. It computes the metric perturbation from a rotating mass distribution, finds the best source/rotor geometry, transforms the signal into frequency space, and estimates the signal-to-noise ratio (SNR) using quantum-noise models.

## Features
- Reusable `ghe/` package for geometry, metric response, optimization, source arrays, signal generation, spectra, noise, and SNR
- Compatibility scripts for the original command-line workflows
- Physics model for a rotating quadrupole source and detector response
- Optimization of best source sky location and rotor axis
- Fourier analysis of generated gravitational-wave signals
- SNR calculation using quantum noise PSD and saved signal spectra
- Detector noise modeling including squeezed-light quantum noise
- Parameter sweeps over test mass and arm length to build SNR tables
- CSV and `.npz` source-array storage

## Module Map
- `scripts/metricCalculate.py` -> `ghe.metric`, `ghe.geometry`
- `scripts/bestPosition.py` -> `ghe.optimization`
- `scripts/sourceArray.py` -> `ghe.source_array.*`
- `scripts/fourier.py` -> `ghe.spectrum`
- `scripts/quantumNoise.py` -> `ghe.noise`
- `scripts/noiseAnalysis.py` -> `ghe.snr`
- `main.py` -> `ghe.signal`, `ghe.spectrum`, `ghe.snr`

The scripts are now thin compatibility wrappers. New reusable code should go in `ghe/`.

## Installation

1. Create a Python environment (conda or venv recommended):
	 ```bash
	 conda create -n gravitationalHertzExperiment python=3.10
	 conda activate gravitationalHertzExperiment
	 # or use venv:
	 # python -m venv .venv && source .venv/bin/activate
	 ```
2. Install dependencies:
	 ```bash
	 pip install -r requirements.txt
	 ```

## Usage

- Run the main analysis:
	```bash
	python main.py
	```
- Detailed Chinese usage manual:
	```text
	docs/user-manual.md
	```
- Optional workflows:
	- Optimize geometry:
		```bash
		python scripts/bestPosition.py
		```
	- Generate FFT data and plots:
		```bash
		python scripts/fourier.py
		```
	- Compute SNR from saved signal spectra:
		```bash
		python scripts/noiseAnalysis.py
		```
	- Sweep arm length and test mass:
		```bash
		python scripts/runSNR.py --masses "20,39.6,80" --lengths "[1000,4000,1000]"
		```
	- Plot SNR results:
		```bash
		python scripts/plotSNRCurve.py --input data/snr_year_table.csv --output "images/SNR (3D).png"
		```
	- Preview a source array:
		```bash
		python scripts/sourceArray.py --summary-only --num-sources 1000
		```
	- Generate binary source-array data:
		```bash
		python scripts/sourceArray.py --num-sources 1000 --format npz
		```

## Output Files
- `data/freqs.npy`, `data/magnitude.npy`: single-source FFT results
- `data/total_freqs.npy`, `data/total_magnitude.npy`: source-array FFT results
- `data/bestPosition.txt`, `data/bestPosition.json`: optimized geometry
- `data/source_array_distribution.csv`: compatibility source-array table
- `data/source_array_distribution.npz`: preferred binary source-array table for generated small and medium arrays
- `data/snr_year_table.csv`: SNR sweep results
- `images/Signal.png`, `images/Fouriered Signal.png`: Plots
- `runs/<name>/`: optional reproducible run output created with `python main.py --run-dir runs/<name>`

## Notes
- The repository uses `numpy`, `matplotlib`, `scipy`, and `gwinc`.
- The main analysis depends on the source distribution and best-position data files in `data/`.
- Source-array write chunk size controls how many rows are generated before writing CSV output.
- Generation strategy is `exact` when every source rotor is optimized, `rigid` when the reference rotor axis is transported without per-source optimization, and `chunk_anchor` when one exact anchor is optimized per approximation group.
- `--approximation-chunk-size` controls the number of nearby sources represented by one chunk-anchor optimization.
- For more details, see `docs/theoreticalDerivation.md` and `docs/current-workflows.md`.
