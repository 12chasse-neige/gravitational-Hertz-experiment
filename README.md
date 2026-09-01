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
- `scr/metricCalculate.py` -> `ghe.metric`, `ghe.geometry`
- `scr/singleSourceNearField.py` -> `ghe.near_field`
- `scr/bestPosition.py` -> `ghe.optimization`
- `scr/sourceArray.py` -> `ghe.source_array.*`
- `scr/fourier.py` -> `ghe.spectrum`
- `scr/quantumNoise.py` -> `ghe.noise`
- `scr/noiseAnalysis.py` -> `ghe.snr`
- `main.py` -> `ghe.signal`, `ghe.spectrum`, `ghe.snr`

The scripts are now thin compatibility wrappers. New reusable code should go in `ghe/`.

## Configuration

Project defaults are stored in two structured YAML files:

- `configs/detector.yaml`: GWINC-compatible detector parameter names and units
- `configs/source.yaml`: source physics, constants, sampling, noise, and source-array defaults

`ghe/config.py` reads both files when a process starts and uses their values as
the dataclass defaults. Existing environment variables remain optional overrides,
and explicit Python arguments or CLI flags take precedence for one run. For
example, these commands do not modify the YAML files:

```bash
python scr/quantumNoise.py --length-sr 70 --t-srm 0.2
python scr/quantumNoise.py --detector-config /path/to/gwinc/ifo.yaml
python scr/sourceArray.py --num-sources 1000 --chunk-size 100 --spacing 8
python scr/armLengthScaling.py --frequency 800 --squeeze-db 8
```

## Installation

1. Create a Python environment (conda or venv recommended):
	 ```bash
	 conda create -n gravitational-Hertz-experiment python=3.10
	 conda activate gravitational-Hertz-experiment
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
- Optional workflows:
	- Optimize geometry:
		```bash
		python scr/bestPosition.py
		```
	- Generate FFT data and plots only:
		```bash
		python scr/fourier.py
		```
	- Directly integrate the single-rotor near-field metric tensor at the detector vertex:
		```bash
		python scr/singleSourceNearField.py
		```
	- Regenerate the signal spectrum from current parameters and compute SNR:
		```bash
		python scr/noiseAnalysis.py
		```
	- Compare gwinc, previous, and detuned signal-recycling noise curves:
		```bash
		python scr/quantumNoise.py
		```
	- Sweep arm length and test mass:
		```bash
		python scr/runSNR.py --masses "20,39.6,80" --lengths "[1000,4000,1000]"
		```
	- Plot SNR results:
		```bash
		python scr/plotSNRCurve.py --input data/snr_year_table.csv --output "paper/figures/SNR (3D).png"
		```
	- Preview a source array:
		```bash
		python scr/sourceArray.py --summary-only --num-sources 1000
		```
	- Generate binary source-array data:
		```bash
		python scr/sourceArray.py --num-sources 1000 --format npz
		```

## Output Files
- `data/freqs.npy`, `data/magnitude.npy`: single-source FFT results
- `data/total_freqs.npy`, `data/total_magnitude.npy`: source-array FFT results
- `data/bestPosition.txt`, `data/bestPosition.json`: optimized geometry
- `data/single_source_metric.json`: direct single-rotor metric tensor at the detector vertex
- `data/source_array_distribution.csv`: compatibility source-array table
- `data/source_array_distribution.npz`: preferred binary source-array table for generated small and medium arrays
- `data/snr_year_table.csv`: SNR sweep results
- `paper/figures/`: Generated plots, including the figures embedded in the manuscript
- `runs/<name>/`: optional reproducible run output created with `python main.py --run-dir runs/<name>`

## Notes
- The repository uses `numpy`, `matplotlib`, `scipy`, and `gwinc`.
- The main analysis depends on the source distribution and best-position data files in `data/`.
- Source-array write chunk size controls how many rows are generated before writing CSV output.
- Generation strategy is `exact` when every source rotor is optimized, `rigid` when the reference rotor axis is transported without per-source optimization, and `chunk_anchor` when one exact anchor is optimized per approximation group.
- `--approximation-chunk-size` controls the number of nearby sources represented by one chunk-anchor optimization.
- For more details, see `docs/theoreticalDerivation.md` and `docs/current-workflows.md`.
