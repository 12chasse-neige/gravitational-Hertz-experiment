# Gravitational Hertz Experiment

This project models a gravitational-wave source based on a rotating hole array and evaluates its detectability with a LIGO-like interferometer. It computes the metric perturbation from a rotating mass distribution, finds the best source/rotor geometry, transforms the signal into frequency space, and estimates the signal-to-noise ratio (SNR) using quantum-noise models.

## Features
- Physics model for a rotating quadrupole source and detector response
- Optimization of best source sky location and rotor axis (`bestPosition.py`)
- Fourier analysis of the generated gravitational-wave signal (`fourier.py`)
- SNR calculation using quantum noise PSD and saved signal spectra (`noiseAnalysis.py`)
- Detector noise modeling including squeezed-light quantum noise (`quantumNoise.py`)
- Parameter sweeps over test mass and arm length to build SNR tables (`runSNR.py`)
- Plotting utilities for SNR curves and tables (`plotSNRCurve.py`)
- Array-source generation for coherent many-source distributions (`sourceArray.py`)

## Main Scripts
- `main.py`: Calculates the total detector signal from the source array distribution, runs FFT, saves frequency and magnitude arrays, computes the 1-year SNR
- `scripts/bestPosition.py`: Optimizes source position and rotor orientation to maximize signal amplitude
- `scripts/fourier.py`: Computes FFT of the model signal, saves data and plots
- `scripts/metricCalculate.py`: Core physics engine for metric/tensor calculation
- `scripts/noiseAnalysis.py`: Loads FFT output, computes SNR using quantum noise PSD
- `scripts/quantumNoise.py`: Models quantum noise for a LIGO-like interferometer
- `scripts/runSNR.py`: Sweeps detector parameters, writes SNR table
- `scripts/plotSNRCurve.py`: Plots 3D SNR dependence on arm length and test mass
- `scripts/sourceArray.py`: Builds a multi-source array distribution

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
		python scripts/plotSNRCurve.py --input sourcedata/snr_year_table.csv --output images/SNR_3D.png
		```

## Output Files
- `sourcedata/freqs.npy`, `sourcedata/magnitude.npy`: FFT results
- `sourcedata/bestPosition.txt`: Optimized geometry
- `sourcedata/snr_year_table.csv`: SNR sweep results
- `images/Signal.png`, `images/Fouriered Signal.png`: Plots

## Notes
- The repository uses `numpy`, `matplotlib`, `scipy`, and `gwinc`.
- The main analysis depends on the source distribution and best-position data files in `sourcedata/`.
- For more details, see the scripts and the `reference/theoreticalDerivation.md` file.
