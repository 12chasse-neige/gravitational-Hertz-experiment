# Gravitational Hertz Experiment

Design a detuned signal-recycling interferometer for the oscillating gravity of
mechanical rotors, study the rotor **Qdot contribution**, and generate coherent
rotor arrays with a total-signal SNR estimate.

The field model retains all five derivative contributions (`Q`, `Qdot`, `Qddot`,
`Q3`, `Q4`), retardation, and finite light-travel time along both arms. The signal
response is an ideal free-mass Michelson; the detuned optical model supplies a
strain-noise proxy. A full suspended cavity response to a nearby rotor is not yet
modeled. A Qdot-phase sine projection can contain other terms at the same frequency.

## Setup

```bash
conda activate ghe
# For a new environment: conda create -n ghe python=3.13
python -m pip install -r requirements.txt
python main.py
```

Run commands from the repository root. `python main.py COMMAND --help` shows the
options for a workflow. Every retained script also runs directly.

## Workflows

| Command | Script | Purpose |
| --- | --- | --- |
| `quantum` | `scripts/quantumNoise.py` | Compare analytic and GWINC quantum noise; vary SRM transmission, SR cavity length, detuning target, squeezing |
| `noise-budget` | `scripts/plotNoiseBudget.py` | Plot analytic quantum, thermal and residual-gas contributions |
| `thermal-gas` | `scripts/estimateThermalGas.py` | Calculate thermal/gas spectra and conditional pressure constraints |
| `qdot` | `scripts/extractQdot.py` | Generate or load one rotor's signal; project onto the Qdot-phase sine and report model overlap |
| `qdot-sweep` | `scripts/sweepQdotArmLength.py` | Optimize modeled Qdot SNR over a bounded arm-length range |
| `array-generate` | `scripts/sourceArray.py` | Generate phase-aligned arrays in streaming CSV or NPZ |
| `array` | `scripts/runArray.py` | Generate or load an array; calculate total-signal SNR by phasor or FFT |
| `array-sweep` | `scripts/sweepArraySNR.py` | Compare total-signal SNR versus source count with ideal coherent scaling |
| `validate` | `scripts/validateResponse.py` | Check field/response agreement, Qdot projections and small array pipelines |

```bash
python main.py quantum --output runs/design/quantum.png
python main.py noise-budget --output runs/design/noise-budget.png
python main.py qdot --output-dir runs/qdot
python main.py qdot-sweep --min-length 4 --max-length 40 --points 5 \
  --output-dir runs/qdot-arm-length

python main.py array --renew-source-array --source-array-num-sources 10 \
  --source-array-chunk-size 5 --source-array-format npz \
  --monochromatic --noise-budget quantum-thermal-gas --run-dir runs/array-example
python main.py array --source-array-input runs/array-example/source_array.npz \
  --noise-budget quantum-thermal-gas --run-dir runs/array-fft
python main.py array-sweep --array-sizes 1,3,10 \
  --noise-budget quantum-thermal-gas --output runs/array-scaling.csv
python main.py validate --output-dir runs/validation
```

Qdot extraction defaults to **quantum + thermal + residual gas**. Array commands
retain their **quantum-only** default; use `--noise-budget quantum-thermal-gas`
for the combined budget, as in the examples above.
The array SNR uses the **complete gravitational response**, including every
derivative term; it does not measure an isolated array Qdot contribution.
`--monochromatic` evaluates the full response phasor directly; the former
`--use-mono-approx` spelling and existing `main.py --renew-source-array ...`
commands remain supported.

## Configuration and results

- `configs/detector.yaml`: active custom detector. Set arm length, optical power,
  SRM parameters, mirror geometry, squeezing, temperatures and gas pressures here.
  A null `Optics.SRM.Tunephase` requests automatic source-frequency tuning.
- `configs/source.yaml`: rotor dimensions, spin, source distance as `R/L`, sampling,
  noise band and array-generation defaults. The gravity signal frequency is twice
  the mechanical rotation frequency.
- `configs/aLIGO.yaml`, `configs/CE2silicon.yaml`, `configs/Voyager.yaml`: separate
  detector-design references; they are not replacements for the active YAML.
- `runs/`: reports, signals, arrays, plots and SNR records. Plotting scripts also
  retain their default publication destinations in `paper/figures/`.
- `data/`: compatible shared geometry cache and optional default array/spectrum
  outputs. Saved payloads and their `.metadata.json` sidecars belong together.

Defaults come from the actual YAML files, not the historical kilometre-scale
examples in older reports. Explicit `SourceConfig.R` stays fixed when replacing
`L`; use `replace(config, L=..., R=None)` to scale distance with arm length.
For large arrays use streaming CSV and check source/arm clearance. Old artifacts
with incompatible source parameters or model metadata must be regenerated.

## Code and scientific notes

| Package | Responsibility |
| --- | --- |
| `ghe/finite_distance.py`, `detector_response.py`, `metric.py` | Conserved rotor quadrupole and finite-arm detector response |
| `ghe/noise.py`, `phase.py`, `phase_convention.py` | Analytic detuned quantum noise and phase conventions |
| `ghe/thermal_gas.py`, `residual_gas.py` | GWINC thermal/gas calculations and gas comparisons |
| `ghe/quadrature.py`, `qdot_scaling.py` | Phase projections and Qdot arm-length study |
| `ghe/optimization.py`, `geometry.py`, `source_array/` | Geometry search, array placement, orientation, coherent phase delays and storage |
| `ghe/signal.py`, `spectrum.py`, `snr.py` | Single/array signal synthesis, FFT and coherent SNR |
| `ghe/config.py`, `paths.py`, `artifacts.py`, `validation.py` | Configuration, paths, provenance and reproducible validation |

See [current commands and assumptions](docs/current-workflows.md),
[model and validation guide](docs/production-integration-review.md), and
[finite-distance field derivation](docs/near-field-analysis.md).
[theoreticalDerivation.md](docs/theoreticalDerivation.md) preserves the original
proposal; its radiative-only treatment is historical.

The cleanup removes standalone legacy wrappers `bestPosition.py`,
`metricCalculate.py`, `fourier.py`, `noiseAnalysis.py`, `runSNR.py` and
`projectConfig.py`. Their relevant calculations remain in the workflows above
and in `ghe/`. Existing local runs, data, reference materials and manuscript files
are preserved. Historical numerical reports do not describe the active detector.
