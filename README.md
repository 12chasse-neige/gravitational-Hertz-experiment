# Gravitational Hertz Experiment

Calculate the oscillating gravitational response of a two-hole rotor and coherent
source arrays in an ideal equal-arm, free-mass Michelson interferometer.
The production model retains all radial terms of the conserved mass quadrupole,
source-to-field retardation, and finite light-travel time along both arms.

The nearby-source response is mainly a driven Newtonian tidal interaction. Its
amplitude is not evidence of isolating emitted gravitational radiation. Reported
SNR uses the existing strain-referred noise models as a conditional calibration
proxy; suspension, feedback, cavity response, and source disturbances are outside
this calculation.

## Start here

- [Review guide](docs/production-integration-review.md): code ownership, equations,
  phase signs, compatibility changes, and validation evidence.
- [Field and response derivation](docs/near-field-analysis.md): equations used by
  the production kernel; its original diagnostic results are historical.
- [Current workflows](docs/current-workflows.md): small runs and large-run commands.
- [Validation record](docs/production-validation.json): fixed physics benchmarks
  and measured 1-, 10-, and 100-source examples.

## Structure

| Layer | Responsibility |
| --- | --- |
| `ghe/finite_distance.py` | Rotor STF moment, physical metric, acceleration, tidal curvature |
| `ghe/detector_response.py` | Converged arm integration and independent harmonic response |
| `ghe/metric.py` | Public geometry-to-phasor and time-domain response APIs |
| `ghe/optimization.py`, `ghe/source_array/` | Geometry search, placement, orientation, phase delays, array storage |
| `ghe/signal.py` | Coherent phasor summation and time-series synthesis |
| `ghe/spectrum.py`, `ghe/noise.py`, `ghe/snr.py` | FFT, retained noise models, conditional SNR |
| `ghe/artifacts.py` | Model provenance and saved-data compatibility |
| `scripts/`, `main.py` | Command-line entry points |
| `tests/support/` | Independent Newtonian and time-domain reference calculations |

## Environment and verification

```bash
conda create -n ghe python=3.13
conda activate ghe
python -m pip install -r requirements.txt pytest
python -m pytest -q
python scripts/validateResponse.py --output-dir runs/validation
```

## Small end-to-end run

```bash
python main.py --renew-source-array --source-array-num-sources 100 \
  --source-array-chunk-size 10 --source-array-format csv \
  --run-dir runs/example
python main.py --source-array-input runs/example/source_array.csv \
  --use-mono-approx --run-dir runs/example-mono
```

Both paths use the same complete response phasor. The first synthesizes a signal
and FFT; the second evaluates monochromatic SNR directly. For a bin-aligned tone,
they agree. `--use-mono-approx` is retained as a historical flag name; the source
model itself is monochromatic.

## Configuration and saved data

`configs/source.yaml` and `configs/detector.yaml` supply defaults. Explicit Python
arguments and supported environment/CLI overrides remain available. Source
`R=None` derives distance from the configured arm-length ratio; explicit distances
are preserved. To rescale arm length and derived placement together, use
`replace(config, L=new_length, R=None)`.

Old best-geometry caches are recomputed. Old arrays and spectra must be regenerated:
their orientations and phases belong to the removed radiative-only model.
CSV and paired NPY outputs have adjacent `.metadata.json` sidecars; NPZ and JSON
outputs embed model identity. Keep payloads and metadata together.

Use streaming CSV for very large arrays. Compressed NPZ is convenient for small
and medium arrays and is loaded as a whole. Chunk-anchor generation approximates
rotor orientation, while **every source phase uses the complete response**.
Under identical, aligned source responses and fixed noise, amplitude and SNR scale
as `N`; squared SNR scales as `N²`. Real spatial arrays need not have identical
source amplitudes.
