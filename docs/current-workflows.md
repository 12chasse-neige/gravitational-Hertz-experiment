# Current calculation workflows

Activate the `ghe` environment, or use a Python installation with the dependencies
in `requirements.txt` plus pytest. Run commands from the repository root.

## Verification and review

```bash
python -m pytest -q
python scripts/validateResponse.py --output-dir runs/validation
```

The validation command fixes all rotor/source parameters and benchmark angles.
It records the active noise/detector configuration, checks harmonic-versus-curvature
response, and writes 1-, 10-, and 100-source arrays, spectra, and SNR comparisons.
See [the review guide](production-integration-review.md).

## Main array analysis

```bash
python main.py --renew-source-array --source-array-num-sources 100 \
  --source-array-chunk-size 10 --source-array-format csv --run-dir runs/example
python main.py --source-array-input runs/example/source_array.csv \
  --use-mono-approx --run-dir runs/example-mono
```

A run directory contains configuration, regenerated arrays when requested, signal
or phasor, spectrum when requested, SNR, and model metadata. Without `--run-dir`,
standard `data/` destinations remain available. The best-position cache remains
shared and is accepted only if its source configuration and model match.

## Individual tools

```bash
python scripts/metricCalculate.py -t 0 -ts 0.6 -ps 0.8 -tr 0.4 -pr 0.8 -R 6000
python scripts/bestPosition.py
python scripts/sourceArray.py --summary-only --num-sources 10
python scripts/fourier.py
python scripts/noiseAnalysis.py
python scripts/quantumNoise.py
python scripts/runSNR.py --masses "100,200" --lengths "2000,4000" --output runs/sweep.csv
python scripts/sweepArraySNR.py --array-sizes 1,10,100 --strategy chunk-center --output runs/array-sweep.csv
python scripts/armLengthScaling.py --frequency 600 --lengths "2000,4000"
```

Plotting scripts write their existing figure destinations unless given an output
option. The arm-length/mass sweep re-optimizes each length and derives placement
from the configured R/L ratio. Noise-only curves do not depend on the source
field replacement.

## Large runs (commands supplied; not executed during migration)

```bash
python main.py --renew-source-array --source-array-num-sources 10000000 \
  --source-array-chunk-size 1000 --source-array-format csv \
  --source-array-chunk-center-approximation --use-mono-approx \
  --run-dir runs/ten-million
```

Here the main CLI uses 1000 rows per I/O chunk and per orientation anchor. Every
row still receives its full detector-response phase. A large run may fail when
its source layout intersects a detector light path or a numerical integral cannot
converge. A ten-million-source runtime/memory guarantee is not inferred from the
small validation runs.

## Migration rules

Do not relabel old data by attaching new metadata. Regenerate geometry, then
arrays, then spectra/SNR. `main.py --renew-source-array` performs the array
regeneration and recomputes incompatible geometry automatically. Explicit old
array/spectrum inputs raise a regeneration error. Default input selection skips
an obsolete NPZ in favor of a compatible CSV.

The former `nearFieldResponse.py`, `verifyNearFieldNewtonian.py`, and
`singleSourceNearField.py` workflows are replaced by `validateResponse.py` and
scientific tests. Their historical reports remain in `near-field-analysis-results.json`
and `near-field-newtonian-check.json`; old FFT/SNR numbers are not acceptance targets.
