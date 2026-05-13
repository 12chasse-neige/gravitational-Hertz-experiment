# Current Workflows

These are the compatibility commands preserved by the package refactor.

```bash
python main.py
python main.py --renew-source-array --source-array-num-sources 100 --source-array-chunk-size 10
python scripts/sourceArray.py --summary-only --num-sources 1000
python scripts/sourceArray.py --summary-only --num-sources 1000 --chunk-center-approximation
python scripts/fourier.py
python scripts/noiseAnalysis.py
python scripts/runSNR.py --masses "20,39.6,80" --lengths "[1000,4000,1000]"
```

Representative saved baseline values from the current `data/` artifacts:

```text
BEST_POSITION: 0.20227104, 5.96344729, 1.72327938, 0.01416633
single FFT peak: 600.0 Hz, magnitude 8.802176681728585e-40
total FFT peak: 600.0 Hz, magnitude 8.801788079624867e-37
single-source SNR/year: 3.359198130766582e-10
source-array SNR/year: 3.359049827510738e-07
```

Suggested regression tolerances:

- Geometry identities: `atol=1e-12`.
- Metric smoke checks: `rtol=1e-12` when comparing package and compatibility wrapper in the same run.
- Saved spectra and source-array CSV values: `rtol=1e-9` for small deterministic runs unless an intentional model change is documented.
