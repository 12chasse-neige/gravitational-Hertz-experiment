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

## Thermal and residual-gas estimates

Run one complete detector YAML per report. Each run reads `configs/source.yaml`
for the common logarithmic frequency grid, exact target frequency, and paper
allowance. Give each configuration its own output directory:

```bash
python scripts/estimateThermalGas.py --config configs/detector.yaml \
  --output-dir runs/thermal-gas-hybrid
python scripts/estimateThermalGas.py --config configs/aLIGO.yaml \
  --output-dir runs/thermal-gas-aligo
python scripts/estimateThermalGas.py --config configs/CE2silicon.yaml \
  --output-dir runs/thermal-gas-ce2-reference
```

`configs/detector.yaml` is the default CE2 silicon hybrid: CE2's 40 km arms,
2 µm laser, silicon optics and suspension, 123 K lower stages, gas partial
pressures, and 15 dB filter-cavity squeezing. It retains GHE's SRM
transmittance `1e-5`, cavity length 55 m, and automatic detuning at the source
GW frequency. The calculated source placement follows `R/L=1.5`, giving
`R=60 km`. This custom signal-recycling design is not the official CE2
sensitivity. `configs/CE2silicon.yaml` is the unchanged CE2 optical reference
apart from the `+inherit` budget selector; `configs/aLIGO.yaml` remains a
separate historical reference. The old 4 km custom detector is discarded.

The command selects GWINC's CE2silicon or aLIGO budget from `+inherit` (the
GWINC default is aLIGO). It calculates only suspension, coating, substrate, and
residual-gas terms. CE2's nested `ITMThermoRefractive` term is included; aLIGO
has no such term, so its matching CSV column is zero. A resolved numerical SR
phase is passed to GWINC when `Optics.SRM.Tunephase` is `null`. The squeezer
from the selected detector YAML is retained. No quantum, seismic, or
Newtonian terms are included in these totals, so the plotted curves are not
full detector sensitivity curves.

Each directory receives `thermal_gas.csv` (frequency, PSD, and ASD columns),
`thermal_gas.json` (exact target row, selected budget class, effective detector
including resolved SR phase, temperatures, partial pressures, warnings, and
conditional gas ceilings), and `thermal_gas.png`. The frequency rows and column
names match across the three runs. Use `--no-plot` when only CSV and JSON are
needed; it removes a stale PNG from the chosen output directory. The optional
`--source-config` selects another source/estimation YAML. GWINC's `.nan`
auto-compute sentinels appear as strings in strict JSON metadata.

### Parameters and interpretation

Edit GWINC-style fields directly in the selected detector YAML:

- `Infrastructure.ResidualGas.H2`, `N2`, `H2O`, and `O2` contain
  `BeamtubePressure` and `ChamberPressure` in Pa, molecular `mass` in kg,
  and `polarizability` volume in m³. Setting one pressure to zero removes
  that species at that location.
- `Infrastructure.Temp` is the beam-tube gas temperature (293 K for CE2).
  `Suspension.Stage[0].Temp` is the test-mass chamber gas temperature used
  by the notebook damping model (123 K for CE2). Other suspension stage
  temperatures and `Materials.Substrate.Temp` affect GWINC thermal terms.
- `Infrastructure.ResidualGas.SqueezedFilm` may be `{}` for an infinite
  reservoir. The notebook model also accepts `{gap: 0.005}` for a 5 mm gap,
  mirror-specific `ITM`/`ETM` gaps, or `ExcessDamping` with `DiffusionTime`.
  The default CE2 configuration does not introduce an elevated-pressure or
  narrow-gap case.
- `Optics.Curvature.ITM/ETM`, `Infrastructure.Length`, and
  `Laser.Wavelength` determine the Gaussian-beam geometry for gas scattering.

In `configs/source.yaml`, `Noise.FrequencyBand` sets the shared range and
`Noise.Estimation.FrequencySamples` sets the logarithmic sample count (1000 by
default). `Noise.Estimation.TargetFrequency: null` selects
`Source.Rotor.AngularVelocity / pi`, currently exactly 600 Hz. An explicit
value inserts that frequency into the grid but does not change automatic SR
detuning. `Noise.Estimation.GasAllowanceASD` remains the paper's conditional
`4.50e-27 Hz^-1/2` reference; it is not a CE2 allocation or a derived CE2
requirement. `Noise.Model` and `Noise.SqueezingDB` still control the project's
analytic SNR model and do not overwrite the detector YAML's GWINC squeezer.

Thermal components are summed in PSD, then converted to ASD. GWINC gas
scattering and damping sum over the four species exactly once. `ThermalPlusGas`
is their PSD sum with `ThermalTotal`; it excludes all other detector noises.
The notebook optical-pathlength and free-mass damping spectra are labeled
comparisons and are never added again. GWINC damping uses its suspension
susceptibility; the notebook damping estimate uses four free masses and is
primarily useful above suspension resonances. The fixed-mixture pressure
ceilings independently assign the *entire* allowance to each mechanism, so
they cannot all be met by spending the same allowance simultaneously.

At 600 Hz with the default CE2 hybrid, the expected thermal ASD is about
`1.24e-26 Hz^-1/2` and GWINC residual-gas ASD about `2.33e-26 Hz^-1/2`.
The conditional notebook beam-tube pressure ceiling is about `1.98e-9 Pa`,
compared with the configured nominal total `5.33e-8 Pa`. The historical
`paper/thermalConstraint/sweepTemperature.py` retains its explicit aLIGO
reference and is not a CE2 temperature requirement. `noiseAnalysis.py` and
other SNR commands still use `ghe/noise.py`'s analytic quantum model; they do
not read `thermal_gas.csv`.

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
