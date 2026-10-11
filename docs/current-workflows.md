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

## Single-source Qdot contribution and quadrature projection

```bash
python scripts/extractQdot.py
python scripts/extractQdot.py --angles 0.6 0.8 0.4 0.8 \
  --distance 6000 --rotor-delay-rad 0.2 --duration 0.01 \
  --sample-rate 120000 --output-dir runs/qdot-example
python scripts/extractQdot.py --angles 0.6 0.8 0.4 0.8 \
  --distance 6000 --rotor-delay-rad 0.2 \
  --signal-input runs/qdot-example/signals.npz \
  --output-dir runs/qdot-loaded
```

With no arguments, the script generates a single-source signal using the active
source and sampling configuration. It reads a compatible best-position cache,
or optimizes the geometry without saving over that cache. `--angles` specifies
`theta_src phi_src theta_rot phi_rot` in detector-frame radians. Distance defaults
to the current source configuration's `R/L` placement; rotor delay defaults to
zero. A positive mechanical delay `delta` multiplies every component phasor by
`exp(2i delta)`. The signal frequency is `Omega=2*omega`, from `SourceConfig`.

A loaded NPZ must contain real one-dimensional `time_s` and `signal` arrays with
matching lengths and at least three finite samples. Times must strictly increase;
their original time zero is preserved. For example, an external single-source
record can be saved with `np.savez("input.npz", time_s=times, signal=values)`.
Loaded data require explicit `--angles`; supply the correct `--distance` and
`--rotor-delay-rad` when they differ from the defaults. Model metadata, when
present, must match; NPZ files produced by this script also check geometry and
rotor delay. The existing array-run `spectrum.npz` is not a single-source input.
`--duration` and `--sample-rate` apply only to generated records.

The field decomposition groups explicit derivatives at retarded time `u=t-r/c`:
`Q`, `Qdot`, `Qddot`, `Q3`, `Q4`, scaling respectively as
`G Q/r^5`, `G Qdot/(c r^4)`, through `G Q4/(c^4 r)`. Every piece retains
`exp(ikr)` and passes through the same finite-arm, ideal free-mass Michelson
response. Thus the output `qdot_model` is the **model-predicted Qdot contribution
to the detector strain**, not the source's quadrupole derivative in kg m²/s.
The other four modeled terms are summed into `remaining_model`.

For uniform whole-period records, sine/cosine orthogonality gives the peak
coefficients `a=2*mean(signal*cos(Omega*time_s))` and
`b=2*mean(signal*sin(Omega*time_s))`. Otherwise the script jointly fits sine,
cosine and a constant to account for finite-window/DC leakage. Nonuniform records
use equal sample weights. In the project's `exp(-i Omega t)` convention the
recovered phasor is `a+i*b`; the reconstruction includes the fitted DC offset.
Each modeled component is also projected to verify its phasor and phase sign.

**The total sine quadrature does not isolate Qdot.** All five physical terms
share the same frequency, and a rotating quadrupole contains both sine and cosine
components. This script assumes the source parameters, geometry, timing and rotor
phase are known. It compares that physical model with measured quadratures; it
does not infer five independent amplitudes from two Fourier coefficients.

By default, `runs/qdot-extraction/` receives:

- `report.json`: configuration, geometry, timing, modeled phasors, recovered
  quadratures, fit residuals, input/model agreement and integration accuracy.
- `signals.npz` and `signals.csv`: original times and input signal, model total,
  Qdot prediction, remaining terms, fitted signal, reconstructed cosine/sine
  quadratures, residuals, and all five component signals and fits. The NPZ also
  contains model and geometry metadata and can be used as a subsequent input.
- `comparison.png`: input/fit/model comparison, Qdot prediction and its
  projection, and the remaining modeled signal and input/model residual.

All amplitudes are peak dimensionless strains, not ASD or FFT magnitudes. The
individual components are additive diagnostics of one detector observable.
Array processing and suspended/cavity detector dynamics are outside this script.

## Qdot SNR versus arm length

```bash
python scripts/sweepQdotArmLength.py
python scripts/sweepQdotArmLength.py --min-length 100 --max-length 59000 \
  --points 121 --observation-years 1 --output-dir runs/qdot-arm-length
python scripts/sweepQdotArmLength.py --min-length 100 --max-length 40000 \
  --angles 1.5707963267948966 1.5707963267948966 0 0
```

The reusable kernel is `ghe.qdot_scaling.sweep_qdot_arm_lengths(lengths_m, ...)`;
`QdotArmLengthStudy.evaluate(arm_length_m)` evaluates one length and caches its
result. By default, a deterministic multistart search optimizes source direction
and rotor-axis orientation for **the Qdot component** at each arm length.
`--angles` instead keeps those four angles fixed. Neither mode overwrites the
shared geometry cache or the input configuration files.

This study preserves the configured `R/L` ratio (currently 1.5), the 30 km mirror
curvatures, and all rotor, mirror, optical-power, squeezing, pressure and
temperature parameters. Length-dependent beam sizes and noise are recalculated.
The analytic SR detuning is retuned to the rotor's signal frequency at each
length. The statistic for one rotor is

$$
\rho_{\dot Q}=|H_{\dot Q}|\sqrt{\frac{T}{S_{\rm quantum}^{\rm analytic}
  +S_{\rm thermal}^{\rm GWINC}+S_{\rm gas}^{\rm GWINC}}}.
$$

These are one-sided PSDs, added once. Notebook gas comparisons and the GWINC
quantum curve are not included. The default coherent observation is one year;
changing its duration scales SNR as the square root without changing the optimum.
The thermal/gas API accepts `arm_length_m` and `frequency_hz` overrides in memory,
allowing exact-frequency evaluation without recomputing an entire plotted spectrum.

The default logarithmic search is 100 m to 59 km. Interior sampled maxima are
refined in log arm length, and both endpoints participate in the comparison.
The report distinguishes an interior maximum from a maximum at the chosen lower
or upper bound. It does not turn a lower-bound result into a claim of an
unconstrained optimum. Shortening the arms also moves the source closer; this is
a coupled detector/source-placement study, not a fixed-distance detector sweep.

Expected optical exclusions are recorded: unstable cavities, the exactly
confocal configuration that GWINC cannot evaluate, and aperture clipping greater
than the configured mirror-loss budget. For the current equal 30 km curvatures,
the exact 30 km case is excluded. Excess beam clipping also excludes some long
arms before the formal 60 km stability boundary. Other numerical failures abort
the calculation instead of silently disappearing from the results.

`runs/qdot-arm-length/` contains `qdot_arm_length.csv`, `qdot_arm_length.json`, and
`qdot_arm_length.png`. The JSON preserves source and detector parameters, input
hashes, rejected cases, refined peaks, component phasors, noise PSDs, optimizer
and integration diagnostics, closest source-arm clearance and `qdot_over_q_amplitude`.
The plot shows SNR, Qdot peak strain and the three noise contributions.

With the current configuration, the best found point is the 100 m lower bound,
with `R=150 m` and one-year `rho_Qdot` about 22.08. It assumes exact subtraction of
all other derivative terms and ideal free-mass response. At short separations,
omitted source multipoles, supports and mechanical coupling need separate
validation before interpreting this component SNR as an experimental detection.

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
phase is passed to GWINC when `Optics.SRM.Tunephase` is `null`. GWINC's
`Tunephase` is `pi - 2 phi_SR` for the analytic Buonanno--Chen detuning
angle; an explicit numerical
YAML value is read with that convention. The squeezer from the selected
detector YAML is retained. No quantum, seismic, or
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
requirement. `Noise.Model` selects the project's analytic SNR model.
The default nominal squeezing level for both SNR and the quantum comparison
comes from `Squeezer.AmplitudedB` in the selected detector YAML. The analytic
model uses that dB level as an ideal reduction. The quantum comparison applies
the same ideal reduction to its unsqueezed GWINC curve; it does not model the
CE2 filter cavity in that figure. An explicit
`NoiseConfig(squeeze_db=...)` or quantum-figure `--squeeze-db` overrides the
level for one run. `configs/source.yaml` has no second squeezing setting.

## Quantum-noise comparison

```bash
python scripts/quantumNoise.py
python scripts/quantumNoise.py --squeeze-db 12 \
  --output runs/quantum-12db.png
python scripts/quantumNoise.py --target-frequency-hz 700 \
  --output runs/quantum-700hz.png
```

The blue curve loads `configs/CE2silicon.yaml` through GWINC, sets the SRM
transmittance to 1 and tune phase to 0, calculates the unsqueezed quantum
noise with GWINC's finite-arm strain calibration, and multiplies its ASD by
`10**(-dB/20)`. This ideal frequency-dependent squeezing rule also scales
GWINC's loss contributions; the blue curve is a modified CE2 optical
comparison, not the official CE2 sensitivity or its physical filter-cavity
prediction. GWINC 0.6.2's full quantum budget has a missing phase subterm at
exactly zero squeezing, so the script sums its direct quantum vacuum terms
once and applies the same strain calibration as the budget.

The purple curve uses the same CE2 GWINC optics with the hybrid SRM
transmittance and cavity length from `configs/detector.yaml`. GWINC has no
resonance-frequency field: its `Optics.SRM.Tunephase` rotates the SR cavity
quadratures, while GWINC also propagates the sidebands through that cavity.
The script varies `Tunephase` until GWINC's quantum PSD has zero frequency
slope and positive curvature at the source GW frequency. It finds that
phase using the unsqueezed GWINC calculation, then applies the same ideal
`10**(-dB/20)` ASD multiplier. For the default 600 Hz target, the solved
GWINC `Tunephase` is about `0.01142099 rad`; the analytic phase cannot be
copied directly because the two models have different optical transfer
calculations.

Orange uses the simple analytic model without signal recycling. Green uses
the analytic model with the hybrid SRM and its source-frequency detuning.
All four curves use the same ideal squeezing level. By default,
`--detector-config configs/detector.yaml` supplies the analytic parameters
and the shared `Squeezer.AmplitudedB`, while
`--gwinc-config configs/CE2silicon.yaml` supplies both GWINC curves' optical
parameters. `--t-srm` and `--length-sr` change the green and purple curves;
blue and orange remain no-SR references. `--target-frequency-hz` retunes both
detuned curves for one run, otherwise the target follows the source YAML.
The script prints the solved GWINC phase and all four ASDs at the target.
At 600 Hz with the current files these are about `1.11e-24` (blue),
`1.87e-25` (purple), `1.10e-24` (orange), and `3.49e-27 Hz^-1/2` (green).
The tuned GWINC dip remains substantially shallower than the analytic one.

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
