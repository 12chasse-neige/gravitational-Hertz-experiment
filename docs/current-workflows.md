# Current calculation workflows

Activate `ghe` or install `requirements.txt`, then run from the repository root.
`python main.py` lists the workflows. Use `python main.py COMMAND --help` for
options; direct `python scripts/NAME.py ...` invocations also work.

## Detuned interferometer design

Edit `configs/detector.yaml` for the active design and `configs/source.yaml` for
the rotor, frequency, placement, sampling and array defaults. Keep comparison
YAMLs separate so comparisons retain their reference optics.

```bash
python main.py quantum --output runs/design/quantum.png
python main.py quantum --target-frequency-hz 700 --squeeze-db 12 \
  --output runs/design/quantum-700hz.png
python main.py quantum --t-srm 0.5 --length-sr 55 \
  --gwinc-sr-tuning analytic-phase --output runs/design/quantum-srm.png
python main.py noise-budget --output runs/design/noise-budget.png
python main.py thermal-gas --config configs/detector.yaml \
  --output-dir runs/design/thermal-gas
```

`quantum` compares two analytic models and two GWINC curves. The analytic curves
use `--detector-config` (default `configs/detector.yaml`); both GWINC curves use
`--gwinc-config` (default `configs/CE2silicon.yaml`). These can have different
arm lengths. To compare the active optical design in both models, pass
`--gwinc-config configs/detector.yaml` explicitly. The script prints their
parameters and target-frequency ASDs. A target override affects the comparison
only; edit the source YAML to change the rotor signal in other workflows.

The detuned GWINC curve uses the `b_2` readout, homodyne angle zero. Its default
`--gwinc-sr-tuning target-minimum` searches for a GWINC PSD minimum at the target.
If none exists, `analytic-phase` uses the analytic physical detuning with
`Tunephase = pi - 2*phi_SR`; this does not guarantee a GWINC minimum. The analytic
and GWINC optical transfer models differ, so their minima and depths can differ.
All comparison curves use the same ideal squeezing multiplier; this figure does
not model filter-cavity loss and rotation.

`noise-budget` writes matching PNG/PDF figures with analytic quantum, individual
thermal terms, gas scattering/damping and their PSD-summed total. `thermal-gas`
writes `thermal_gas.csv`, `.json` and `.png`; `--no-plot` omits the PNG. Its
notebook gas comparisons and pressure ceilings are diagnostics, excluded from
the SNR noise budget. Reference runs can select `configs/aLIGO.yaml`,
`configs/CE2silicon.yaml` or `configs/Voyager.yaml` with separate output directories.

The combined SNR budget is

$$
S_h = S_{\rm quantum}^{\rm analytic} + S_{\rm thermal}^{\rm GWINC}
      + S_{\rm gas}^{\rm GWINC}.
$$

Each one-sided PSD is added once. Squeezing affects quantum noise only. Seismic,
Newtonian and rotor-induced environmental disturbances are not included.

## Single-rotor Qdot-phase projection

```bash
python main.py qdot --output-dir runs/qdot
python main.py qdot --angles 0.6 0.8 0.4 0.8 --distance 40 \
  --rotor-delay-rad 0.2 --duration 0.01 --sample-rate 120000 \
  --output-dir runs/qdot-example
python main.py qdot --angles 0.6 0.8 0.4 0.8 --distance 40 \
  --rotor-delay-rad 0.2 --signal-input runs/qdot-example/signals.npz \
  --output-dir runs/qdot-loaded
python main.py qdot --noise-budget quantum --observation-years 0.25 \
  --output-dir runs/qdot-quantum
python main.py qdot --extraction-method model-subtraction \
  --output-dir runs/qdot-model-subtraction
```

Angles are `theta_src phi_src theta_rot phi_rot` in detector-frame radians.
Without explicit angles the script reads a compatible best-geometry cache or
optimizes the **total** response without overwriting that cache. Distance defaults
to configured `R/L`; rotor delay defaults to zero. A mechanical delay `delta`
multiplies every phasor by `exp(2i delta)`. The signal uses `Omega=2*omega` and
`h(t)=Re[H exp(-i Omega t)]` at the original reception timestamps.

The explicit retarded-time derivative components `Q`, `Qdot`, `Qddot`, `Q3`, `Q4`
scale from `G Q/r^5` through `G Q4/(c^4 r)` in the tidal field and pass through
the same finite-arm response. `qdot_model` is the predicted Qdot contribution to
detector **strain**, not a quadrupole derivative in kg m²/s.

The default `matched-sine` method projects the full input onto

```text
s(t) = sin(Omega*time_s - theta_Qdot + pi/2)
h_projected(t) = b*s(t)
```

Only the Qdot phase comes from the model. No remaining-term amplitude is
subtracted. `--qdot-phase-rad THETA` supplies a known detector phasor phase,
including the timing reference and rotor delay. For a uniform whole-period
record, `b=2*mean(h*s)`; otherwise a joint cosine/sine/DC least-squares fit
accounts for finite-window overlap using nuisance coefficients fitted to data.
Nonuniform increasing timestamps receive equal sample weights.

**Identifying this projection with Qdot assumes Q/Qddot are orthogonal to the
Qdot-phase sine and Q3/Q4 are negligible.** The full response need not satisfy
these assumptions. Reports retain every term's phase overlap and signed
projection, so a small overlap with a much larger Q term remains visible.
The optional `model-subtraction` diagnostic instead assumes all four remaining
modeled terms are known exactly and subtracts them before fitting.

A loaded NPZ needs finite real one-dimensional `time_s` and `signal` arrays of
equal length, with at least three strictly increasing timestamps. Loaded inputs
require explicit angles and the correct distance/delay. If model and geometry
metadata are present they must match. `--duration` and `--sample-rate` apply only
to generated signals. An array spectrum is not a single-rotor input record.

Outputs are `report.json`, `snr.json`, `signals.npz`, `signals.csv` and
`comparison.png`. They preserve input times, modeled components, reference phases,
fit residuals, overlap diagnostics and noise components. The measured signed
coefficient is `signed_sine_coefficient`; `qdot_extracted` is the historical array
name for the selected estimate and `qdot_sine_projection` labels it explicitly.
`rho_sine_projection` and `rho_qdot_model` distinguish projected and modeled SNR.

For peak strain amplitude `A`, the coherent statistic is
`rho=A*sqrt(T_obs/S_h(f_gw))`. `--observation-years` (default one) controls this
observation independently of the short synthetic record. The default budget is
`quantum-thermal-gas`; `--noise-budget quantum` selects quantum only.

## Qdot SNR versus arm length

```bash
python main.py qdot-sweep --min-length 4 --max-length 40 --points 5 \
  --output-dir runs/qdot-short-arms
python main.py qdot-sweep --min-length 100 --max-length 59000 --points 121 \
  --output-dir runs/qdot-long-arms
python main.py qdot-sweep --min-length 4 --max-length 40 --points 5 \
  --angles 0.6 0.8 0.4 0.8 --output-dir runs/qdot-fixed-angles
```

`ghe.qdot_scaling.QdotArmLengthStudy` evaluates and caches individual lengths.
By default it optimizes geometry for **the modeled Qdot component** at each
length. `--angles` keeps geometry fixed. It preserves the configured `R/L`,
rotor properties, mirror curvatures and optical/material parameters, recalculates
beam sizes and noise, and retunes the analytic SR phase to the source frequency.
It does not change YAML files or the shared geometry cache.

This is a coupled arm-length/source-distance study. Its modeled component SNR
assumes exact separation of the other terms; it is distinct from the measured
sine-projection SNR. Optical instability, confocal degeneracy and excessive
clipping are reported as exclusions. Interior maxima are refined in log length;
a maximum at a chosen endpoint is reported as a boundary result. The default
search remains 100 m to 59 km; choose bounds appropriate to the current design.

Outputs are `qdot_arm_length.csv`, `.json` and `.png`, including source/detector
parameters, input hashes, component phasors, noise terms, geometry, source-arm
clearance, exclusions and refinement diagnostics.

## Coherent rotor arrays and total-signal SNR

```bash
# Generate CSV and NPZ without calculating SNR.
python main.py array-generate --num-sources 10 --chunk-size 5 \
  --chunk-center-approximation --approximation-chunk-size 5 --format both \
  --output runs/array-input/source_array.csv \
  --npz-output runs/array-input/source_array.npz

# Reuse the array: direct phasor or time-series/FFT calculation.
python main.py array --source-array-input runs/array-input/source_array.npz \
  --monochromatic --noise-budget quantum-thermal-gas --run-dir runs/array-mono
python main.py array --source-array-input runs/array-input/source_array.csv \
  --noise-budget quantum-thermal-gas --run-dir runs/array-fft
python main.py array --source-array-input runs/array-input/source_array.npz \
  --monochromatic --noise-budget quantum --run-dir runs/array-quantum

# Or generate and calculate in one run.
python main.py array --renew-source-array --source-array-num-sources 10 \
  --source-array-chunk-size 5 --source-array-format csv \
  --monochromatic --noise-budget quantum-thermal-gas --run-dir runs/array-example
python main.py array-sweep --array-sizes 1,3,10 --strategy chunk-center \
  --noise-budget quantum-thermal-gas --output runs/array-scaling.csv
```

Array phase delays align each source's **complete complex detector response**,
including source retardation and finite-arm propagation. No extra propagation
delay is applied afterward. Chunk-center approximation and rigid transport
approximate orientation; every source phase still uses its full response.
`--source-array-exact-optimization` in `array`, or
`--no-chunk-center-approximation` in `array-generate`, selects per-source rotor
optimization. `--no-optimize-each-source` selects rigid transport.

`array` and `array-sweep` retain their quantum-only default. Select
`--noise-budget quantum-thermal-gas` for the combined budget shown above.
The chosen budget is stored with SNR and
configuration reports; `array-sweep` also stores it in the CSV metadata sidecar.
The package SNR APIs retain their quantum-only default, so select
`noise_budget="quantum-thermal-gas"` explicitly when calling them directly.

The array statistic is **total-signal SNR**, including Q and every derivative,
not an isolated Qdot measurement. Phasor and FFT estimates agree for a
bin-aligned monochromatic signal. Peak amplitude and SNR scale as `N`, and
squared SNR as `N²`, only for equal aligned source responses at fixed noise;
spatially extended arrays have unequal source amplitudes.

Use streaming CSV for very large arrays; compressed NPZ is loaded in memory.
Check source/arm clearance and multipole validity when enlarging an array around
a nearby detector. Counts in examples are smoke runs, not capacity estimates.
The configured production count is not started by invoking `python main.py`
without a command.

A run directory contains `config.json`, `snr.json`, optionally regenerated arrays,
and either `phasor.npy` or time-domain signal plus spectrum files. Payloads and
`.metadata.json` sidecars must stay together. Changed physical parameters require
regeneration; compatible geometry caches may be reused. The former spelling
`--use-mono-approx` and original `main.py --renew-source-array ...` syntax remain
supported.

## Verification and retained material

```bash
python main.py validate --output-dir runs/validation
```

Validation compares curvature and independent harmonic responses at fixed
4 km/6 km benchmarks, checks that all derivative components recombine, verifies
Qdot-phase projections on shifted non-whole-period timestamps for both frozen
and active configurations, checks the active combined PSD, and exercises
1/10/100-source arrays with both quantum-only and combined-budget FFT/phasor SNR.
It writes `validation.json` and small array/spectrum artifacts in its output
folder. Geometry caches for validation stay in that folder.

The derivations, historical JSON reports, local manuscript/reference material,
existing `data/` and previous runs are retained. Historical reports have frozen
inputs and are not claims about the active YAML configuration. The old standalone
metric/FFT/noise/mass-length wrappers were removed; numerical APIs remain in
`ghe/`, with array SNR in `array` and rotor studies in `qdot`/`qdot-sweep`.
