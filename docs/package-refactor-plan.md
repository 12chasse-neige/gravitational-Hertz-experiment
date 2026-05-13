# Package Refactor Plan

## Goal

Refactor the project from a collection of script-oriented modules into a small,
testable Python package while preserving the current physics behavior at each
step. The refactor should make large source-array experiments easier to run,
reproduce, cache, and validate without changing the scientific model as part of
the structural move.

## Non-Goals

- Do not redesign the gravitational-wave source model during the package move.
- Do not change SNR formulas, FFT normalization, or quantum-noise formulas unless
  a dedicated validation step identifies a bug.
- Do not remove the existing command-line workflows until replacement CLIs are
  working and documented.
- Do not rewrite the whole codebase in one commit. The migration should happen in
  small, reversible phases.

## Current Problems To Address

- Core calculations, command-line parsing, plotting, and file I/O are mixed in
  the same files.
- `scripts/sourceArray.py` is too large and currently owns layout generation,
  geometry transforms, optimization strategy, phase recovery, CSV writing, and
  CLI behavior.
- Several modules communicate through mutable files in `data/` instead of typed
  function inputs and outputs.
- Configuration is split between environment variables, module constants, and
  dataclass defaults. Some defaults can diverge, such as source-side arm length
  and detector-side arm length.
- Large-array workflows still rely on CSV and repeated row reads, which will not
  scale to `10^7` sources.
- Documentation paths and output paths are inconsistent in places.

## Target Layout

```text
gravitational-Hertz-experiment/
  ghe/
    __init__.py
    config.py
    paths.py
    geometry.py
    metric.py
    optimization.py
    signal.py
    spectrum.py
    noise.py
    snr.py
    source_array/
      __init__.py
      schema.py
      layout.py
      phase.py
      strategies.py
      io.py
      generation.py
  scripts/
    bestPosition.py
    fourier.py
    metricCalculate.py
    noiseAnalysis.py
    plotSNRCurve.py
    quantumNoise.py
    runSNR.py
    sourceArray.py
  main.py
  docs/
    theoreticalDerivation.md
    package-refactor-plan.md
  tests/
    test_geometry.py
    test_metric_smoke.py
    test_source_array_generation.py
    test_snr.py
```

The `ghe/` package should contain reusable library code. The existing files in
`scripts/` and `main.py` should become thin compatibility wrappers that call
package functions.

## Proposed Module Responsibilities

### `ghe.paths`

- Own repository-relative paths such as `DATA_DIR`, `IMAGES_DIR`, and default
  output file names.
- Keep path constants separate from physical configuration.

### `ghe.config`

- Define dataclasses for:
  - `SourceConfig`
  - `DetectorConfig`
  - `SamplingConfig`
  - `SourceArrayConfig`
  - `NoiseConfig`
  - `RunConfig`
- Provide explicit constructors from environment variables where needed.
- Make all run-critical values serializable to JSON.

### `ghe.geometry`

- Own coordinate transforms:
  - spherical to Cartesian
  - Cartesian to spherical
  - body-to-detector rotation
  - vector transport between propagation directions
- Include low-cost unit tests for normalization, angle round trips, and rotation
  orthogonality.

### `ghe.metric`

- Own the source quadrupole and detector response:
  - hole coordinates
  - quadrupole tensor
  - second derivative of quadrupole tensor
  - metric tensor in body frame
  - TT projection
  - arm delay integrals
  - final metric response
- The public function should accept explicit config and geometry arguments rather
  than reading `data/bestPosition.txt` internally.

### `ghe.optimization`

- Own best-geometry optimization and fixed-source rotor optimization.
- Return typed results instead of writing files directly.
- Provide a separate helper for saving/loading best-position results.

### `ghe.source_array.schema`

- Own the source-array dtype, column names, and any conversion helpers.
- Keep CSV/NPZ/HDF5 schema definitions in one place.

### `ghe.source_array.layout`

- Own lattice dimension selection and source position generation.
- Provide chunk iterators over source IDs and positions.

### `ghe.source_array.phase`

- Own signal amplitude and phase recovery.
- Own phase wrapping and rotor phase conversion.

### `ghe.source_array.strategies`

- Own generation strategies:
  - `exact`: optimize each source.
  - `rigid`: rigidly transport the reference rotor axis.
  - `chunk_anchor`: optimize one anchor per approximation group.
- Make strategy selection explicit with an enum or string literal.

### `ghe.source_array.io`

- Own reading/writing source arrays.
- Keep CSV compatibility, but add a binary format such as `.npz` as the preferred
  large-array format.

### `ghe.source_array.generation`

- Orchestrate source-array context building and chunk generation.
- Return chunks as structured arrays or typed records.

### `ghe.signal`

- Own signal generation for:
  - single-source time series
  - source-array time series
  - chunked accumulation over source arrays
- Avoid re-reading the source-array file inside the innermost loop.

### `ghe.spectrum`

- Own FFT calculation and spectrum data objects.
- Keep plotting separate from FFT calculation.

### `ghe.noise`

- Own detector quantum-noise functions.
- Keep analytic model and `gwinc` comparison paths clearly separated.

### `ghe.snr`

- Own SNR integration.
- Accept arrays and config directly.
- File-based wrappers can remain in legacy scripts.

## Migration Phases

### Phase 0: Baseline Capture

Purpose: establish behavior before moving code.

Tasks:

- Record the current commands used for common workflows:
  - `python main.py`
  - `python main.py --renew-source-array ...`
  - `python scripts/sourceArray.py --summary-only ...`
  - `python scripts/fourier.py`
  - `python scripts/noiseAnalysis.py`
  - `python scripts/runSNR.py ...`
- Save representative small-run outputs for comparison.
- Add a short `docs/current-workflows.md` if the commands are not already clear.
- Decide the numeric tolerances for regression checks.

Acceptance criteria:

- At least one small source-array run can be reproduced from a documented command.
- Baseline values for best position, sample signal values, FFT peak, and SNR are
  recorded.

### Phase 1: Add Package Skeleton

Purpose: create `ghe/` without changing behavior.

Tasks:

- Add `ghe/__init__.py`.
- Add `ghe/paths.py` and move path constants from `scripts/projectConfig.py`.
- Keep `scripts/projectConfig.py` as a compatibility wrapper importing from the
  new package.
- Add minimal tests that import the new package.

Acceptance criteria:

- Existing scripts still run.
- `python -m py_compile main.py scripts/*.py ghe/*.py` passes.

### Phase 2: Split Configuration

Purpose: make physical and runtime parameters explicit.

Tasks:

- Move `ExperimentConfig` and sampling config into `ghe.config`.
- Rename config classes to express responsibility:
  - source parameters
  - detector parameters
  - sampling parameters
  - run parameters
- Make `DetectorConfig.length` and source response arm length come from the same
  run config when used together.
- Keep environment-variable support, but isolate it in constructor helpers.

Acceptance criteria:

- Existing scripts still accept the same environment variables.
- A serialized run config can fully describe a small run.

### Phase 3: Extract Geometry Utilities

Purpose: isolate coordinate math from source-array and metric code.

Tasks:

- Move spherical/cartesian conversion and vector rotation utilities into
  `ghe.geometry`.
- Replace local copies in `metricCalculate.py` and `sourceArray.py` with imports.
- Add tests for:
  - unit vector norm
  - spherical round trip
  - rotation matrix orthogonality
  - transported vector norm preservation

Acceptance criteria:

- Tests pass.
- Source-array preview output is unchanged within tolerance.

### Phase 4: Extract Metric Core

Purpose: make the detector response a clean library API.

Tasks:

- Move tensor and response functions into `ghe.metric`.
- Change the core response API to accept explicit config and angles.
- Keep the legacy default best-position lookup only in wrapper code.
- Update `scripts/metricCalculate.py` to call `ghe.metric`.

Acceptance criteria:

- For a fixed set of angles and times, old and new metric responses match within
  tolerance.
- `scripts/bestPosition.py` still produces the same best-position format.

### Phase 5: Extract Optimization

Purpose: remove circular script-level dependencies between metric and optimizer.

Tasks:

- Move optimization objective and SciPy minimization wrapper into
  `ghe.optimization`.
- Define a `BestGeometry` result dataclass.
- Add load/save helpers for `data/bestPosition.txt` or a new JSON equivalent.
- Keep `scripts/bestPosition.py` as a CLI wrapper.

Acceptance criteria:

- Best-position CLI produces compatible output.
- Source-array generation can request cached or recomputed geometry through the
  package API.

### Phase 6: Split Source Array Code

Purpose: reduce `sourceArray.py` into focused modules.

Tasks:

- Move dtype and column definitions to `ghe.source_array.schema`.
- Move lattice helpers to `ghe.source_array.layout`.
- Move amplitude/phase helpers to `ghe.source_array.phase`.
- Move exact, rigid, and chunk-anchor logic to `ghe.source_array.strategies`.
- Move CSV writing and future binary writing to `ghe.source_array.io`.
- Move orchestration to `ghe.source_array.generation`.
- Keep `scripts/sourceArray.py` as a wrapper around the package API.

Acceptance criteria:

- `python scripts/sourceArray.py --summary-only --num-sources 1000` still works.
- `python scripts/sourceArray.py --chunk-center-approximation ...` still works.
- Generated CSV columns and values match the current implementation within
  tolerance.

### Phase 7: Add Binary Source-Array Storage

Purpose: prepare for `10^7` source arrays without forcing CSV as the main format.

Tasks:

- Add `.npz` read/write support for structured source arrays.
- Keep CSV export for inspection and compatibility.
- Update `main.py` or its package backend to prefer `.npz` when available.
- Include metadata with:
  - source-array config
  - generation strategy
  - optimization chunk size
  - code version or git commit if available

Acceptance criteria:

- Small arrays round-trip through `.npz`.
- CSV and NPZ generation produce equivalent arrays for a small run.

### Phase 8: Refactor Signal Generation

Purpose: remove repeated full-file CSV reads and enable chunked accumulation.

Tasks:

- Move source-array signal accumulation into `ghe.signal`.
- Replace the current loop shape:

```text
for time:
  for source:
    read source row
    calculate response
```

  with:

```text
h_total = zeros_like(time_axis)
for source_chunk:
  h_total += calculate_chunk_response(time_axis, source_chunk)
```

- Keep a simple implementation first, then vectorize where safe.
- Add a smoke test comparing the old and new paths on a tiny array.

Acceptance criteria:

- For a small array, old and new total signals match within tolerance.
- The code no longer reads the whole CSV once per source per time sample.

### Phase 9: Extract Spectrum, Noise, And SNR

Purpose: make post-processing reusable and testable.

Tasks:

- Move FFT logic into `ghe.spectrum`.
- Move quantum-noise logic into `ghe.noise`.
- Move SNR integration into `ghe.snr`.
- Update `scripts/fourier.py`, `scripts/quantumNoise.py`, and
  `scripts/noiseAnalysis.py` to call package functions.
- Keep plotting as CLI or `ghe.plotting` helpers, but do not mix plotting with
  numerical kernels.

Acceptance criteria:

- Existing `.npy` spectrum outputs are still produced by the compatibility CLIs.
- SNR from saved data matches previous results within tolerance.

### Phase 10: Introduce Run Directories

Purpose: make experiments reproducible.

Tasks:

- Add a run-output helper that creates:

```text
runs/<timestamp-or-name>/
  config.json
  best_position.json
  source_array.npz
  signal.npy
  spectrum.npz
  snr.json
  plots/
```

- Keep legacy `data/` outputs for now.
- Add a CLI option such as `--run-dir` or `--output-dir`.

Acceptance criteria:

- A full small run can be reproduced from `config.json`.
- Legacy workflows still write the expected files unless explicitly redirected.

### Phase 11: Documentation And Cleanup

Purpose: make the new structure understandable.

Tasks:

- Update README paths from old names to current names.
- Document exact meanings of:
  - source-array write chunk size
  - optimization strategy
  - chunk-anchor approximation group size
- Add a module map from old scripts to new package modules.
- Add a short contributor note explaining where new code should go.

Acceptance criteria:

- README examples run.
- No stale references to `sourcedata/` or missing documentation paths remain.

## Suggested Commit Order

1. Add package skeleton and compatibility imports.
2. Move path/config definitions.
3. Move geometry helpers and tests.
4. Move metric core and tests.
5. Move optimization helpers.
6. Split source-array modules.
7. Add binary source-array I/O.
8. Refactor total signal generation.
9. Move spectrum/noise/SNR modules.
10. Add run-directory support.
11. Update README and cleanup.

## Testing Strategy

Use small deterministic cases first. The goal is not broad physics validation in
every structural commit, but quick detection of accidental behavior changes.

Recommended checks:

- Geometry unit tests with exact or tight numerical assertions.
- Metric smoke tests at a few fixed times and angles.
- Source-array generation tests for `exact`, `rigid`, and `chunk_anchor`.
- FFT tests using a synthetic sinusoid with known frequency.
- SNR integration tests with simple synthetic arrays.
- CLI smoke tests for the existing commands.

Recommended command group:

```bash
python -m py_compile main.py scripts/*.py ghe/**/*.py
pytest
python scripts/sourceArray.py --num-sources 100 --summary-only
python main.py --renew-source-array --source-array-num-sources 100 --source-array-chunk-size 10
```

## Risk Controls

- Keep compatibility wrappers until all replacement paths are verified.
- Move code before changing behavior.
- Add tests before optimizing large-array performance.
- Avoid changing formulas and storage formats in the same commit.
- Preserve legacy output files until the new run-directory workflow is stable.
- When a result changes, document whether it is a deliberate physics/model fix or
  an unintended regression.

## First Implementation Slice

The safest first slice is:

1. Add `ghe/__init__.py`, `ghe/paths.py`, and `ghe/config.py`.
2. Make `scripts/projectConfig.py` import and re-export those definitions.
3. Add one import smoke test.
4. Run `python -m py_compile main.py scripts/*.py ghe/*.py`.

This gives the project a package anchor without forcing a big rewrite.
