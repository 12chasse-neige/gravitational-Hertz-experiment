# Production integration: review guide

## What changed

The production observable is now the conserved leading mass-quadrupole response
of an equal-arm, ideal free-mass Michelson. No production calculation applies a
position-dependent TT projector. The source remains compact and slow-moving;
there is no expansion in kR, kL, or L/R. Static moments are excluded from the
oscillating signal. Suspension, feedback, optical springs, cavity calibration,
and environmental coupling are not added by this migration.

The theoretical equations are in [near-field-analysis.md](near-field-analysis.md).
That report predates the integration; its final implementation checklist describes
the old state. The original [theoreticalDerivation.md](theoreticalDerivation.md)
is retained as historical background, not the production response specification.

## Suggested line-by-line review order

| File / layer | What to inspect | Independent evidence |
| --- | --- | --- |
| `ghe/config.py` | Omitted versus explicit R; source/detector length, frequency and c synchronization | Explicit construction and replacement tests |
| `ghe/finite_distance.py` | Two-hole STF phasor; derivatives of exp(ikr)/r; trace reversal; mixed-index sign; all radial tidal terms | Time-domain hole Fourier sum, vacuum trace, harmonic constraint, Newtonian gradient, wave-zone limit |
| `ghe/detector_response.py` | Closed-arm clearance; batched longitudinal curvature; photon sampling factors; 1/(2LΩ²) normalization; refinement | Separate endpoint/path response and short-arm limit |
| `ghe/metric.py` | Source direction, body axis, enclosing radius, distance override, reception-time synthesis | Fixed benchmark and explicit-time tests |
| `ghe/optimization.py` | Configuration propagation, normalized amplitude objective, deterministic starts, cache invalidation | No-worse-than-seed and stale-cache tests |
| `ghe/source_array/` | Cartesian placement, per-row distance, transported axes, full-response phase delays, schema/metadata | All three strategies, split chunks, CSV/NPZ round trips |
| `ghe/signal.py` | Mechanical-delay sign, no double retardation, compensated sum, synthesis after summation | Independent delayed time evaluation, coherent gain, cancellation |
| `ghe/spectrum.py`, `ghe/snr.py` | Existing FFT normalization; paired file identity; matching noise inputs | FFT/phasor SNR agreement and N/N² scaling |
| `ghe/artifacts.py` | Physics and numerical identity gates | Missing, stale, mixed and modified artifact tests |
| `ghe/validation.py`, `scripts/` | No duplicate field equations; recorded inputs; isolated run outputs | Actual CLI tests and saved validation record |

Detailed comments explain physical steps and sign choices beside the relevant
operations. Array operations use explicit geometry axes and intermediate radial
quantities rather than collapsing the derivation into opaque expressions.

## Contracts and deliberate compatibility changes

### Public response

`calculate_response_phasor(theta_src, phi_src, theta_rot, phi_rot, *, config, R,
settings)` returns a **peak**, dimensionless complex response. Angles are in
radians; source angles point from vertex to COM and rotor angles along body +z.
`calculate_metric_response(t, ...)` returns `Re[H exp(-iΩt)]` at reception time in
seconds, with Ω=2ω. Compute H once for a time series.

`quadrupole_field` accepts positions with shape `(..., 3)` and one STF moment
with shape `(3, 3)`. Metric, acceleration, and tidal arrays retain their leading
batch dimensions. They have units 1, m/s², and s⁻² respectively.

### Phase alignment

For each source, the raw signal is `Re[H_i exp(-iΩt)]`. The existing helper returns
cosine phase `φ_i = -arg(H_i)`. Store

```
gw_phase_offset = wrap(φ_i - φ_reference)
rotor_phase_offset = gw_phase_offset / 2
```

A mechanical delay δ means `h_i(t - δ/ω)`, hence the corrected phasor is
`H_i exp(+2iδ)`. Its argument equals the reference argument. The stored geometric
`propagation_compensation_s` is for inspection only: H already contains source
retardation and the detector's light-travel response. Applying it again is wrong.

The `gw_*` field names remain for table compatibility; they identify the
oscillating gravitational response, not a radiative-only observable. Chunk-anchor
mode approximates rotor orientation only. Rigid mode still uses exact response
phases at its transported orientations. Zero-amplitude phase is defined as zero.

### Geometry and numerical settings

`SourceConfig(R=None)` derives distance from the configured R/L ratio. Explicit
R survives construction and `replace`. `replace(cfg, L=..., R=None)` requests
joint scaling; `replace(cfg, L=...)` preserves the existing distance.

Production validates a finite, physical two-hole rotor. The enclosing sphere has
radius `hypot(D/2,H/2)`; intersection with either closed arm segment is rejected.
This is a domain guard, not a guarantee that higher multipoles are negligible
immediately outside the sphere. Source compactness, slow motion and excluded
higher multipoles remain scientific assumptions.

Gauss–Legendre integration starts at 48 points per arm, doubles to at most 768,
and requires successive estimates to agree within 1e-9 times the sum of absolute
arm integrands. This scale is meaningful at differential-response nulls. The
reported refinement error is an estimate, not a rigorous error bound. Failure
raises `ArithmeticError`; invalid geometry raises `ValueError`. Numerical settings
and physical approximations are separate concepts.

### Storage and workflows

The model identifier is `conserved-quadrupole-free-michelson-v1`. Artifacts record
that identifier, phase convention, source configuration and integration settings.
Array metadata additionally records generation strategy and reference geometry.
CSV and paired NPY files have sidecars; NPZ and JSON embed metadata. Paired spectra
also store SHA-256 digests to reject mixed or modified payloads. An interrupted
CSV rewrite invalidates its old sidecar before truncation.

Production readers validate metadata. Low-level CSV/NPZ inspection helpers can
still read historical material but do not authorize its use in a calculation.
Unversioned geometry caches are recomputed. Unversioned arrays/spectra are
rejected, because their old optimized axes and phases are not interchangeable
with the new model. Geometry text now preserves 17 significant digits.

The scripts directory is `scripts/`; `SCR_DIR` remains only a path alias. Deleted
low-level TT/incomplete-metric functions have no compatibility backend. Independent
Newtonian mass quadrature and the old time-domain hole moment now live under
`tests/support/`. The maintained validation command replaces temporary diagnostics.
NPZ remains an in-memory format; streaming CSV is the supported large-array path.

## Verification record and known limits

The usable local environment was `ghe` (Python 3.13). The initial full test run
stopped at four missing `scr` imports. Once collection worked, the old resonance
check used an extra signal-recycling propagation term absent from both the
existing phase solver and noise function. Its expectation was corrected to the
retained model; the production noise equations were not changed.

[production-validation.json](production-validation.json) records frozen-source
benchmarks, dual-response residuals, quadrature refinement, complete source/noise
inputs, and timed small arrays. The expected benchmark peak amplitudes are
approximately 1.93584722024e-32 and 2.02610052329e-30. The independent harmonic and
curvature implementations agree far below the requested integration tolerance.
The 1/10/100-source examples compare direct monochromatic SNR to a bin-aligned FFT.

The tests also check the independent finite cylindrical-hole Newtonian integral,
which contains higher source multipoles and therefore agrees with the leading
quadrupole only within finite-size errors. Default absolute floating-point
assertion tolerances are not used to validate tiny strains.

Final verification: **76 tests passed** (`python -m pytest -q`), undefined-name
and duplicate-definition checks passed, and `git diff --check` was clean. The
arm-length plotting CLI also completed at 2000 m and 4000 m.

The test suite exercises actual CSV and NPZ CLI runs, mass/length sweeps, array
sweeps, and validation reports in temporary output directories. Small timings
exclude a ten-million-source performance claim. Array phase generation and later
signal evaluation each integrate a source once; responses are not persisted as a
second potentially stale cache. Fixed-anchor orientation remains an approximation,
and deterministic multistart searches are not a proof of a global optimum.

No full ten-million-source calculation or manuscript revision was performed.
