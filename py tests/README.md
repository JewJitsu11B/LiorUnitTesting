# LiorUnitTesting

Validation and unit-test suite for the Causal Accumulation Law (CAL) framework.

This repository is the computational companion to the CAL manuscripts. It checks, in
fully deterministic (seeded) code, the algebraic and functional claims those papers make:
biquaternion source currents and their two intrinsic closures, the bilocal variable-order
entropy/detector functionals, the causal memory kernel, and the emergence of Lorentzian
and Born structure. Every result is reproducible bit-for-bit from a fixed seed.

Related record: doi:10.5281/zenodo.19781544

## Layout

### `cal/` - the package under test
- `biquaternion.py` - biquaternion (C tensor H) arithmetic: products, the two conjugations
  (reduced-norm bar and Hermitian dagger), the M2(C) matrix embedding, and the
  token-channel embeddings (geometric and spectral rails).
- `entropy.py` - the bilocal energy density, the causal kernel, the variable-order field,
  and the variable-order entropy / detector functionals (Shannon, NIST single-log, Renyi,
  the mixed Renyi-Tsallis lens form, and the Hermitian-norm form).
- `cal_tensor.py` - the causal accumulation tensor and the power-law memory kernel.
- `propagator.py` - the causal propagator and biquaternion transport.
- `metric.py` - the emergent-metric machinery.

### `test_*.py` - the pytest suite
- `test_biquaternion.py`, `test_entropy.py`, `test_metric.py`, `test_associator.py`,
  `test_dissolution.py` - the core algebra / entropy / metric / transport test groups.
- `test_entropy_nist_vs_renyi.py`, `test_dual_log_entropy_comparison.py`,
  `test_dual_entropy_channels.py` - the single-log entropy vs dual-log detector comparisons
  (concavity, sign structure, lens/content channel decomposition).
- `test_effective_temperature.py` - the ET1-ET5 superstatistical-form suite (q = 1 + 1/alpha_K).
- `test_superstatistics_bornmax.py` - the Gamma-temperature / q-exponential BornMax identities.
- `test_bornmax_architecture.py` - end-to-end: entropy in the energy slot, detector in the
  temperature slot.

### `exp_*.py` - standalone validation experiments
Each is a self-contained script (`python exp_<name>.py`) that prints a labelled,
reproducible result. The headline checks:
- `exp_two_closures.py` - the two intrinsic biquaternion closures: the reduced norm
  q\*qbar (the determinant; multiplicative; indefinite; vanishes on the null cone) versus
  the Hermitian self-product <q qdag>_0 (sum |z|^2; positive; the Born channel). C1-C7.
- `exp_holonomy_detlift.py` - path-accumulated transport lifts the causally integrated
  field off the null cone (aggregate det 0 -> ~3.6e3) while a flat connection does not.
- `exp_hermitian_norm_renyi.py` - the lens-form detector is degenerate at the unitary
  corner (the order nu -> 1); the Hermitian-norm form is non-degenerate and recovers the
  Shannon value as the analytic limit.
- `exp_five_way_static.py` - the five-way corr(functional, surprisal) on the static field,
  swept over the effective order; documents the (1 - nu_bar) clamp inflation (0.99 vs 0.79).
- `exp_fracint_check.py` - the memory kernel (1 + tau)^(-alpha_K) reproduces the
  Riemann-Liouville fractional integral I^(1 - alpha_K) asymptotically, with the semigroup.
- `exp_double_cover_repro.py` - the Spin(2) -> SO(2) double cover: the period-2 modulus
  component and the period-4 sign-flipping interference component.
- `exp_expectancy_walk.py` - the detector as a read along a biased random walk: learning
  improves transport, history dependence is non-Markovian, and diverse paths give a
  convergent outcome (the many-to-one signature).
- `exp_feedback_necessity.py` - an honest negative: self-referential feedback alone does
  not produce deterministic chaos; the effective stochasticity is access-limited, not
  sensitive dependence.

Additional sweeps and diagnostics: `exp_corr_sweep.py`, `exp_nubar_twosided.py`,
`exp_detector_corr_check.py`, `exp_full_detector_sweep.py`, `exp_entropy_detector.py`,
`exp_cp_kernel_sweep.py`, `exp_born_input.py`, `exp_trough_principled_E.py`, and
`exp_closed_loop_field.py` (the closed-loop field harness).

#### The Born-rule audit group (2026-07)

Eight scripts checking the derivation in `born_axiomatic.tex` and the `thm:Born_fp` chain in
`CAL_Unified_Manual.tex`. Written up in `reports_n_findings/CAL_BORN_AUDIT_2026-07.md`. These do
**not** duplicate `exp_two_closures.py` (C1-C7); each adds a distinct claim. Several are negatives
and are kept as negatives.

- `exp_clifford_conjugation_identification.py` - the bar IS Clifford conjugation and the dagger IS
  REVERSION in Cl(3,0), so the Born closure is the canonical geometric-algebra norm rather than a
  bespoke composite. Also refutes the claim that reversion sends `t + x` to `t - x` (it fixes
  grade-1 vectors; Clifford conjugation is what flips them).
- `exp_born_form_uniqueness.py` - Lemma 4.3's uniqueness is false: `a|q0|^2 + b*sum|qk|^2` with
  `a != b` passes every invariance the lemma states. Repaired by requiring invariance under the
  spinor action `q -> u q` (SU(2)), which costs unitarity as a premise.
- `exp_born_paravector_forward_cone.py` - `W^dag W` is a forward-cone paravector (grade 0+1 only),
  `t^2 - |x|^2 = |det W|^2`, null exactly on the zero divisors, rank 2 generic / 1 on the cone.
  Also: the Born weight is a time component, not a Lorentz scalar.
- `exp_born_invariance_vs_positivity.py` - for quadratic forms the algebra gives positive XOR
  invariant; `|N(W)|^2` escapes only by going quartic and degenerate, and fails Parseval. Also the
  "split (2,2)" carrier computes to (4,4).
- `exp_born_fixedpoint_channel_width.py` - **headline negative.** The Born fixed point requires
  `tau = 2` (the map returns its input only there), while the theorem states it "requires no
  condition on the channel width tau" and its own proof preamble says Steps 1-4 "read off the
  channel width it requires".
- `exp_frft_even_order_reality.py` - `F^alpha` preserves reality iff alpha is even (`F^2` is
  parity), but being real-linear it cannot *create* reality: a complex action channel stays complex
  at every alpha.
- `exp_holomorphic_dual_channels.py` - `H = -ln|psi|^2` is harmonic and its holomorphic dual is
  `arg(psi)`; good structure, but absent from the manuscript, which instead needs an imaginary
  action. Includes the decisive check that `theta_R = theta_I` plus Cauchy-Riemann forces both
  drivers constant.
- `exp_polar_form_interference.py` - the quaternionic polar form `J = rho e^{n theta}` destroys the
  interference cross term (`|J|^2 = rho^2`), reintroduces the preferred quaternionic direction the
  centrality argument forecloses, and confines `J` to a 2-of-8-dimensional slice.

**Import note.** `cal/` lives at the repo root, one level above this folder, and Python puts the
*script's* directory on `sys.path` -- not the cwd. So a bare `from cal.biquaternion import ...`
raises `ModuleNotFoundError` when an `exp_*.py` is run directly, which is why the audit scripts
insert the parent directory on `sys.path` first. This affects the older `exp_*.py` scripts too:
they currently need `PYTHONPATH=<repo root>` to run standalone.

### `reports_n_findings/`
- `FINDINGS_REPORT.md` - a written summary of the validated results, the methods, and the
  "so what" for each, including the one honest negative.

## Running

Tests:
```
python -m pytest -q
```
Experiments:
```
python exp_two_closures.py
```
`conftest.py` puts this folder on `sys.path` so `import cal...` resolves to the local
`cal/`. Requires `numpy`, `torch`, and `scipy`. All scripts are seeded; identical-seed
reruns are bit-for-bit identical.

## Scope

These tests validate the functionals and the algebra; they do not re-derive the framework
(the full derivations are in the separate CAL derivations manuscript). A result that comes
back negative is reported and kept as a negative, not reframed.
