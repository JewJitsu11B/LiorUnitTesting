# CAL validation suite -- findings report

Companion unit-test / validation results for `CAL_Unified_Manual.tex`. Every number below
is from a fresh, deterministic (seeded) re-run of the suite in `LiorUnitTesting`
(github.com/JewJitsu11B/LiorUnitTesting), built on the `cal` package. The full derivations
live in the separate `CAL_deriv` manuscript; this suite's job is validation, not derivation.

## Methods

- Backend: PyTorch on CUDA (RTX 4090), `cal` package (`biquaternion`, `entropy`,
  `propagator`, `cal_tensor`). All scripts seeded (`torch.manual_seed`,
  `np.random.default_rng`); identical-seed reruns are bit-for-bit identical.
- Token-channel embeddings (`embed_tokens_channels`, geo + spec rails); bilocal energy
  density `energy_matrix`; causal kernel `causal_kernel_matrix` (zeros `y>=x`); variable
  order field `nu_field`.
- Each functional is run on its own and against fixed-order / NIST single-log baselines, so
  any reported correlation is read against a floor, not in isolation.
- Honest-negative discipline: a test that fails to confirm a claim is reported as a negative,
  not reframed into a pass.

## Results

### 1. Two intrinsic closures of the biquaternion (`exp_two_closures.py`) -- PASS (C1-C7)

The biquaternion carries two distinct conjugations with two distinct jobs.

- Reduced norm `q*qbar` is a pure scalar, equals `quat_norm_sq` and `det(X_q)`, and is
  multiplicative: `N(qr) = N(q)N(r)` to `max|diff| = 3.3e-5` (float roundoff).
- Hermitian closure `<q qdag>_0 = sum|z_n|^2` is real and `>= 0`, and is NOT multiplicative
  (`mean|ratio-1| = 0.279`).
- Complementarity on the null cone: the explicit zero divisor `z = (1, h, 0, 0)` has reduced
  norm `N(z) = 0` and `det(X_z) = 0` (on the cone, non-invertible), yet a positive Born
  weight `<z zdag>_0 = 2.0`. The metric channel degenerates exactly where the Born channel
  stays well-defined.
- Mechanism: the privileged complex imaginary obeys `h_bar = +h` (unchanged by the bar) but
  `h_dag = -h` (flipped by the dagger). So `<W Wbar>_0` is complex (interference survives,
  `max|imag| = 11.61`) while `<W Wdag>_0` is real (interference cancels).

So-what: the Born posit uses the DAGGER closure, not the reduced norm. The interference
cancellation that makes the Born weight real is the `h_dag = -h` structure, and it is
independent of the indefinite metric the reduced norm carries. One algebra, two channels.

### 2. Holonomy lifts the integrated field off the null cone (`exp_holonomy_detlift.py`) -- PASS

Starting from a genuine null element (`det = 0`): path-accumulated transport keeps each
individual transported copy on the cone (`max|det| = 1.8e-7`, single conjugation preserves
det), but the causally accumulated AGGREGATE reaches `|det| = 3567` (off the cone). The flat
(identity-connection) control keeps the aggregate at `det = 0`.

So-what: the curved-connection holonomy is what carries the realized field off the
zero-divisor locus, so the Born observable is well-defined on physical support. The zero
divisors are not removed (a fixed algebraic fact); the field is carried off them. This is the
"spatialize, not dissolve" property.

### 3. Detector degeneracy at the unitary corner, and the fix (`exp_hermitian_norm_renyi.py`) -- PASS

The lens-wrapped detector `H^v_hyb` is degenerate at `nu = 1` (alpha = 1, the unitary
corner): the lens factor `-ln nu` zeros the integrand exactly where the `1/(1-nu_bar)`
prefactor diverges. Measured: detector(lens) runs `-18, -456, -6903, -6912, -465, -27` across
`nu in [0.9, 1.1]` -- it blows up toward `nu = 1`, unconditionally (seed/shape-robust).

The fix is the order-dependent functional built on the Hermitian-norm probability
(positive, partition `-> 1` as `nu -> 1`, which cancels the pole). Measured: `renyi_H` runs
`4.107, 4.031, 4.023, 4.022, 4.014, 3.940` -- smooth through `nu = 1`, recovering the Shannon
value `4.0222` there.

Caveat (carried, not hidden): the `nu -> 1` recovery is an ANALYTIC limit (L'Hopital, exact,
seed-robust). Numerically in float32 it is stable only at a stand-off `|1-nu| ~ 1e-3..1e-2`;
within ~1e-6 of 1 it is roundoff-unstable and NaN at `nu = 1`. Use a stand-off, float64, or
special-case `nu = 1 = Shannon`.

So-what: this is the basis for the manuscript's detector retraction. The well-posed Born cost
is the Hermitian closure, not the lens form.

### 4. The clamp inflation, documented (`exp_five_way_static.py`) -- diagnostic

On the static single-snapshot field (which carries the "wrong field" caveat -- it is not the
closed-loop fixed point), swept over `nu_bar`: at the natural `nu_bar = 1.18` the detector
correlates with surprisal at `det_raw = 0.789` but `det_clamped = 0.985`. The Shannon
baseline floor is `-0.828` (i.e. ~0.83 in magnitude). The clamp `(1-nu_bar).clamp(min=1e-6)`
strips the divergent wrapper and reports the inner integrand's value as the detector's.

So-what: the previously cited ~0.98 detector-surprisal correlation was the clamped number.
The honest (raw) number sits at the level of the plain baseline. This is the evidence behind
the manuscript correction.

### 5. Memory kernel is asymptotically a fractional integral (`exp_fracint_check.py`) -- PASS

The CAL memory kernel `(1+tau)^(-alpha_K)` reproduces the Riemann-Liouville `I^mu` with
`mu = 1 - alpha_K`. On `f(t) = t^2`, the fitted exponent shift matches: `p_err = 0.007,
0.022, 0.054` for `alpha_K = 0.3, 0.5, 0.7`; the semigroup (composition adds orders) holds.
The match is asymptotic (the `1+tau` shift regularizes `tau = 0`), and the residual is
reported, not hidden. `alpha_K < 1` is an integral, `= 1` the marginal case, `> 1` a
derivative -- the fractional interpolation the framework claims.

So-what: "how much of the past contributes to the present" is literally a fractional-integral
order, validating the kernel interpretation.

### 6. Spin double cover reproduced (`exp_double_cover_repro.py`) -- PASS

The period-2 modulus component and the period-4 sign-flipping interference component
reproduce: the period-4 correlation flips `+0.436` (alpha = 0) to `-0.436` (alpha = 2), and
the period-2 component zeros at integer alpha. `modulus = (interference)^2` is the
`Spin(2) -> SO(2)` squaring map. The period-4 CORRELATION's extremum is offset from alpha = 2
(driven by `cov(Im a, E)`); this is a fit-quality property of the diagnostic curve and is
captured by the manuscript's stated residual -- it does NOT move the closure, which sits at
the antiparallel `e^{i pi} = -1` sign flip at alpha = 2.

So-what: the `e^{i pi} = -1` sign flip at alpha = 2 IS the double-cover marker; its necessity
is the closure mechanism, not a triviality.

### 7. Expectancy / biased-random-walk model (`exp_expectancy_walk.py`) -- PASS

A "ride" as a biased random walk on a depleting lattice, with feedback and bias-update
learning, under incomplete information:

- LEARNING improves transport: cumulative reward `1573.6` (learn ON) vs `1116.0` (frozen
  prior control) -- a 41% improvement. The within-run decline is the depletion coupling
  (overcommitment erodes the exploited waves), present in both arms.
- NON-MARKOV: `KL(P_next | early-pi || late-pi) = 0.188 > 0` at fixed states -- the
  transition depends on history, not just position.
- MANY-TO-ONE: high path diversity (mean pairwise Jaccard distance `0.416`) but convergent
  outcome (across-ride reward `CV = 0.102`). Diverse rides, similar reads.

So-what: the detector is a read along a ride (a biased-walk decision system), not a fixed
point. The emergent stochasticity is epistemic (stochastic policy + incomplete info) on a
deterministic landscape.

### 8. Feedback alone is NOT deterministic chaos (`exp_feedback_necessity.py`) -- honest negative

Two identical deterministic loops, the only manipulated variable being self-referential
back-coupling ON vs OFF, swept over decision sharpness `beta in [2, 32]`. No `beta` produces
the confirming split (ON diverges while OFF stays predictable): every row reads "neither
(predictable)". Determinism check passed (`max|trajA - trajB| = 0`).

So-what: the unpredictability in the framework is NOT sensitive-dependence/chaos from the
feedback term. This negative is what redirected the model to the epistemic
expectancy/biased-walk account (test 7). Reported as a negative, not reframed into a pass.

## Summary

Confirmed: the two-closure algebra (1), holonomy lift off the null cone (2), the detector
degeneracy and its Hermitian-closure fix (3, 4), the fractional-integral kernel (5), the spin
double cover (6), and the expectancy/biased-walk decision model (7). One honest negative (8)
that sharpened the randomness account rather than supporting a chaos claim. The central
detector-as-lens-cost claim is retracted in the manuscript on the basis of (3) and (4); the
corrected object is the Hermitian closure of (1).
