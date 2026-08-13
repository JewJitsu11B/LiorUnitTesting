# Born rule from biquaternionic source closure (alpha = 2) — Lean 4 formalization

Machine-checked formalization of the null-idempotent source-closure core of
**"A Biquaternionic Source-Closure Derivation of the Born Rule at the delta-Kernel Limit and
Closure Order alpha = 2 of the Causal Accumulation Law"** (S. Leizerman, Aug 2026 — the source PDF
`born axiomatic updated.pdf` is in this folder).

Carrier: `C (x) H = M2(C)`. Lean 4 (`v4.14.0`) + Mathlib.

## What is proved (all `sorry`-free; standard axioms only)

- **Carrier / conjugations** (`BornRule/Carrier.lean`, `Basic.lean`): `M2(C)`; dagger = conjugate
  transpose, bar = adjugate (quaternion reversal), reduced norm = det; quaternion-unit algebra
  (`qi^2=qj^2=qk^2=-1`, `e_n^2 = -||n||^2`, `E_n^2 = +1` on the unit sphere).
- **Lemma 4.1** (`L41`): reduced norm multiplicative (`Matrix.det_mul`); the null divisor `1 + i.i`
  has `N = 0` while the dagger form is positive; dagger form positive **definite**
  (`Corollaries.dagger_form_trace_pos`).
- **Lemma 4.3** (`L43`): null idempotents — `E_n^dag = E_n`, `J_pm^2 = 2 J_pm`, `N(J_pm) = 0`, and
  `P_pm^2 = P_pm`, `P_pm^dag = P_pm`, `P_+ P_- = 0`, `P_+ + P_- = 1`.
- **Lemma 4.5** (`L45`): interference cancellation `W^dag W = g^2 + t^2`; commutator traceless.
- **Prop 5.2** (`P52`): the alpha = 2 closure `J J^dag = 2 J` forces the phase `lambda = +/- 1`
  (the branch collapse — the dynamical phase selection).
- **Theorem 6.1** (`T61`): Born readout `B(P_+(m) P_+(n)) = (1 + m.n)/2 = cos^2(theta/2)`, and the
  two outcomes sum to 1; plus (`Corollaries.born_prob_mem_Icc`) `0 <= P <= 1` under unit hypotheses.

## Build and verify

```bash
lake exe cache get            # fetch Mathlib oleans (first time only)
lake build BornRule           # full build
bash audit.sh                 # no-cheating grep gate (sorry/admit/native_decide/uncited axiom)
lake build BornRule.Audit     # emits `#print axioms` for every capstone
bash check.sh BornRule/<Module>.lean   # lock-free single-file type-check
```

Acceptance gate: full build clean; no `sorry`/`admit`/`native_decide`/uncited `axiom`;
`#print axioms` on every capstone shows only `[propext, Classical.choice, Quot.sound]`.

## Provenance

Foundation (`Carrier`, `Basic`) hand-authored and verified. The lemma modules were written by a
parallel team of prover agents, then adversarially reviewed (faithfulness audit + non-vacuity /
oracle cross-check); the review findings were closed in `Corollaries.lean`. Full ledger in
`CONVERGENCE_REPORT.md`. `BornRule/ReviewCheck.lean` cross-checks concrete Born values against the
paper's exact-rational Python harness (`1`, `1/2`, `0`, `9/10`).
