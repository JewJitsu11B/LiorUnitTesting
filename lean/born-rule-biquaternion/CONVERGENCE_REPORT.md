# Convergence report — Born-rule (alpha = 2) Lean formalization

Multi-agent write + adversarial review. Target manuscript: `born axiomatic updated.pdf`.
Carrier `M2(C)`, Lean 4 v4.14.0 + Mathlib.

## Foundation (hand-authored, verified)
- `Carrier.lean` — `M2(C)`, dagger/bar/redNorm, `en/En/Jbranch/Pbranch/traceRead`, `IsUnit3`.
- `Basic.lean` — quaternion-unit algebra; `en^2 = -||n||^2`, `En^2 = +1` on the unit sphere.

## Provers (parallel, isolated; each barred from sorry / axiom / native_decide)
| Module | Content | Result |
|---|---|---|
| `L41_Complementarity` | reduced norm multiplicative; null divisor; dagger form PosSemidef + trace >= 0 | clean, 5 lemmas |
| `L43_NullIdempotents` | `E_n^dag=E_n`, `J^2=2J`, `N(J)=0`, `P^2=P`, `P^dag=P`, `P_+P_-=0`, `P_++P_-=1` | clean, 10 + 3 helpers |
| `L45_Interference` | commutator traceless; `W^dag W = g^2 + t^2`; dagger-form trace real | clean, 4 lemmas |
| `P52_Closure` | closure `J J^dag = 2J` forces `lambda = +/- 1` (branch collapse) | clean, 4 lemmas incl. the collapse |
| `T61_Born` | `trace(E_m E_n)=2(m.n)`; `B(P_+ P_+)=(1+m.n)/2`; outcomes sum to 1 | clean, 5 lemmas |

## No-cheating gate
- grep gate (`audit.sh`): no `sorry` / `admit` / `native_decide` / uncited `axiom`.
- `#print axioms` (`BornRule/Audit.lean`) on all 17 capstones + corollaries: each depends on
  exactly `[propext, Classical.choice, Quot.sound]`. No `sorryAx`.

## Adversarial review
- **Faithfulness reviewer** — confirmed with compiled `example`s that the definitions match the
  paper (Def 2.4 / 2.7): dagger conjugates the central `i` and negates units; bar FIXES the central
  `i` and negates units (`bar(i.1) != dagger(i.1)`, exactly as Def 2.4 requires); `traceRead = Re tr`
  equals `B(q) = 2 Re q0`; the units are genuinely Hamiltonian (`qi qj = qk`, etc.). Verdict:
  substantially faithful, no wrong or vacuous statements.
- **Non-vacuity + oracle reviewer** — hypotheses satisfiable at explicit witnesses; the Born theorem
  instantiated at concrete vectors reproduces the exact-rational Python oracle: parallel `1`,
  orthogonal `1/2`, antipodal `0`, `(3/5,0,4/5).zhat = 9/10`. (`BornRule/ReviewCheck.lean`.)

## Findings and resolution
- **A** — `born_plus_plus` lacked unit hypotheses, so it is a correct trace identity but not enforced
  as a probability (yields `2` off the unit sphere). CLOSED: `Corollaries.dot_bounds`
  (`-1 <= m.n <= 1` for unit vectors) + `Corollaries.born_prob_mem_Icc` (`0 <= P <= 1`).
- **B** — Lemma 4.1(b) "positive definite" was only formalized as PosSemidef. CLOSED:
  `Corollaries.dagger_form_trace_pos` (`q != 0 -> 0 < <q,q>`).
- **C** — `<q,q> = sum |q_n|^2` is realized as a Frobenius entry-sum rather than the
  biquaternion-coefficient sum; the two agree up to basis normalization. DOCUMENTED (minor,
  no code change).

## Scope and honesty
- Formalized: the algebraic alpha = 2 source-closure core — Def 2.3-2.7, Lemma 4.1 / 4.3 / 4.5,
  Prop 5.2, Theorem 6.1 (trace-readout Born rule) and the probability bound.
- NOT formalized (out of current scope): Prop 5.3 (rotor / state-vector equivalence, the sandwich
  presentation), and the physical Relations 3.1-3.3 (parent CAL accumulation, channel-wise source
  embedding), which are inputs rather than theorems.
- Lemma 4.4 (the STATIC "unique conjugation-invariant quadratic form") is deliberately NOT
  formalized: its load-bearing content is the dynamical branch collapse Prop 5.2 (which IS proved),
  and the static wording is a known over-statement (the space of both-involution-invariant forms is
  14-dimensional). The Born rule here rests on Prop 5.2 + Thm 6.1, not on Lemma 4.4.
