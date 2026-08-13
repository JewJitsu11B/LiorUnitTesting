# Convergence report — chiral-parity (middle-grade Hodge duality) Lean formalization

Multi-agent write + adversarial review. Target manuscript: `chiral parity 1.pdf`.
Model: middle grade `k = m` of `n = 2m`, blades = `m`-subsets of `Fin (2m)`, `⋆` as an explicit
`ℚ`-matrix with inversion-count sign. Lean 4 v4.14.0 + Mathlib.

## Foundation (hand-authored, oracle-cross-checked)
- `Carrier.lean` — blades, `eps`, `inv` (inversion count), `starSign`, `starMat`, `starScalar`, `P±`.

## Provers (parallel, isolated; each barred from sorry / axiom / native_decide)
| Module | Content | Result |
|---|---|---|
| `StarSquare` | Lemma 3.1: `⋆² = (-1)^{m²+q}·id`; helpers `inv I + inv Iᶜ = m²`, `ε_I·ε_{Iᶜ} = (-1)^q` | clean, 3 lemmas |
| `EqualSplit` | Prop 4.1: `tr ⋆ = 0`, `tr P₊ = tr P₋`, `P₊+P₋=1`, `P₊−P₋=⋆` | clean, 5 lemmas |
| `ParityCriterion` | Thm 3.2: chiral pair iff `m²+q` even; exact failure constants `−½`, `+½`; Cor 3.3 | clean, 7 lemmas |
| `Forcing` | Cor 9.7 (`MW ⟺ p−q≡0 mod 8`); Thm 9.10 (`(4,4)` forced in n=8); 6 anchor checks | clean, 4 + 6 examples |

## No-cheating gate
- grep gate (`audit.sh`): no `sorry` / `admit` / `native_decide` / uncited `axiom`.
- `#print axioms` (`ChiralParity/Audit.lean`) on all 14 capstones: each depends on exactly
  `[propext, Classical.choice, Quot.sound]`. No placeholder axiom.

## Oracle cross-check (`chiral_oracle.py`, independent of the paper's harness)
For `n = 8`, all `q = 0..8`:
- The Lean model's INVERSION-COUNT sign equals the paper's PERMUTATION-shuffle sign (`sign_match`
  True for every q) — validating that `starMat` faithfully implements Convention 2.2.
- `⋆² = (-1)^{m²+q}·I` holds (`starScalar` correct).
- Ranks `35/35` (q even, chiral pair) vs `70/70` (q odd, failure mode) — matches Table 1.

## Adversarial faithfulness review
See the session ledger; the review checked star-entry placement, the parity-criterion both
directions, the exact `±½` failure constants, hypothesis satisfiability, and the Prop 4.1
trace-vs-rank scope.

## Scope and honesty
- Formalized: middle-grade Lemma 3.1, Theorem 3.2 (+ exact failure constants), Corollary 3.3,
  Proposition 4.1 (trace-level equal split), Corollary 9.7 and Theorem 9.10 (the `(4,4)` forcing).
- NOT formalized: general-grade Lemma 3.1 (only `k = m`), Prop 4.3 (paired-grade extension),
  Prop 5.1 (block eigenvectors / handedness label), and Lemma 9.4 (the trace-form → R/C/H type
  discriminator), which rests on the Atiyah–Bott–Shapiro Clifford classification the paper itself
  cites rather than proves. Prop 4.1's rank-equality (`35 = 35`) is captured at the trace level
  (`tr P₊ = tr P₋`); rank = trace for idempotents over ℚ, but the rank statement is not separately
  formalized.
