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
| `RankSplit` | Prop 4.1 concrete: `card (Blade 4) = 70`, `rank P₊ = rank P₋ = 35`, `70 = 35 ⊕ 35`, via `rank = trace` for ℚ-idempotents (`LinearMap.IsProj.trace`) | clean, standard axioms |
| `Forcing` | Cor 9.7 (`MW ⟺ p−q≡0 mod 8`); Thm 9.10 (`(4,4)` forced in n=8); 6 anchor checks | clean, 4 + 6 examples |
| `Section9` | Weyl condition proved computably from `ω²=(-1)^{n(n-1)/2+q}` (`weyl_iff_wsign`, no axiom); one `CITED-AXIOM` (ABS classification) for the Majorana/type side; `forcing_n8_from_clifford` on top | clean; 1 cited axiom |
| `Corollary8` | Cor 8.1 (4d case, m=2): real split `6 = 3 ⊕ 3` (Euclid/split), `⋆²=-1` complex structure (Lorentzian) | clean, 7 lemmas |
| `BlockEigen` | Prop 5.1 / Cor 5.2: block forms, `⋆ω₁=σ₁ω₂`, `ω₊` a genuine `⋆`-eigenvector, block-dependent handedness | clean, standard axioms |
| `General` / `GeneralStar` | general-grade foundation and Lemma 3.1 for arbitrary `k`, `j` (`k+j=n`): `⋆∘⋆ = (-1)^{k·j+q}·id` (`gstar_sq`) | clean, standard axioms |
| `PairedGrade` | Prop 4.3: paired operator `Z` on `Λᵏ ⊕ Λʲ`, `Z² = (-1)^{k·j+q}·id` (`gZ_sq`); middle-grade uniqueness; n=8 grade-parity rule | clean, standard axioms |

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
  Proposition 4.1 INCLUDING the concrete rank split `70 = 35 ⊕ 35` (`RankSplit`), Corollary 9.7 and
  Theorem 9.10 (the `(4,4)` forcing), and the computable Weyl reduction from the volume-element sign
  (`Section9.weyl_iff_wsign`).
- Represented as ONE cited axiom: Lemma 9.4 (trace-form → R/C/H representation type = the
  Atiyah–Bott–Shapiro classification of `Cl(p,q)`), which Mathlib v4.14.0 lacks and the paper itself
  cites. `Section9.majorana_iff_realstructure` is tagged `CITED-AXIOM`; `forcing_n8_from_clifford`
  legitimately shows it in `#print axioms`. The Weyl half of that bridge is proved, not cited.
- Also formalized: Corollary 8.1 (classical 4d case, `Corollary8`), Proposition 5.1 / Cor 5.2
  (block eigenvectors and handedness, `BlockEigen`), and the GENERAL-grade Lemma 3.1 + Proposition
  4.3 (`General`, `GeneralStar`, `PairedGrade`) on a `k`/`j` (`k+j=n`) carrier.
- The ONLY piece not proved from first principles: Lemma 9.4 (the Atiyah–Bott–Shapiro
  classification), a single explicit `CITED-AXIOM`; its Weyl half is proved (`weyl_iff_wsign`).
  Sections 3, 4, 5, 8, the general grade, and the forcing are otherwise complete.
