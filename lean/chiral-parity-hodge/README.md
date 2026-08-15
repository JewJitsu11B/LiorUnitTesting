# Parity criterion for native chiral projectors — Lean 4 formalization

Machine-checked formalization of the middle-grade Hodge-duality core of
**"The Parity Criterion for Native Chiral Projectors: Middle-Grade Hodge Duality on the
Biquaternionic Carrier"** (S. Leizerman, 2026 — source PDF `chiral parity 1.pdf` in this folder).

Model: the MIDDLE grade `k = m` of an `n = 2m`-dimensional carrier. Blades = `m`-element subsets of
`Fin (2m)`; the Hodge star is an explicit `ℚ`-matrix whose shuffle sign `sgn(I, Iᶜ)` is realized as
an INVERSION COUNT `inv I = #{(a,b) ∈ I × Iᶜ : a > b}` (proved equal to the permutation sign by the
oracle, `chiral_oracle.py`). Lean 4 (`v4.14.0`) + Mathlib.

## What is proved (all `sorry`-free; standard axioms only)

- **Foundation** (`ChiralParity/Carrier.lean`): blades, signature `ε_I = (-1)^{|I∩S|}`,
  inversion-count sign, star matrix `⋆`, projectors `P± = ½(id ± ⋆)`.
- **Lemma 3.1** (`StarSquare`): `⋆² = (-1)^{m²+q}·id`, via the two clean Finset facts
  `inv I + inv Iᶜ = m²` and `ε_I · ε_{Iᶜ} = (-1)^q`.
- **Theorem 3.2** (`ParityCriterion`): `P±` is a chiral pair (`P₊²=P₊`, `P₊P₋=0`) iff `m²+q` even;
  and when odd, the EXACT failure constants `P₊² − P₊ = −½·1` and `P₊P₋ = ½·1`; plus the parity
  characterization `starScalar = 1 ↔ Even (m²+q)`.
- **Corollary 3.3**: at `n = 8` (`m = 4`), chiral pair iff `q` even.
- **Proposition 4.1** (`EqualSplit`, `RankSplit`): `tr ⋆ = 0` (zero diagonal), the equal split
  `tr P₊ = tr P₋`, and `P₊ + P₋ = 1`, `P₊ − P₋ = ⋆`; and the concrete DIMENSION split at `n = 8`,
  `rank P₊ = rank P₋ = 35`, `rank P₊ + rank P₋ = 70` (`RankSplit`, via `rank = trace` for
  ℚ-idempotents, `LinearMap.IsProj.trace`).
- **Corollary 9.7 / Theorem 9.10** (`Forcing`, `Section9`): for even `n`, Majorana–Weyl
  `⟺ p − q ≡ 0 (mod 8)`; and in `n = 8`, Weyl ∧ Majorana ∧ (nonempty light cone) forces
  `(p,q) = (4,4)` uniquely. The **Weyl condition is proved computably** from the Clifford
  volume-element sign `ω² = (-1)^{n(n-1)/2+q}` (`weyl_iff_wsign`, no axiom); the Majorana /
  representation-type side is a single explicit **cited axiom** (`majorana_iff_realstructure`,
  Atiyah–Bott–Shapiro, which Mathlib lacks and the paper itself cites), and
  `forcing_n8_from_clifford` rebuilds the forcing on top. Anchor checks for `(4,4)`, `(6,2)`,
  `(2,6)`, `(8,0)`, `(0,8)` included.
- **Corollary 8.1** (`Corollary8`): the classical 4d case (`m=2`, `Λ²(ℝ⁴)`): `⋆²=+id` with the real
  chiral split `6 = 3 ⊕ 3` in Euclidean/split signature, and `⋆²=−id` (a complex structure) in
  Lorentzian signature.
- **Proposition 5.1 / Corollary 5.2** (`BlockEigen`): the block volume forms `ω₁=e_{0123}`,
  `ω₂=e_{4567}`; `⋆ω₁=σ₁ω₂`, `⋆ω₂=σ₂ω₁` with `σᵢ=(-1)^{qᵢ}`; `ω₊=ω₁+ω₂` is a genuine `⋆`-eigenvector
  (`star_eigen`); and the handedness label depends on `q₁` (the block split), a declared convention.
- **General grade** (`General`, `GeneralStar`, `PairedGrade`): Lemma 3.1 for an ARBITRARY grade `k`
  and its complement `j = n − k` — `⋆∘⋆ = (-1)^{k(n-k)+q}·id` (`gstar_sq`); Proposition 4.3 — the
  paired operator `Z` on `Λᵏ ⊕ Λʲ` with `Z² = (-1)^{k(n-k)+q}·id` (`gZ_sq`), the middle grade is the
  unique self-paired grade (`star_is_endo_iff`, `k=j ⟺ 2k=n`), and at `n=8` the chiral pair exists at
  grade `k` iff `k` and `q` have the same parity (`chiral_grade_parity_n8`).

## Oracle cross-check (`chiral_oracle.py`)

Independent Python computation (replicating the paper's harness spec) confirms, at `n = 8` for all
`q = 0..8`: the Lean inversion-count sign EQUALS the paper's permutation-shuffle sign;
`⋆² = (-1)^{m²+q}·I`; and the ranks `35/35` (q even, chiral pair) vs `70/70` (q odd, failure) —
matching the paper's Table 1 (`dim Λ⁴(ℝ⁸) = 70`) with zero disagreements.

## Build and verify

```bash
lake exe cache get            # fetch Mathlib oleans (first time only)
lake build ChiralParity       # full build
bash audit.sh                 # no-cheating grep gate
lake build ChiralParity.Audit # emits `#print axioms` for every capstone
bash check.sh ChiralParity/<Module>.lean   # lock-free single-file type-check
python chiral_oracle.py       # independent numeric oracle (n=8 ledger)
```

Gate: full build clean; no `sorry`/`admit`/`native_decide`/uncited `axiom`; `#print axioms` on
every capstone shows only `[propext, Classical.choice, Quot.sound]`.

## Scope

Formalizes Sections 3, 4, 5, 8, the general-grade Lemma 3.1 and Proposition 4.3, and the mod-8
signature forcing (Cor 9.7 / Thm 9.10). The ONLY piece not proved from first principles is:
- **Lemma 9.4 (the Atiyah–Bott–Shapiro classification of `Cl(p,q)` as matrix algebras over
  ℝ/ℂ/ℍ)** — Mathlib v4.14.0 does not have it, and a from-scratch proof is a months-scale
  development; the paper itself cites it. It is represented here as the single `CITED-AXIOM`
  `majorana_iff_realstructure`, so the `(4,4)` forcing's dependence on it is explicit rather than
  silent. The Weyl half of that bridge IS proved from first principles (`weyl_iff_wsign`).

Everything else — including the general-grade star-square and the paired-grade operator, which were
built on a `k`/`j` (`k + j = n`) carrier so no `n − k` subtraction appears in the types — is proved
on only `[propext, Classical.choice, Quot.sound]`.

## Provenance

Foundation (`Carrier`) hand-authored and cross-checked by the oracle. The lemma modules were written
by a parallel team of prover agents and adversarially reviewed. See `CONVERGENCE_REPORT.md`.
