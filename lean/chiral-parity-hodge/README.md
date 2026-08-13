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
- **Proposition 4.1** (`EqualSplit`): `tr ⋆ = 0` (zero diagonal), the equal split
  `tr P₊ = tr P₋`, and `P₊ + P₋ = 1`, `P₊ − P₋ = ⋆`.
- **Corollary 9.7 / Theorem 9.10** (`Forcing`): for even `n`, Majorana–Weyl `⟺ p − q ≡ 0 (mod 8)`;
  and in `n = 8`, Weyl ∧ Majorana ∧ (nonempty light cone) forces `(p,q) = (4,4)` uniquely
  (`omega`). Anchor checks for `(4,4)`, `(6,2)`, `(2,6)`, `(8,0)`, `(0,8)` included.

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

Formalizes the middle-grade parity criterion (Sections 3–4) and the mod-8 forcing (Cor 9.7 / Thm
9.10). NOT formalized: the general-grade Lemma 3.1 (done at middle grade `k = m`), Prop 4.3
(paired-grade extension), Prop 5.1 (block eigenvectors / handedness), and the Weyl/Majorana
Clifford-classification bridge (Lemma 9.4), which the paper itself cites from Atiyah–Bott–Shapiro.
Proposition 4.1's "equal split" is formalized at the level of equal TRACE (`tr P₊ = tr P₋`); the
rank-equality `35 = 35` follows over ℚ since rank = trace for idempotents, but the rank statement
itself is not separately formalized.

## Provenance

Foundation (`Carrier`) hand-authored and cross-checked by the oracle. The lemma modules were written
by a parallel team of prover agents and adversarially reviewed. See `CONVERGENCE_REPORT.md`.
