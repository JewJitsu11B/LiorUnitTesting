import ChiralParity.GeneralStar

/-!
# Paired-grade chiral operator and grade-parity criterion (Proposition 4.3)

Builds on `GeneralStar.gstar_sq` (Lemma 3.1, general grade: `⋆∘⋆ = (-1)^{k·j+q} · id`).

* `gscalar_symm` — the star-square scalar is symmetric in the two grades.
* `gZ` — the paired-grade chiral operator `Z` on `Λᵏ ⊕ Λʲ` (Definition 2.4), block-antidiagonal.
* `gZ_sq` — Proposition 4.3, eq. (6): `Z² = (-1)^{k·j+q} · id` on `Λᵏ ⊕ Λʲ`.
* `star_is_endo_iff` — Prop 4.3(c): the star is a single-grade endomorphism iff the grade is
  the middle grade (`k = j ↔ 2k = n`).
* `chiral_grade_parity_n8` — Prop 4.3(a),(b) at `n = 8`: the chiral pair exists (scalar `= +1`)
  iff the grade `k` and the signature count `q = |S|` share the same parity.
-/

namespace ChiralParity

open Matrix Finset

variable {n : ℕ}

/-- Item 1: the star-square scalar is symmetric in its two grade arguments. -/
theorem gscalar_symm (n k j : ℕ) (S : Finset (Fin n)) :
    gscalar n j k S = gscalar n k j S := by
  unfold gscalar
  rw [Nat.mul_comm]

/-- Item 2 (Definition 2.4): the paired-grade chiral operator `Z` on `Λᵏ ⊕ Λʲ`,
block-antidiagonal `Z(α, β) = (⋆β, ⋆α)`. -/
def gZ (n k j : ℕ) (S : Finset (Fin n)) :
    Matrix (GBlade n k ⊕ GBlade n j) (GBlade n k ⊕ GBlade n j) ℚ :=
  Matrix.fromBlocks 0 (gstar n k j S) (gstar n j k S) 0

/-- Item 3 — MAIN (Proposition 4.3, eq. (6)): `Z² = (-1)^{k·j+q} · id` on `Λᵏ ⊕ Λʲ`. -/
theorem gZ_sq {k j : ℕ} (hkj : k + j = n) (S : Finset (Fin n)) :
    gZ n k j S * gZ n k j S
      = gscalar n k j S • (1 : Matrix (GBlade n k ⊕ GBlade n j) (GBlade n k ⊕ GBlade n j) ℚ) := by
  unfold gZ
  rw [Matrix.fromBlocks_multiply]
  -- Kill the zero blocks / zero summands; only the two star-products survive on the diagonal.
  simp only [Matrix.mul_zero, Matrix.zero_mul, add_zero, zero_add]
  -- Top-left diagonal block: ⋆∘⋆ on Λᵏ.
  rw [gstar_sq hkj S]
  -- Bottom-right diagonal block: ⋆∘⋆ on Λʲ, scalar rewritten by symmetry.
  rw [gstar_sq (show j + k = n by omega) S, gscalar_symm]
  -- Reduce the RHS `1` to blocks and push the scalar through.
  rw [← Matrix.fromBlocks_one, Matrix.fromBlocks_smul]
  simp only [smul_zero]

/-- Item 4 — Prop 4.3(c): the star `⋆ : Λᵏ → Λʲ` is a single-grade endomorphism iff `k = j`,
i.e. iff `k` is the middle grade `2k = n`. -/
theorem star_is_endo_iff {k j : ℕ} (hkj : k + j = n) : k = j ↔ 2 * k = n := by
  omega

/-- Item 5 — Prop 4.3(a),(b) at `n = 8`: the chiral pair exists (star-square scalar `= +1`)
iff the grade `k` and the signature count `q = |S|` share the same parity.
(Even `q` ⇒ even grades; odd `q` ⇒ odd grades.) -/
theorem chiral_grade_parity_n8 {k j : ℕ} (hkj : k + j = 8) (S : Finset (Fin 8)) :
    gscalar 8 k j S = 1 ↔ (Even k ↔ Even (S.card)) := by
  unfold gscalar
  rw [neg_one_pow_eq_one_iff_even (by norm_num : (-1 : ℚ) ≠ 1), Nat.even_add]
  -- `k` and `j` have equal parity because `k + j = 8` is even.
  have hj : Even k ↔ Even j := by
    have h8 : Even (k + j) := by rw [hkj]; decide
    rwa [Nat.even_add] at h8
  -- `k·j` is even iff `k` is even (given equal parities of `k` and `j`).
  have hmul : Even (k * j) ↔ Even k := by
    rw [Nat.even_mul]
    constructor
    · rintro (h | h)
      · exact h
      · exact hj.mpr h
    · intro h; exact Or.inl h
  rw [hmul]

end ChiralParity
