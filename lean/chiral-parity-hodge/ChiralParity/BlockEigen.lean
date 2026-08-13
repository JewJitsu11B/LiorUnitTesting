import ChiralParity.StarSquare

/-!
# Proposition 5.1 / Corollary 5.2 (block volume forms and the handedness label)

At the middle grade `m = 4` (carrier `Λ⁴(ℝ⁸)`), split `Fin 8` into two blocks
`B₁ = {0,1,2,3}` and `B₂ = {4,5,6,7}`.  The block volume forms are `ω₁ = e_{B₁}`,
`ω₂ = e_{B₂}`, and since `B₂ = B₁ᶜ` the Hodge star sends each block form to the other:
`⋆ ω₁ = σ₁ ω₂`, `⋆ ω₂ = σ₂ ω₁`.

Proposition 5.1 computes the shuffle signs: `sgn(B₁, B₁ᶜ) = +1` (inversion count `inv B₁ = 0`,
every element of `B₁` is below every element of `B₂`) and `sgn(B₂, B₂ᶜ) = +1`
(`inv B₂ = 16`, even).  Hence `σ₁ = ε_{B₁} = (-1)^{q₁}` and `σ₂ = ε_{B₂} = (-1)^{q₂}`,
with `q₁ = |S ∩ B₁|`, `q₂ = |S ∩ B₂|`.

Corollary 5.2 (the declared handedness convention): the label attached to `ω₊ = ω₁ + ω₂`
is `ζ = σ₁`, which depends on `q₁` — the split of negative directions into the FIRST block —
not merely on the total `q`.
-/

namespace ChiralParity

open Matrix Finset

/-- First block `B₁ = {0,1,2,3} ⊂ Fin 8` as a middle-grade blade. -/
def B1 : Blade 4 := ⟨{0, 1, 2, 3}, by decide⟩

/-- Second block `B₂ = {4,5,6,7} ⊂ Fin 8` as a middle-grade blade. -/
def B2 : Blade 4 := ⟨{4, 5, 6, 7}, by decide⟩

/-- Item 1: the second block is the complement of the first, `B₂ = B₁ᶜ`. -/
theorem B2_eq_compl :
    (B2 : Finset (Fin (2 * 4))) = (B1 : Finset (Fin (2 * 4)))ᶜ := by decide

/-- The first block is the complement of the second, `B₁ = B₂ᶜ`. -/
theorem B1_eq_compl :
    (B1 : Finset (Fin (2 * 4))) = (B2 : Finset (Fin (2 * 4)))ᶜ := by decide

/-- Item 2 (Prop 5.1): the inversion count of `(B₁, B₁ᶜ)` is `0` — every element of `B₁`
lies below every element of `B₂`, so the shuffle sign `sgn(B₁, B₁ᶜ) = +1`. -/
theorem inv_B1 : inv (B1 : Finset (Fin (2 * 4))) = 0 := by decide

/-- Item 3 (Prop 5.1): the inversion count of `(B₂, B₂ᶜ)` is even (`inv B₂ = 16`), so the
shuffle sign `sgn(B₂, B₂ᶜ) = +1`. -/
theorem inv_B2_even : Even (inv (B2 : Finset (Fin (2 * 4)))) := by decide

/-- Item 4 (Prop 5.1): the star of the first block form. Because `sgn(B₁, B₁ᶜ) = +1`,
`⋆ ω₁ = σ₁ ω₂` with `σ₁ = ε_{B₁}`. -/
theorem star_ω1 (S : Finset (Fin (2 * 4))) :
    starMat S B1 B2 = ((eps S {0, 1, 2, 3} : ℤ) : ℚ) := by
  unfold starMat
  rw [if_pos B2_eq_compl]
  unfold starSign
  rw [inv_B1, pow_zero, mul_one]
  rfl

/-- Item 5 (Prop 5.1): the star of the second block form. Because `sgn(B₂, B₂ᶜ) = +1`
(inversion count even), `⋆ ω₂ = σ₂ ω₁` with `σ₂ = ε_{B₂}`. -/
theorem star_ω2 (S : Finset (Fin (2 * 4))) :
    starMat S B2 B1 = ((eps S {4, 5, 6, 7} : ℤ) : ℚ) := by
  unfold starMat
  rw [if_pos B1_eq_compl]
  unfold starSign
  rw [Even.neg_one_pow inv_B2_even, mul_one]
  rfl

/-- Item 6 (Cor 5.2): the first block sign is `σ₁ = (-1)^{q₁}` with `q₁ = |S ∩ B₁|`
the number of negative directions in the FIRST block. -/
theorem eps_B1_eq (S : Finset (Fin (2 * 4))) :
    eps S {0, 1, 2, 3} = (-1) ^ ((S ∩ {0, 1, 2, 3}).card) := by
  unfold eps
  rw [Finset.inter_comm]

/-- Item 6 (Cor 5.2): the second block sign is `σ₂ = (-1)^{q₂}` with `q₂ = |S ∩ B₂|`. -/
theorem eps_B2_eq (S : Finset (Fin (2 * 4))) :
    eps S {4, 5, 6, 7} = (-1) ^ ((S ∩ {4, 5, 6, 7}).card) := by
  unfold eps
  rw [Finset.inter_comm]

/-- Item 7 (Cor 5.2, handedness convention): the star entry on the block forms — the
handedness label carried by `ω₊ = ω₁ + ω₂` — is `(-1)^{q₁}`, depending on the split of
negative directions into the first block, not on the total `q`. -/
theorem handedness_depends_on_block (S : Finset (Fin (2 * 4))) :
    starMat S B1 B2 = ((-1 : ℚ)) ^ ((S ∩ {0, 1, 2, 3}).card) := by
  rw [star_ω1, eps_B1_eq]
  push_cast
  ring

/-!
## Optional: `ω₊ = ω₁ + ω₂` as an eigenvector of `⋆`

The combined block form `ω₊ = e_{B₁} + e_{B₂}` (as a coordinate vector on blades) is an
eigenvector of the star matrix acting by `mulVec`, with eigenvalue `σ₁ = ε_{B₁}`, provided
`σ₁ = σ₂` (the even-`q` handedness convention of Corollary 5.2). Off the two block indices
the star matrix carries the block forms only onto each other, so the remaining coordinates
vanish.
-/

/-- The combined block volume form `ω₊ = ω₁ + ω₂ = e_{B₁} + e_{B₂}` as a coordinate vector. -/
def omegaPlus : Blade 4 → ℚ := fun I => if I = B1 then 1 else if I = B2 then 1 else 0

/-- `ω₊` is the sum of the two standard block basis vectors. -/
theorem omegaPlus_eq :
    omegaPlus = Pi.single B1 (1 : ℚ) + Pi.single B2 (1 : ℚ) := by
  funext I
  simp only [omegaPlus, Pi.add_apply, Pi.single_apply]
  by_cases h1 : I = B1
  · subst h1
    have hne : B1 ≠ B2 := by decide
    simp [hne]
  · by_cases h2 : I = B2
    · subst h2
      have hne : B2 ≠ B1 := by decide
      simp [hne]
    · simp [h1, h2]

/-- Corollary 5.2 (eigenvector packaging): when `σ₁ = σ₂` (even total `q`), the combined
block form `ω₊ = ω₁ + ω₂` is an eigenvector of the middle-grade Hodge star with eigenvalue
`σ₁ = ε_{B₁}`.  The handedness label `ζ = σ₁` depends on the split of negatives into the
first block. -/
theorem star_eigen (S : Finset (Fin (2 * 4)))
    (h : eps S {0, 1, 2, 3} = eps S {4, 5, 6, 7}) :
    (starMat S).mulVec omegaPlus = ((eps S {0, 1, 2, 3} : ℤ) : ℚ) • omegaPlus := by
  ext I
  -- Expand the matrix-vector product into the two block contributions.
  have hexp : (starMat S).mulVec omegaPlus I = starMat S I B1 + starMat S I B2 := by
    rw [omegaPlus_eq, Matrix.mulVec_add]
    simp only [Matrix.mulVec_single, mul_one, Pi.add_apply]
  rw [hexp, Pi.smul_apply, smul_eq_mul]
  by_cases hI1 : I = B1
  · -- coordinate at B₁: 0 + σ₁ = σ₁ · 1
    subst hI1
    have d1 : starMat S B1 B1 = 0 := by unfold starMat; exact if_neg (by decide)
    have o1 : omegaPlus B1 = 1 := by unfold omegaPlus; rw [if_pos rfl]
    rw [d1, star_ω1, o1, mul_one, zero_add]
  · by_cases hI2 : I = B2
    · -- coordinate at B₂: σ₂ + 0 = σ₁ · 1, using σ₁ = σ₂
      subst hI2
      have d2 : starMat S B2 B2 = 0 := by unfold starMat; exact if_neg (by decide)
      have o2 : omegaPlus B2 = 1 := by
        unfold omegaPlus; rw [if_neg (by decide), if_pos rfl]
      rw [d2, star_ω2, o2, mul_one, add_zero, h]
    · -- all other coordinates vanish on both sides
      have hne1 : ¬ ((B1 : Finset (Fin (2 * 4))) = (I : Finset (Fin (2 * 4)))ᶜ) := by
        intro hcon
        apply hI2
        apply Subtype.ext
        rw [B2_eq_compl, hcon, compl_compl]
      have hne2 : ¬ ((B2 : Finset (Fin (2 * 4))) = (I : Finset (Fin (2 * 4)))ᶜ) := by
        intro hcon
        apply hI1
        apply Subtype.ext
        rw [B1_eq_compl, hcon, compl_compl]
      have e1 : starMat S I B1 = 0 := by unfold starMat; exact if_neg hne1
      have e2 : starMat S I B2 = 0 := by unfold starMat; exact if_neg hne2
      have o0 : omegaPlus I = 0 := by
        unfold omegaPlus; rw [if_neg hI1, if_neg hI2]
      rw [e1, e2, o0, mul_zero, add_zero]

end ChiralParity
