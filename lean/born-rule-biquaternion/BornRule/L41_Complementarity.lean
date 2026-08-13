import BornRule.Basic

namespace BornRule

open Matrix
open scoped ComplexOrder

/-!
# Lemma 4.1 — complementarity of the two closures

The reduced-norm (determinant) closure `N(q) = q q̄` is multiplicative but degenerate
(admits null divisors), whereas the dagger closure `q† q` is positive semidefinite with a
strictly positive trace on the null divisor. The two closures are complementary: one is a
multiplicative form that vanishes on genuine sources, the other a positive form that does not.
-/

/-- (a) The reduced norm is multiplicative: `N(q q') = N(q) N(q')` (it is the determinant). -/
theorem redNorm_mul (M M' : Carrier) :
    redNorm (M * M') = redNorm M * redNorm M' := by
  simp [redNorm, Matrix.det_mul]

/-- The paper's canonical null divisor `q = 1 + i·i` equals the diagonal matrix `diag(0,2)`. -/
theorem null_example_eq : (1 : Carrier) + Complex.I • qi = !![0, 0; 0, 2] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [qi, Matrix.one_apply, Matrix.add_apply, Matrix.smul_apply, Complex.I_mul_I] <;>
    norm_num

/-- (b) The null divisor `q = 1 + i·i` has vanishing reduced norm. -/
theorem redNorm_null_example : redNorm ((1 : Carrier) + Complex.I • qi) = 0 := by
  rw [redNorm, null_example_eq, Matrix.det_fin_two_of]
  ring

/-- (c) The dagger closure `q† q` is positive semidefinite. -/
theorem dagger_form_posSemidef (M : Carrier) : (dagger M * M).PosSemidef := by
  simp only [dagger]
  exact Matrix.posSemidef_conjTranspose_mul_self M

/-- Auxiliary: for a complex scalar, `(star z * z).re = ‖z‖² ≥ 0`. -/
private lemma star_mul_self_re (z : ℂ) : (star z * z).re = Complex.normSq z := by
  rw [Complex.star_def, mul_comm, Complex.mul_conj, Complex.ofReal_re]

/-- (d) The dagger closure has nonnegative trace: `Re tr(q† q) = ∑ |qᵢⱼ|² ≥ 0`. -/
theorem dagger_form_trace_nonneg (M : Carrier) : 0 ≤ (dagger M * M).trace.re := by
  simp only [dagger, Matrix.trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two,
      Matrix.conjTranspose_apply, Complex.add_re, star_mul_self_re]
  exact add_nonneg (add_nonneg (Complex.normSq_nonneg _) (Complex.normSq_nonneg _))
    (add_nonneg (Complex.normSq_nonneg _) (Complex.normSq_nonneg _))

/-- (e) Complementarity: the canonical source `q = 1 + i·i` is a null divisor for the
reduced-norm closure yet has strictly positive dagger-trace. The two closures disagree. -/
theorem redNorm_null_but_dagger_pos :
    redNorm ((1 : Carrier) + Complex.I • qi) = 0 ∧
      0 < (dagger ((1 : Carrier) + Complex.I • qi) *
            ((1 : Carrier) + Complex.I • qi)).trace.re := by
  refine ⟨redNorm_null_example, ?_⟩
  rw [null_example_eq]
  simp only [dagger]
  rw [show (!![0, 0; 0, 2] : Carrier)ᴴ = !![0, 0; 0, 2] by
        ext i j; fin_cases i <;> fin_cases j <;> simp]
  rw [Matrix.trace_fin_two, Matrix.mul_apply, Matrix.mul_apply, Fin.sum_univ_two,
      Fin.sum_univ_two]
  norm_num

end BornRule
