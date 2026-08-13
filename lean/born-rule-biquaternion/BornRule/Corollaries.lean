import BornRule.T61_Born
import BornRule.L41_Complementarity

namespace BornRule

open Matrix
open scoped ComplexOrder

/-!
# Corollaries closing two findings from the adversarial review

* Finding A: the Born readout is a genuine probability (lies in `[0,1]`) once the two
  directions are unit vectors.
* Finding B: the dagger (Hermitian) form is strictly positive-definite, upgrading
  Lemma 4.1(b) from positive-semidefinite to the paper's "positive definite".
-/

/-- Auxiliary: for a complex scalar, `(star z * z).re = ‖z‖²`. -/
private lemma star_mul_self_re' (z : ℂ) : (star z * z).re = Complex.normSq z := by
  rw [Complex.star_def, mul_comm, Complex.mul_conj, Complex.ofReal_re]

/-! ## Finding A -/

/-- The Euclidean dot product of two unit 3-vectors lies in `[-1, 1]`
(the Cauchy–Schwarz bound; here `2(1 ∓ m·n) = ∑(mᵢ ± nᵢ)² ≥ 0`). -/
theorem dot_bounds {m n : Fin 3 → ℝ} (hm : IsUnit3 m) (hn : IsUnit3 n) :
    -1 ≤ (m 0 * n 0 + m 1 * n 1 + m 2 * n 2) ∧
      (m 0 * n 0 + m 1 * n 1 + m 2 * n 2) ≤ 1 := by
  rw [IsUnit3] at hm hn
  constructor <;>
    nlinarith [sq_nonneg (m 0 - n 0), sq_nonneg (m 1 - n 1), sq_nonneg (m 2 - n 2),
      sq_nonneg (m 0 + n 0), sq_nonneg (m 1 + n 1), sq_nonneg (m 2 + n 2)]

/-- The Born readout `B(P₊(m) P₊(n))` is a genuine probability: it lies in `[0, 1]`
whenever `m` and `n` are unit vectors. This is the faithful "it is a probability"
statement that the bare identity `born_plus_plus` lacked. -/
theorem born_prob_mem_Icc {m n : Fin 3 → ℝ} (hm : IsUnit3 m) (hn : IsUnit3 n) :
    0 ≤ traceRead (Pbranch 1 m * Pbranch 1 n) ∧
      traceRead (Pbranch 1 m * Pbranch 1 n) ≤ 1 := by
  rw [born_plus_plus]
  obtain ⟨hlo, hhi⟩ := dot_bounds hm hn
  constructor <;> linarith

/-! ## Finding B -/

/-- The Hermitian self-product has trace `Re tr(q† q) = ∑ᵢⱼ |qᵢⱼ|²`, which is STRICTLY
positive when `q ≠ 0`. This upgrades Lemma 4.1(b) from positive-semidefinite to the
paper's "positive definite". -/
theorem dagger_form_trace_pos {q : Carrier} (hq : q ≠ 0) :
    0 < (dagger q * q).trace.re := by
  -- Expand the trace of the Hermitian self-product into a sum of four moduli squared.
  have hsum : (dagger q * q).trace.re =
      Complex.normSq (q 0 0) + Complex.normSq (q 1 0)
        + Complex.normSq (q 0 1) + Complex.normSq (q 1 1) := by
    simp only [dagger, Matrix.trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two,
      Matrix.conjTranspose_apply, Complex.add_re, star_mul_self_re']
    ring
  -- `q ≠ 0` yields a nonzero entry.
  have hne : ∃ i j, q i j ≠ 0 := by
    by_contra h
    push_neg at h
    exact hq (by ext i j; simp [h i j])
  obtain ⟨i, j, hij⟩ := hne
  rw [hsum]
  have h00 := Complex.normSq_nonneg (q 0 0)
  have h10 := Complex.normSq_nonneg (q 1 0)
  have h01 := Complex.normSq_nonneg (q 0 1)
  have h11 := Complex.normSq_nonneg (q 1 1)
  fin_cases i <;> fin_cases j <;>
    · have hpos := Complex.normSq_pos.mpr hij
      simp only [Fin.isValue, Fin.mk_zero, Fin.mk_one] at hpos
      linarith

end BornRule
