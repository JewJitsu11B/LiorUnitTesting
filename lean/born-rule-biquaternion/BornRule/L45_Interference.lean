import BornRule.Basic

/-!
# Lemma 4.5: interference cancellation at the dagger

The daggered self-product `q† q` has real trace, and cross (interference) terms
between the real and imaginary parts of a central-scalar amplitude cancel because
`i† = -i`.  The observable readout `B = Re tr` is therefore free of interference.
-/

namespace BornRule

open Matrix

/-- `trace` of a commutator vanishes: `tr(AB) = tr(BA)` (Lemma 4.5, base identity). -/
theorem trace_commutator (A B : Carrier) : (A * B - B * A).trace = 0 := by
  rw [Matrix.trace_sub, Matrix.trace_mul_comm A B, sub_self]

/-- The internal readout of a commutator vanishes. -/
theorem traceRead_commutator (A B : Carrier) : traceRead (A * B - B * A) = 0 := by
  unfold traceRead
  rw [trace_commutator]
  simp

/-- Scalar interference cancellation: for the central-scalar amplitude
`W = g•1 + t•(i•1)`, the daggered self-product is `|W|² • 1 = (g² + t²)•1`.
The interference terms `±i g t` cancel because `i† = -i`. -/
theorem scalar_interference (g t : ℝ) :
    dagger (((g:ℂ)•(1:Carrier) + (t:ℂ)•(Complex.I•(1:Carrier)))) *
      ((g:ℂ)•(1:Carrier) + (t:ℂ)•(Complex.I•(1:Carrier)))
      = (((g^2 + t^2 : ℝ)):ℂ) • (1:Carrier) := by
  unfold dagger
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.conjTranspose_apply,
          Matrix.add_apply, Matrix.smul_apply, Matrix.one_apply,
          Complex.conj_ofReal, Complex.conj_I] <;>
    ring_nf <;> (try simp only [Complex.I_sq]) <;> ring

/-- The daggered self-product `q† q` has real trace (imaginary part vanishes),
since `q† q = ∑ |qᵢⱼ|²`.  This is the interference-free observable. -/
theorem dagger_trace_real (M : Carrier) : ((dagger M * M).trace).im = 0 := by
  unfold dagger
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply,
             Matrix.conjTranspose_apply, Fin.sum_univ_two]
  simp [Complex.add_im, Complex.mul_im, Complex.conj_re, Complex.conj_im]
  ring

end BornRule
