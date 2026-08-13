import BornRule.Basic

/-!
# Lemma 4.3 — Null-idempotent projectors

Formalization of Lemma 4.3 of the Born-rule paper: the split units `E_n` are
Hermitian and traceless, the branch currents `J_s(n) = 1 + s E_n` are null
(`N(J) = 0`) and self-dagger, and the half-currents `P_s(n) = J_s(n)/2` form a
complete orthogonal pair of idempotent projectors.
-/

namespace BornRule

open Matrix

/-- Each quaternion unit is anti-Hermitian: `qi† = -qi`. -/
lemma qi_dagger : qiᴴ = -qi := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [qi, Matrix.conjTranspose_apply, Matrix.neg_apply, Complex.conj_I]

lemma qj_dagger : qjᴴ = -qj := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [qj, Matrix.conjTranspose_apply, Matrix.neg_apply]

lemma qk_dagger : qkᴴ = -qk := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [qk, Matrix.conjTranspose_apply, Matrix.neg_apply, Complex.conj_I]

/-- Item 1: `e_n† = -e_n`, i.e. `e_n` is anti-Hermitian. -/
lemma En_conjTranspose (n : Fin 3 → ℝ) : (en n)ᴴ = - en n := by
  unfold en
  rw [conjTranspose_add, conjTranspose_add, conjTranspose_smul, conjTranspose_smul,
    conjTranspose_smul, qi_dagger, qj_dagger, qk_dagger]
  simp only [Complex.star_def, Complex.conj_ofReal, smul_neg]
  abel

/-- Item 2: `E_n` is Hermitian, `E_n† = E_n`. -/
lemma En_dagger (n : Fin 3 → ℝ) : dagger (En n) = En n := by
  show (En n)ᴴ = En n
  unfold En
  rw [conjTranspose_smul, En_conjTranspose,
    show star Complex.I = -Complex.I from by rw [← starRingEnd_apply, Complex.conj_I]]
  rw [neg_smul, smul_neg, neg_neg]

/-- Item 3: `E_n` is traceless, so its adjugate (bar) is `-E_n`. -/
lemma En_bar (n : Fin 3 → ℝ) : bar (En n) = - En n := by
  unfold bar
  rw [adjugate_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [En, en, qi, qj, qk, Matrix.add_apply, Matrix.smul_apply, Matrix.neg_apply,
      smul_eq_mul, Complex.I_sq]

/-- Item 4: the branch current is self-dagger when the parameter is real. -/
lemma Jbranch_dagger (n : Fin 3 → ℝ) {s : ℂ} (hs : (starRingEnd ℂ) s = s) :
    dagger (Jbranch s n) = Jbranch s n := by
  show (Jbranch s n)ᴴ = Jbranch s n
  unfold Jbranch
  rw [conjTranspose_add, conjTranspose_one, conjTranspose_smul]
  have hE : (En n)ᴴ = En n := En_dagger n
  rw [hE, show star s = s from by rw [← starRingEnd_apply]; exact hs]

/-- Item 5: `J_s(n)² = 2 J_s(n)` on the unit sphere with `s² = 1`. -/
lemma Jbranch_sq {n : Fin 3 → ℝ} (hn : IsUnit3 n) {s : ℂ} (hs : s ^ 2 = 1) :
    Jbranch s n * Jbranch s n = (2 : ℂ) • Jbranch s n := by
  have expand : Jbranch s n * Jbranch s n
      = (1 : Carrier) + s • En n + s • En n + (s * s) • (En n * En n) := by
    unfold Jbranch
    rw [mul_add, add_mul, add_mul, smul_mul_smul_comm]
    simp only [one_mul, mul_one]
    abel
  rw [expand, En_sq_unit hn, ← pow_two, hs, one_smul]
  unfold Jbranch
  module

/-- Item 6: the branch current is null, `N(J_s(n)) = 0`. -/
lemma redNorm_Jbranch_zero {n : Fin 3 → ℝ} (hn : IsUnit3 n) {s : ℂ} (hs : s ^ 2 = 1) :
    redNorm (Jbranch s n) = 0 := by
  have hn' : (n 0 : ℂ) ^ 2 + (n 1 : ℂ) ^ 2 + (n 2 : ℂ) ^ 2 = 1 := by
    have h : (n 0) ^ 2 + (n 1) ^ 2 + (n 2) ^ 2 = 1 := hn
    exact_mod_cast h
  have key : (Jbranch s n).det
      = 1 - s ^ 2 * ((n 0 : ℂ) ^ 2 + (n 1 : ℂ) ^ 2 + (n 2 : ℂ) ^ 2) := by
    rw [Matrix.det_fin_two]
    simp [Jbranch, En, en, qi, qj, qk, Matrix.add_apply,
      Matrix.smul_apply, Matrix.one_apply, smul_eq_mul]
    ring_nf
    simp only [Complex.I_sq, show Complex.I ^ 4 = (1 : ℂ) from by
      rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, Complex.I_sq]; norm_num]
    ring
  unfold redNorm
  rw [key, hn', hs]
  ring

/-- Item 7: `P_s(n)` is idempotent. -/
lemma Pbranch_idem {n : Fin 3 → ℝ} (hn : IsUnit3 n) {s : ℂ} (hs : s ^ 2 = 1) :
    Pbranch s n * Pbranch s n = Pbranch s n := by
  unfold Pbranch
  rw [smul_mul_smul_comm, Jbranch_sq hn hs, smul_smul,
    show (2⁻¹ * 2⁻¹ * (2 : ℂ)) = 2⁻¹ from by norm_num]

/-- Item 8: `P_s(n)` is self-dagger (Hermitian) for real `s`. -/
lemma Pbranch_dagger {n : Fin 3 → ℝ} {s : ℂ} (hs : (starRingEnd ℂ) s = s) :
    dagger (Pbranch s n) = Pbranch s n := by
  show (Pbranch s n)ᴴ = Pbranch s n
  unfold Pbranch
  rw [conjTranspose_smul]
  have hj : (Jbranch s n)ᴴ = Jbranch s n := Jbranch_dagger n hs
  rw [hj, show star (2⁻¹ : ℂ) = 2⁻¹ from by simp]

/-- Item 9: the two branch projectors are orthogonal. -/
lemma Pbranch_orthogonal {n : Fin 3 → ℝ} (hn : IsUnit3 n) :
    Pbranch 1 n * Pbranch (-1) n = 0 := by
  have hJ : Jbranch 1 n * Jbranch (-1) n = 0 := by
    unfold Jbranch
    rw [one_smul, neg_one_smul, mul_add, mul_one, mul_neg, add_mul, one_mul, En_sq_unit hn]
    abel
  unfold Pbranch
  rw [smul_mul_smul_comm, hJ, smul_zero]

/-- Item 10: the two branch projectors are complete, summing to the identity. -/
lemma Pbranch_complete (n : Fin 3 → ℝ) : Pbranch 1 n + Pbranch (-1) n = 1 := by
  unfold Pbranch Jbranch
  rw [one_smul, neg_one_smul]
  module

end BornRule
