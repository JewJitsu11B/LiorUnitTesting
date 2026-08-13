import BornRule.Basic

/-!
# Proposition 5.2 — emergent order-two source closure / branch collapse

We show that imposing the closure identity `J J† = 2 J` on the null source current
`J(λ,n) = 1 + λ·E_n` forces the phase `λ` to be real and of unit modulus, i.e. `λ = ±1`.
These are exactly the two physical branches of Definition 2.7.

Self-contained: the Hermiticity `E_n† = E_n` is re-proved here as a private helper.
-/

namespace BornRule

open Matrix

/-- `e_n` is anti-Hermitian: `(e_n)ᴴ = -e_n` (each quaternion unit satisfies `qᴴ = -q`). -/
private lemma en_conjTranspose (n : Fin 3 → ℝ) : (en n)ᴴ = - en n := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [en, qi, qj, qk, Matrix.conjTranspose_apply, Matrix.smul_apply, Matrix.add_apply,
      Matrix.neg_apply, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one,
      Matrix.head_fin_const, smul_eq_mul] <;> ring

/-- Item 1: `E_n` is Hermitian, `E_n† = E_n` (Definition 2.7, `E_n = i·e_n`). -/
private lemma En_dagger (n : Fin 3 → ℝ) : dagger (En n) = En n := by
  have hI : star (Complex.I) = -Complex.I := Complex.conj_I
  unfold dagger En
  rw [conjTranspose_smul, en_conjTranspose, hI, smul_neg, neg_smul, neg_neg]

/-- Item 2: the closure expansion `J J† = (1+λλ̄)·1 + (λ+λ̄)·E_n`. -/
lemma closure_expand {n : Fin 3 → ℝ} (hn : IsUnit3 n) (lam : ℂ) :
    Jbranch lam n * dagger (Jbranch lam n)
      = (1 + lam * (starRingEnd ℂ) lam) • (1 : Carrier)
          + (lam + (starRingEnd ℂ) lam) • En n := by
  have hd : dagger (Jbranch lam n) = 1 + ((starRingEnd ℂ) lam) • En n := by
    unfold dagger Jbranch
    rw [conjTranspose_add, conjTranspose_one, conjTranspose_smul]
    rw [show ((En n)ᴴ) = En n from En_dagger n]
    rfl
  rw [hd, Jbranch, mul_add, add_mul, add_mul]
  simp only [one_mul, mul_one, smul_mul_smul_comm, En_sq_unit hn]
  module

/-- Item 3: `2·J = 2·1 + (2λ)·E_n`. -/
lemma two_J {n : Fin 3 → ℝ} (lam : ℂ) :
    (2 : ℂ) • Jbranch lam n = (2 : ℂ) • (1 : Carrier) + (2 * lam) • En n := by
  rw [Jbranch, smul_add, smul_smul]

/-- `E_n` is traceless (helper for coefficient extraction). -/
private lemma trace_En (n : Fin 3 → ℝ) : (En n).trace = 0 := by
  simp only [En, en, qi, qj, qk, Matrix.trace_fin_two, Matrix.smul_apply,
    Matrix.add_apply, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one,
    Matrix.head_fin_const, smul_eq_mul]
  ring

/-- Item 4: THE COLLAPSE. Closure forces the phase real and of unit modulus. -/
theorem closure_forces_real {n : Fin 3 → ℝ} (hn : IsUnit3 n) (lam : ℂ)
    (hclose : Jbranch lam n * dagger (Jbranch lam n) = (2 : ℂ) • Jbranch lam n) :
    lam = 1 ∨ lam = -1 := by
  -- Match coefficients of the linearly independent `{1, E_n}`.
  have hcoll : (1 + lam * (starRingEnd ℂ) lam) • (1 : Carrier)
        + (lam + (starRingEnd ℂ) lam) • En n
      = (2 : ℂ) • (1 : Carrier) + (2 * lam) • En n := by
    rw [← closure_expand hn lam, ← two_J lam]; exact hclose
  have htr : (En n).trace = 0 := trace_En n
  have hEE : En n * En n = 1 := En_sq_unit hn
  -- Coefficient of `1`: apply the trace functional (tr 1 = 2, tr E_n = 0).
  have h1 := congrArg Matrix.trace hcoll
  simp only [Matrix.trace_add, Matrix.trace_smul, htr, Matrix.trace_one,
    Fintype.card_fin, smul_eq_mul, mul_zero, add_zero, Nat.cast_ofNat] at h1
  -- Coefficient of `E_n`: apply `X ↦ tr (E_n * X)` (tr(E_n·1)=0, tr(E_n·E_n)=2).
  have h2 := congrArg (fun M => (En n * M).trace) hcoll
  simp only [mul_add, mul_smul_comm, Matrix.trace_add, Matrix.trace_smul, mul_one, hEE,
    htr, Matrix.trace_one, Fintype.card_fin, smul_eq_mul, mul_zero, zero_add, add_zero,
    Nat.cast_ofNat] at h2
  -- From h2: λ̄ = λ.  From h1: λ·λ̄ = 1.  Hence λ² = 1.
  have hconj : (starRingEnd ℂ) lam = lam := by linear_combination h2 / 2
  have hsq : lam * lam = 1 := by
    have hll : lam * (starRingEnd ℂ) lam = 1 := by linear_combination h1 / 2
    rw [hconj] at hll; exact hll
  exact mul_self_eq_one_iff.mp hsq

end BornRule
