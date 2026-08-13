import BornRule.Carrier

/-!
# Base-layer algebra of the quaternion units in `M₂(ℂ)`

Shared, tested building blocks for the null-idempotent derivation. Every downstream
lemma (4.1, 4.3, 4.5, Props 5.x, Thm 6.1) reduces to these unit relations plus `Complex.I_sq`.
-/

namespace BornRule

open Matrix

/-- Tactic recipe: expand a `!![..]` matrix identity entrywise, normalize, then use `I²=-1`. -/
macro "unit_alg" : tactic =>
  `(tactic| (ext i j; fin_cases i <;> fin_cases j <;>
      simp [qi, qj, qk, en, Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply,
            Matrix.add_apply, Matrix.smul_apply] <;>
      ring_nf <;> (try simp only [Complex.I_sq]) <;> ring))

@[simp] lemma qi_sq : qi * qi = -1 := by unit_alg
@[simp] lemma qj_sq : qj * qj = -1 := by unit_alg
@[simp] lemma qk_sq : qk * qk = -1 := by unit_alg

/-- `e_n² = -‖n‖² • 1`.  With `‖n‖²=1` this gives `e_n² = -1`. -/
lemma en_sq (n : Fin 3 → ℝ) :
    en n * en n = (-((n 0)^2 + (n 1)^2 + (n 2)^2 : ℝ) : ℂ) • (1 : Carrier) := by
  unit_alg

/-- `E_n² = ‖n‖² • 1`, derived from `en_sq` (avoids `I⁴`); the paper's `E_n²=+1` is the unit case. -/
lemma En_sq (n : Fin 3 → ℝ) :
    En n * En n = (((n 0)^2 + (n 1)^2 + (n 2)^2 : ℝ) : ℂ) • (1 : Carrier) := by
  rw [En, smul_mul_smul_comm, Complex.I_mul_I, en_sq, smul_smul]
  congr 1
  push_cast
  ring

/-- `E_n² = 1` on the unit sphere (Definition 2.7 / Lemma 4.3). -/
lemma En_sq_unit {n : Fin 3 → ℝ} (hn : IsUnit3 n) : En n * En n = 1 := by
  rw [En_sq n]
  rw [IsUnit3] at hn
  rw [hn]
  simp

end BornRule
