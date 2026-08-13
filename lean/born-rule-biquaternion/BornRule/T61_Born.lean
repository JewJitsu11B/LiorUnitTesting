import BornRule.Basic

namespace BornRule

open Matrix

/-- Real dot product of two 3-vectors. -/
def dot (m n : Fin 3 → ℝ) : ℝ := m 0 * n 0 + m 1 * n 1 + m 2 * n 2

/-- Item 1: `E_n` is traceless. -/
lemma trace_En (n : Fin 3 → ℝ) : (En n).trace = 0 := by
  simp only [En, en, qi, qj, qk, Matrix.trace_fin_two, Matrix.smul_apply,
    Matrix.add_apply, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one,
    Matrix.head_fin_const, smul_eq_mul]
  ring

/-- Item 2 (KEY): `tr(E_m E_n) = 2 (m · n)`. -/
lemma trace_EmEn (m n : Fin 3 → ℝ) :
    (En m * En n).trace = (2 : ℂ) * ((m 0 * n 0 + m 1 * n 1 + m 2 * n 2 : ℝ) : ℂ) := by
  simp only [En, en, qi, qj, qk, Matrix.trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two,
    Matrix.smul_apply, Matrix.add_apply, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one,
    Matrix.head_fin_const, smul_eq_mul]
  ring_nf
  simp only [Complex.I_sq,
    show (Complex.I) ^ 4 = 1 from by
      rw [show (4 : ℕ) = 2 + 2 from rfl, pow_add, Complex.I_sq]; ring]
  push_cast
  ring

/-- Item 3: `traceRead 1 = 2`. -/
lemma traceRead_one : traceRead (1 : Carrier) = 2 := by
  simp [traceRead, Matrix.trace_one]

/-- Item 4: the `s = +1` Born readout. -/
lemma born_plus_plus (m n : Fin 3 → ℝ) :
    traceRead (Pbranch 1 m * Pbranch 1 n) = (1 + (m 0 * n 0 + m 1 * n 1 + m 2 * n 2)) / 2 := by
  have htr : (Pbranch 1 m * Pbranch 1 n).trace
      = (((1 + (m 0 * n 0 + m 1 * n 1 + m 2 * n 2)) / 2 : ℝ) : ℂ) := by
    simp only [Pbranch, Jbranch, one_smul, smul_mul_smul_comm, Matrix.trace_smul, smul_eq_mul,
      add_mul, mul_add, one_mul, mul_one, Matrix.trace_add, Matrix.trace_one, Fintype.card_fin,
      trace_En, trace_EmEn]
    push_cast
    ring
  rw [traceRead, htr, Complex.ofReal_re]

/-- Item 5: the `+` and `-` outcomes for `m` sum to `1`. -/
lemma born_normalization (m n : Fin 3 → ℝ) :
    traceRead (Pbranch 1 m * Pbranch 1 n) + traceRead (Pbranch (-1) m * Pbranch 1 n) = 1 := by
  -- The two branch projectors for `m` sum to the identity.
  have h1 : Pbranch 1 m + Pbranch (-1) m = (1 : Carrier) := by
    simp only [Pbranch, Jbranch, one_smul]
    module
  -- Hence the two products sum to `Pbranch 1 n`.
  have hsum : Pbranch 1 m * Pbranch 1 n + Pbranch (-1) m * Pbranch 1 n = Pbranch 1 n := by
    rw [← add_mul, h1, one_mul]
  -- `traceRead` is additive, so reduce to the readout of a single projector.
  have hadd : traceRead (Pbranch 1 m * Pbranch 1 n) + traceRead (Pbranch (-1) m * Pbranch 1 n)
      = traceRead (Pbranch 1 m * Pbranch 1 n + Pbranch (-1) m * Pbranch 1 n) := by
    simp only [traceRead, Matrix.trace_add, Complex.add_re]
  rw [hadd, hsum]
  -- `traceRead (Pbranch 1 n) = 1`.
  have hpn : (Pbranch 1 n).trace = ((1 : ℝ) : ℂ) := by
    simp only [Pbranch, Jbranch, one_smul, Matrix.trace_smul, smul_eq_mul,
      Matrix.trace_add, Matrix.trace_one, Fintype.card_fin, trace_En]
    push_cast
    ring
  rw [traceRead, hpn, Complex.ofReal_re]

end BornRule
