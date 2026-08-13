import ChiralParity.StarSquare

/-!
# Theorem 3.2 (Parity Criterion) and Corollary 3.3

The chiral pair `P₊ = ½(1+⋆)`, `P₋ = ½(1−⋆)` is a genuine pair of complementary
idempotents iff the star-square scalar `s = starScalar S = (-1)^{m²+q}` equals `1`
(equivalently `m²+q` is even). When `s = -1` the projector relations fail by exact
scalar constants (`½` and `−½`).
-/

namespace ChiralParity

open Matrix Finset

variable {m : ℕ}

/-- Item 1: `s = 1` iff `m² + q` is even. -/
theorem starScalar_eq_one_iff (S : Finset (Fin (2 * m))) :
    starScalar S = 1 ↔ Even (m ^ 2 + qneg S) := by
  unfold starScalar
  exact neg_one_pow_eq_one_iff_even (by norm_num)

/-- Item 2: `s = -1` iff `m² + q` is odd. -/
theorem starScalar_eq_neg_one_iff (S : Finset (Fin (2 * m))) :
    starScalar S = -1 ↔ Odd (m ^ 2 + qneg S) := by
  unfold starScalar
  constructor
  · intro hh
    by_contra hodd
    rw [Nat.not_odd_iff_even] at hodd
    rw [hodd.neg_one_pow] at hh
    norm_num at hh
  · intro h
    exact h.neg_one_pow

/-- Item 3: when `s = 1`, `P₊` is idempotent. -/
theorem Pplus_sq_of_even (S : Finset (Fin (2 * m))) (h : starScalar S = 1) :
    Pplus S * Pplus S = Pplus S := by
  simp only [Pplus]
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, mul_add, add_mul, add_mul]
  simp only [mul_one, one_mul]
  rw [starMat_sq, h]
  module

/-- Item 4: when `s = 1`, `P₊ P₋ = 0`. -/
theorem Pplus_Pminus_of_even (S : Finset (Fin (2 * m))) (h : starScalar S = 1) :
    Pplus S * Pminus S = 0 := by
  simp only [Pplus, Pminus]
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, add_mul, mul_sub, mul_sub]
  simp only [mul_one, one_mul]
  rw [starMat_sq, h]
  module

/-- Item 5: when `s = -1`, `P₊² − P₊ = −½·1` (exact scalar failure). -/
theorem Pplus_sq_sub_of_odd (S : Finset (Fin (2 * m))) (h : starScalar S = -1) :
    Pplus S * Pplus S - Pplus S
      = (-(2⁻¹) : ℚ) • (1 : Matrix (Blade m) (Blade m) ℚ) := by
  simp only [Pplus]
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, mul_add, add_mul, add_mul]
  simp only [mul_one, one_mul]
  rw [starMat_sq, h]
  module

/-- Item 6: when `s = -1`, `P₊ P₋ = ½·1` (exact scalar failure). -/
theorem Pplus_Pminus_of_odd (S : Finset (Fin (2 * m))) (h : starScalar S = -1) :
    Pplus S * Pminus S = (2⁻¹ : ℚ) • (1 : Matrix (Blade m) (Blade m) ℚ) := by
  simp only [Pplus, Pminus]
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, add_mul, mul_sub, mul_sub]
  simp only [mul_one, one_mul]
  rw [starMat_sq, h]
  module

/-- Corollary 3.3: at `n = 8`, `m = 4`, the carrier chirality condition is `Even q`. -/
theorem carrier_condition (S : Finset (Fin (2 * 4))) :
    starScalar S = 1 ↔ Even (qneg S) := by
  rw [starScalar_eq_one_iff]
  have h16 : (4 : ℕ) ^ 2 = 16 := by norm_num
  have h16e : Even (16 : ℕ) := by decide
  rw [h16, Nat.even_add]
  simp [h16e]

end ChiralParity
