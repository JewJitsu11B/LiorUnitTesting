import Mathlib

/-
IR fine-structure constant:  alpha_EM^{-1}(0) = 133 + 4 + 1/D,
  D = 27 + 1/sqrt2 + 1/14 - 1/(133*61).
Cited band: CODATA 2022, 137.035999177(21)  =>  [137.035999156, 137.035999198].
The TIGHT-BAND concern (+-2.1e-8). D is kept faithful to the paper (uses (sqrt2)^{-1});
we prove (sqrt2)^{-1} = sqrt2/2 so the interval arithmetic is degree-1 in sqrt2, and the
single 1/D inversion is handled with le_div_iff + an explicit D > 0.
-/
namespace CALConsistencyReal.FSC
open Real

noncomputable def r2 : ℝ := Real.sqrt 2
noncomputable def D : ℝ := 27 + (r2)⁻¹ + (1 : ℝ) / 14 - (1 : ℝ) / (133 * 61)
noncomputable def aInv : ℝ := 133 + 4 + D⁻¹

lemma r2_sq : r2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
lemma r2_pos : 0 < r2 := Real.sqrt_pos.mpr (by norm_num)
lemma r2_lb : (1.414213 : ℝ) ≤ r2 := by nlinarith [r2_sq, r2_pos.le]
lemma r2_ub : r2 ≤ (1.414214 : ℝ) := by nlinarith [r2_sq, r2_pos.le]

/-- 1/sqrt2 = sqrt2/2 (since sqrt2^2 = 2). Keeps the interval arithmetic degree-1. -/
lemma inv_r2 : (r2)⁻¹ = r2 / 2 := by
  have h : r2 * r2 = 2 := by rw [← r2_sq]; ring
  field_simp [r2_pos.ne']
  linarith [h]

lemma D_lb : (27.778411 : ℝ) ≤ D := by unfold D; rw [inv_r2]; nlinarith [r2_lb]
lemma D_ub : D ≤ (27.778413 : ℝ) := by unfold D; rw [inv_r2]; nlinarith [r2_ub]
lemma D_pos : 0 < D := by have := D_lb; linarith

/-- The derived alpha_EM^{-1}(0) lands in the CODATA band. Non-vacuous. -/
theorem aInv_in_band : (137.035999156 : ℝ) ≤ aInv ∧ aInv ≤ (137.035999198 : ℝ) := by
  have hDpos := D_pos
  -- D in [27.778411, 27.778413] => D^{-1} in [0.0359991767, 0.0359991793];
  -- safe rounded bounds 0.03599917 / 0.03599918, both well inside the CODATA band.
  have hinv_lo : (0.03599917 : ℝ) ≤ D⁻¹ := by
    rw [show D⁻¹ = 1 / D from (one_div D).symm, le_div_iff₀ hDpos]
    nlinarith [D_ub]
  have hinv_hi : D⁻¹ ≤ (0.03599918 : ℝ) := by
    rw [show D⁻¹ = 1 / D from (one_div D).symm, div_le_iff₀ hDpos]
    nlinarith [D_lb]
  unfold aInv
  refine ⟨?_, ?_⟩
  · linarith [hinv_lo]
  · linarith [hinv_hi]

end CALConsistencyReal.FSC
