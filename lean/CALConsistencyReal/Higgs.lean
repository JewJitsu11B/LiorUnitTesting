import Mathlib

/-
Corrected Higgs mass (unified-doc value): m_H = v*28/55 - 14*sqrt2/100, v = 246.22 GeV.
The coset present term v*28/55 = 125.348 minus the G2 spectral+geometric residual
14*sqrt2/100 = 0.198, giving 125.150 GeV. Cited band: PDG 2024 125.25 +- 0.17
=> [125.08, 125.42]. (The sqrt2 is subtracted, so this is cleaner as a direct sqrt2
interval than the rational squaring trick.)
-/
namespace CALConsistencyReal.Higgs
open Real

noncomputable def r2 : ℝ := Real.sqrt 2
noncomputable def mH : ℝ := 246.22 * 28 / 55 - 14 * r2 / 100

lemma r2_sq : r2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
lemma r2_nonneg : 0 ≤ r2 := Real.sqrt_nonneg 2
lemma r2_lb : (1.414213 : ℝ) ≤ r2 := by nlinarith [r2_sq, r2_nonneg]
lemma r2_ub : r2 ≤ (1.414214 : ℝ) := by nlinarith [r2_sq, r2_nonneg]

/-- The corrected Higgs mass lands in the PDG band. -/
theorem mH_in_band : (125.08 : ℝ) ≤ mH ∧ mH ≤ (125.42 : ℝ) := by
  unfold mH
  refine ⟨?_, ?_⟩
  · nlinarith [r2_ub]   -- lower bound: -14*r2/100 is least negative when r2 is largest
  · nlinarith [r2_lb]   -- upper bound: holds with margin (P < hi already)

end CALConsistencyReal.Higgs
