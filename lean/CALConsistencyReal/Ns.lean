import Mathlib

/-
Spectral index n_s = 1 - phi^{-7}, phi = (1+sqrt5)/2.
Cited band: Planck 2018, n_s = 0.9649 +- 0.0042  =>  [0.9607, 0.9691].
Algebraic-irrational prediction (sqrt5); the wide band makes the enclosure forgiving.
Reference case confirming the Mathlib Real-enclosure template works.
-/
namespace CALConsistencyReal.Ns
open Real

noncomputable def s : ℝ := Real.sqrt 5
noncomputable def phi : ℝ := (1 + s) / 2
/-- n_s = 1 - phi^{-7}. -/
noncomputable def ns : ℝ := 1 - (phi ^ 7)⁻¹

lemma s_sq : s ^ 2 = 5 := Real.sq_sqrt (by norm_num)
lemma s_nonneg : 0 ≤ s := Real.sqrt_nonneg 5
lemma s_lb : (2.236 : ℝ) ≤ s := by nlinarith [s_sq, s_nonneg]
lemma s_ub : s ≤ (2.2361 : ℝ) := by nlinarith [s_sq, s_nonneg]
lemma phi_pos : 0 < phi := by unfold phi; nlinarith [s_nonneg]
lemma phi_lb : (1.618 : ℝ) ≤ phi := by unfold phi; nlinarith [s_lb]
lemma phi_ub : phi ≤ (1.61805 : ℝ) := by unfold phi; nlinarith [s_ub]

/-- The derived value lands in the Planck band. Non-vacuous: perturb either bound and it fails. -/
theorem ns_in_band : (0.9607 : ℝ) ≤ ns ∧ ns ≤ (0.9691 : ℝ) := by
  have h7l : (1.618 : ℝ) ^ 7 ≤ phi ^ 7 := by gcongr; exact phi_lb
  have h7u : phi ^ 7 ≤ (1.61805 : ℝ) ^ 7 := by
    gcongr
    · exact le_of_lt phi_pos
    · exact phi_ub
  have hlo : (25.45 : ℝ) ≤ phi ^ 7 := by
    have : (25.45 : ℝ) ≤ (1.618 : ℝ) ^ 7 := by norm_num
    linarith
  have hhi : phi ^ 7 ≤ (32.36 : ℝ) := by
    have : (1.61805 : ℝ) ^ 7 ≤ (32.36 : ℝ) := by norm_num
    linarith
  unfold ns
  constructor
  · have hinv : (phi ^ 7)⁻¹ ≤ (25.45 : ℝ)⁻¹ := by gcongr
    have : (25.45 : ℝ)⁻¹ ≤ 0.0393 := by norm_num
    linarith
  · have hinv : (32.36 : ℝ)⁻¹ ≤ (phi ^ 7)⁻¹ := by gcongr
    have : (0.0309 : ℝ) ≤ (32.36 : ℝ)⁻¹ := by norm_num
    linarith

end CALConsistencyReal.Ns
