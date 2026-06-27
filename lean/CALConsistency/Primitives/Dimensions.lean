/-
Dimension identities [INT].

These are the genuine arithmetic the derivations rely on. The dimension VALUES
(dimE7 = 133, etc.) are established-math inputs -- Cartan classification of the
exceptional Lie algebras, Clifford 2^n, the exceptional Jordan algebra. They enter
as `[established-math]` inputs; the THEOREMS below are the non-trivial integer
identities the framework asserts, each discharged by `decide` (no Mathlib, no axioms
of our own). If a value were wrong, the corresponding `decide` would fail to compile.
-/

namespace CALConsistency.Dim

/-- Established-math dimension inputs (Cartan / Clifford / Jordan). -/
def dimE8  : Nat := 248
def dimE7  : Nat := 133
def dimE6  : Nat := 78
def dimF4  : Nat := 52
def dimG2  : Nat := 14
def dimImO : Nat := 7      -- imaginary octonions
def dimImH : Nat := 3      -- imaginary quaternions
def dimh3O : Nat := 27     -- exceptional Jordan algebra h3(O)
def dimCl9 : Nat := 512    -- Cl(9) = 2^9
def dimOP2 : Nat := 16     -- Cayley plane F4/Spin(9)

/-! ## Derived identities the framework uses. -/

/-- Higgs coset: dim(E7/E6) = 133 - 78 = 55. -/
theorem coset_E7_E6 : dimE7 - dimE6 = 55 := by decide

/-- Half the E7 fundamental: dim(56)/2 = 28. -/
theorem fund56_half : 56 / 2 = 28 := by decide

/-- Clifford closure: dim(Cl 9) = 2^9. -/
theorem cl9_pow : dimCl9 = 2 ^ 9 := by decide

/-- Rank-2 forcing / G2: n(n-3)/2 = 14 at n = 7 = dim(Im O). -/
theorem rank2_forcing : dimImO * (dimImO - 3) / 2 = dimG2 := by decide

/-- Traceless exceptional Jordan algebra: 27 = 26 + 1. -/
theorem jordan_traceless : dimh3O = 26 + 1 := by decide

/-- Grade-4 of Cl(9): C(9,4) = 9*8*7*6/(4*3*2*1) = 126. -/
theorem grade4_Cl9 : 9 * 8 * 7 * 6 / (4 * 3 * 2 * 1) = 126 := by decide

/-- Completion index cubed: (3+7)^3 = 1000. -/
theorem completion_cube : (dimImH + dimImO) ^ 3 = 1000 := by decide

/-- E6/F4 coset: 78 - 52 = 26. -/
theorem E6F4_coset : dimE6 - dimF4 = 26 := by decide

/-- The e<->nu2 exclusivity signature: 26 - 7 = 19 (the same 19 in electron p1 = 19/26).
    This is the licensed correction structure for the nu2 sector (B15). -/
theorem e_nu2_signature : (dimE6 - dimF4) - dimImO = 19 := by decide

/-- The CC-2 inter-rung weight denominator: 133 * 61 = 8113. -/
theorem inter_rung_weight : dimE7 * 61 = 8113 := by decide

/-- 61 = 56 + 4 + 1 (E7 fundamental + spacetime + observer scalar). -/
theorem horizon_61 : 56 + 4 + 1 = 61 := by decide

end CALConsistency.Dim
