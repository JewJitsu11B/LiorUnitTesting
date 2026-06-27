/-
Rational constant checks [ARITH], denominator-cleared to Nat so every check is a
pure `decide` (no Mathlib, no `Rat`, no axioms). Each theorem encodes an exact
arithmetic claim from the derivations paper; a wrong value fails to compile.

These mechanize part of the June numeric audit. In particular `alphaLambda_*`
catches B1 and `nu2_ratio_*` catches B15 -- the build refutes the stale values.
-/

namespace CALConsistency.Const

/-! ## alpha_Lambda (cosmological constant exponent) -- audit finding B1.

The octonionic exponent is `2 - 1/280 + 1/14000 = (559 + 1/50)/280 = 27951/14000`,
which is EXACTLY `1.9965`. The paper printed `1.9965071`, which is wrong, and the
"7e-6 octonionic-vs-empirical" gap is an artifact of that typo. -/

/-- `2 - 1/280 + 1/14000 = 27951/14000` (combine over 14000: 28000 - 50 + 1). -/
theorem alphaLambda_octonionic : 28000 - 50 + 1 = 27951 := by decide

/-- `(559 + 1/50)/280 = 27951/14000` (numerator 559*50 + 1 = 27951; denom 50*280). -/
theorem alphaLambda_fraction : 559 * 50 + 1 = 27951 ∧ 50 * 280 = 14000 := by decide

/-- `alpha_Lambda = 1.9965` exactly: cross-multiply 27951/14000 = 19965/10000. -/
theorem alphaLambda_eq_1p9965 : 27951 * 10000 = 19965 * 14000 := by decide

/-- The printed `1.9965071` is WRONG: 27951/14000 ≠ 19965071/10000000. -/
theorem alphaLambda_ne_typo : 27951 * 10000000 ≠ 19965071 * 14000 := by decide

/-- The intermediate algebra slip `(558 + 1/50)/280` (B13) gives 27901/14000,
    which is NOT 27951/14000 -- the simplification `560-2 = 558` was wrong. -/
theorem alphaLambda_intermediate_wrong : 558 * 50 + 1 ≠ 27951 := by decide

/-! ## Neutrino mass-squared splitting ratio -- audit finding B15.

The constraint-forced ratio is `512/(16 - 7/19) = 9728/297` (correction carries the
e<->nu2 signature 19 = 26 - 7, sits INSIDE the effective Cayley dimension, subtractive
internal sign). The competing `237/7 = 32 + 26/14` and the uncorrected `32` are wrong. -/

/-- Effective Cayley denominator: `16 - 7/19 = 297/19` (16*19 - 7 = 297). -/
theorem nu2_eff_denom : 16 * 19 - 7 = 297 := by decide

/-- Forced ratio numerator: `512/(297/19) = 9728/297` (512*19 = 9728). -/
theorem nu2_ratio_forced : 512 * 19 = 9728 := by decide

/-- The competing form `32 + 26/14 = 237/7` (32*7 + 13 = 237, since 26/14 = 13/7). -/
theorem nu2_ratio_alt : 32 * 7 + 13 = 237 := by decide

/-- The two are INCONSISTENT: 9728/297 ≠ 237/7 (9728*7 = 68096 ≠ 70389 = 237*297). -/
theorem nu2_ratio_inconsistent : 9728 * 7 ≠ 237 * 297 := by decide

/-! ## Electron mass -- `m_e = 512 - 1 - 21/20000 = 510.99895 keV`. -/

/-- Residual numerator `21 = 1 + 4 + 16` (observer + spacetime + spinor). -/
theorem me_residual_num : 1 + 4 + 16 = 21 := by decide

/-- Residual denominator `20000 = (4+16) * (3+7)^3 = 20 * 1000`. -/
theorem me_residual_den : (4 + 16) * (3 + 7) ^ 3 = 20000 := by decide

/-- `m_e = 511 - 21/20000 = 10219979/20000 = 510.99895` (cross-mult vs 51099895/100000). -/
theorem me_eq : (511 * 20000 - 21) * 5 = 51099895 := by decide

/-! ## Strong coupling -- `alpha_s^{-1}(MZ) = 8 + 1/2 - 1/21 = 355/42`. -/

/-- Combine over 42: `8 + 1/2 - 1/21 = (336 + 21 - 2)/42 = 355/42`. -/
theorem alphaS_inv : 336 + 21 - 2 = 355 := by decide

/-! ## IR fine-structure skeleton -- `alpha_EM^{-1}(0) = 133 + 4 + 1/D_eff`.
    The rational skeleton; the irrational `1/sqrt2` part of D_eff is checked in the
    Mathlib `Real` module. -/

/-- Dominant term `dim(E7) + spacetime = 133 + 4 = 137`. -/
theorem alphaEM_dominant : 133 + 4 = 137 := by decide

/-- The finite-channel weight `133 * 61 = 8113` (CC-2 inter-rung, see Dim.inter_rung_weight). -/
theorem alphaEM_inter_rung : 133 * 61 = 8113 := by decide

end CALConsistency.Const
