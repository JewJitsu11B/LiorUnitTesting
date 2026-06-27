/-
External experimental bands [ARITH band-membership].

Each band below is an INDEPENDENTLY MEASURED value with its cited source -- NOT the
framework's prediction. The theorems prove `prediction ∈ [lo, hi]`, the genuine
non-vacuous check (compute from structure, compare to cited measurement). Only
constants whose PREDICTION is rational appear here, so every check stays a pure
`decide`; the irrational-prediction constants (FSC IR with 1/sqrt2, muon, Y_B, ...)
move to the Mathlib `Real` module.

All inequalities are cross-multiplied to Nat so `decide` discharges them. Provenance
is in each doc comment (the source docs must carry the same citations -- see B18).
-/

namespace CALConsistency.Bands

/-! ## m_e -- CODATA 2022: 510.99895000(15) keV. Prediction 512-1-21/20000 = 510.99895000. -/
/-- m_e prediction lands in the CODATA band [510.99894985, 510.99895015] keV
    (all x 1e8 keV). Prediction = 51099895000. -/
theorem me_in_band :
    51099894985 ≤ 51099895000 ∧ 51099895000 ≤ 51099895015 := by decide

/-! ## alpha_s(M_Z) -- PDG 2024: 0.1180 ± 0.0009. Prediction 42/355 = 0.11830986. -/
/-- 42/355 ∈ [0.1171, 0.1189]: 42*10000 ≥ 1171*355 and 42*10000 ≤ 1189*355. -/
theorem alphaS_in_band :
    1171 * 355 ≤ 42 * 10000 ∧ 42 * 10000 ≤ 1189 * 355 := by decide

/-! ## alpha_EM^{-1}(M_Z) -- PDG 2024 (EW, MSbar 5-flavor): 127.930 ± 0.008.
    Prediction 128 - 1/14 - 1/14000 = 127.9285. -/
/-- 127.9285 ∈ [127.922, 127.938] (all x 1e4): 1279220 ≤ 1279285 ≤ 1279380. -/
theorem alphaEM_MZ_in_band :
    1279220 ≤ 1279285 ∧ 1279285 ≤ 1279380 := by decide

/-! ## m_H -- PDG 2024: 125.25 ± 0.17 GeV. Prediction v*28/55 with v = 246.22 GeV
    = 689416/5500 = 125.3484 GeV (the uncorrected derivations-paper value; the unified
    doc applies the -14*sqrt2/100 residual, which needs the Real module). -/
/-- v*28/55 ∈ [125.08, 125.42] GeV: 689416*100 ≥ 12508*5500 and ≤ 12542*5500. -/
theorem mH_in_band :
    12508 * 5500 ≤ 689416 * 100 ∧ 689416 * 100 ≤ 12542 * 5500 := by decide

/-! ## Delta m^2_31 / Delta m^2_21 -- NuFIT 6.0 (IC19, NO): 32.6 ± 0.9 (ratio-direct).
    Prediction (constraint-forced, B15) 512/(16-7/19) = 9728/297 = 32.754. -/
/-- 9728/297 ∈ [31.7, 33.5]: 9728*10 ≥ 317*297 and 9728*10 ≤ 335*297. -/
theorem nu2_ratio_in_band :
    317 * 297 ≤ 9728 * 10 ∧ 9728 * 10 ≤ 335 * 297 := by decide

end CALConsistency.Bands
