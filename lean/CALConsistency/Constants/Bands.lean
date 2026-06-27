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

/-! ## Mixing angles (rational sin^2 predictions).

Predictions are rational; bands are sin^2 of the cited angle ranges (conservative
rational outer bounds), so each check stays a pure `decide`. Sources: PMNS from
NuFIT 6.0 (IC19, NO); CKM Cabibbo from PDG 2024. -/

/-- PMNS solar: sin^2 th12 = (1/3)(1 - 7/78) = 71/234. NuFIT th12 = 33.68+-0.72 deg
    => sin^2 in [0.296, 0.319]. 71/234 = 0.3034. -/
theorem pmns_th12_in_band :
    296 * 234 ≤ 71 * 1000 ∧ 71 * 1000 ≤ 319 * 234 := by decide

/-- PMNS atmospheric: sin^2 th23 = (1/2)(1 + 1/7) = 4/7. NuFIT th23 = 48.5+0.7-0.9 deg
    => sin^2 in [0.546, 0.573]. 4/7 = 0.5714 (upper octant). -/
theorem pmns_th23_in_band :
    546 * 7 ≤ 4 * 1000 ∧ 4 * 1000 ≤ 573 * 7 := by decide

/-- PMNS reactor: sin^2 th13 = 1/45. NuFIT th13 = 8.52+-0.11 deg
    => sin^2 in [0.0214, 0.0225]. 1/45 = 0.02222. -/
theorem pmns_th13_in_band :
    214 * 45 ≤ 1 * 10000 ∧ 1 * 10000 ≤ 225 * 45 := by decide

/-- CKM Cabibbo: sin th12 = 41/182. PDG lambda = 0.22501+-0.00067
    => [0.224340, 0.225680]. 41/182 = 0.225275. -/
theorem ckm_cabibbo_in_band :
    22434 * 182 ≤ 41 * 100000 ∧ 41 * 100000 ≤ 22568 * 182 := by decide

end CALConsistency.Bands
