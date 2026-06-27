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

/-! ## Batch 2: five more rational-prediction constants (pure decide). -/

/-- Heaviest neutrino: m_nu3 = m_e * alpha^3 / 4, alpha = 1/137.036,
    m_e = 510.99895 keV. Exact (meV): 1996089648437500/40209071635979 = 49.6428.
    NuFIT sqrt|dm2_31| = 49.53 +- 0.33 meV => band [49.20, 49.86] = [246/5, 2493/50]. -/
theorem mnu3_in_band :
    246 * 40209071635979 ≤ 1996089648437500 * 5
  ∧ 1996089648437500 * 50 ≤ 2493 * 40209071635979 := by decide

/-- CKM atmospheric: sin th23 = A * sin^2 th12 = (5/6)*(41/182)^2 = 8405/198744 = 0.042291.
    PDG 0.04182 +- 0.00085 => band [0.04097, 0.04267]. -/
theorem ckm_th23_in_band :
    4097 * 198744 ≤ 8405 * 100000
  ∧ 8405 * 100000 ≤ 4267 * 198744 := by decide

/-- Charm quark: m_c = 4*Lambda_QCD*C_EW(u)*(211/128), Lambda_QCD = 218 MeV,
    C_EW(u) = 1 - sin^2thW/2 = 1 - 0.23122/2. Exact (MeV): 2034008561/1600000 = 1271.26.
    PDG 1273 +- 4.6 => band [1268.4, 1277.6] = [6342/5, 6388/5]. -/
theorem mc_in_band :
    6342 * 1600000 ≤ 2034008561 * 5
  ∧ 2034008561 * 5 ≤ 6388 * 1600000 := by decide

/-- Top quark: m_t = m_H*C_EW(u)*(42/27 + 1/882), m_H = v*28/55 (v = 246.22).
    Exact (GeV): 1494884682317/8662500000 = 172.570. PDG 172.52 +- 0.33
    => band [172.19, 172.85] = [17219/100, 3457/20]. -/
theorem mt_in_band :
    17219 * 8662500000 ≤ 1494884682317 * 100
  ∧ 1494884682317 * 20 ≤ 3457 * 8662500000 := by decide

/-- Second neutrino: m_nu2 = m_nu3 / sqrt(R), R = dm2_31/dm2_21 = 9728/297 (B15).
    m_nu2 in [lo,hi] <=> lo^2*R <= m_nu3^2 <= hi^2*R (all positive), a RATIONAL check
    by squaring -- no sqrt needed. m_nu2 = 8.674 meV; NuFIT 8.678 +- 0.104
    => band [8.574, 8.782] meV. (m_nu3^2 numerator = 1996089648437500^2.) -/
theorem mnu2_in_band :
    1241560928 * 1616769441827290935487288441
      ≤ 3984373884599342346191406250000 * 515625
  ∧ 3984373884599342346191406250000 * 4640625
      ≤ 11722775648 * 1616769441827290935487288441 := by decide

end CALConsistency.Bands
