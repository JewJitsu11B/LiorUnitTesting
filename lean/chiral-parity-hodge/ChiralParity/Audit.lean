import ChiralParity.StarSquare
import ChiralParity.EqualSplit
import ChiralParity.ParityCriterion
import ChiralParity.RankSplit
import ChiralParity.Forcing
import ChiralParity.Section9
import ChiralParity.Corollary8
import ChiralParity.BlockEigen

/-!
# Axiom audit (no-cheating gate)

`#print axioms` on every capstone. A clean result shows only
`[propext, Classical.choice, Quot.sound]` — the standard Mathlib base — and no placeholder axiom.
-/

namespace ChiralParity

-- Lemma 3.1 (star square) and its combinatorial helpers
#print axioms starMat_sq
#print axioms eps_mul_eps_compl
#print axioms inv_add_inv_compl
-- Proposition 4.1 (equal split)
#print axioms trace_starMat
#print axioms trace_Pplus_eq_trace_Pminus
-- Theorem 3.2 (parity criterion + exact failure constants) and Corollary 3.3
#print axioms Pplus_sq_of_even
#print axioms Pplus_Pminus_of_even
#print axioms Pplus_sq_sub_of_odd
#print axioms Pplus_Pminus_of_odd
#print axioms starScalar_eq_one_iff
#print axioms carrier_condition
-- Corollary 9.7 / Theorem 9.10 (signature forcing to (4,4))
#print axioms mw_iff_mod8
#print axioms forcing_n8
#print axioms forcing_n8_full
-- Proposition 4.1 concrete rank split 70 = 35 ⊕ 35
#print axioms rank_eq_trace_of_idempotent
#print axioms rank_Pplus_n8
#print axioms rank_split_n8
-- Section 9: Weyl side computable (no axiom); forcing on volume-element sign + CITED ABS axiom
#print axioms weyl_iff_wsign
#print axioms forcing_n8_from_clifford
-- Corollary 8.1 (classical 4d case) and Proposition 5.1 / Cor 5.2 (block eigenvectors)
#print axioms rank_split_4d
#print axioms star4d_sq_odd
#print axioms star_eigen
#print axioms handedness_depends_on_block

end ChiralParity
