import BornRule.L41_Complementarity
import BornRule.L43_NullIdempotents
import BornRule.L45_Interference
import BornRule.P52_Closure
import BornRule.T61_Born
import BornRule.Corollaries

/-!
# Axiom audit (no-cheating gate)

`#print axioms` on every capstone. A clean result shows only
`[propext, Classical.choice, Quot.sound]` — the standard Mathlib base. The gate fails if the
placeholder axiom appears, which would mean a proof secretly used an unfinished-proof stub.
-/

namespace BornRule

-- Theorem 6.1 (Born rule)
#print axioms born_plus_plus
#print axioms born_normalization
#print axioms trace_EmEn
-- Prop 5.2 (branch collapse = dynamical uniqueness)
#print axioms closure_forces_real
#print axioms closure_expand
-- Lemma 4.3 (null idempotents)
#print axioms Pbranch_idem
#print axioms Pbranch_orthogonal
#print axioms Pbranch_complete
#print axioms redNorm_Jbranch_zero
#print axioms En_dagger
-- Lemma 4.1 (complementarity) / 4.5 (interference)
#print axioms redNorm_mul
#print axioms dagger_form_posSemidef
#print axioms scalar_interference
#print axioms trace_commutator
-- Corollaries (review findings A, B closed)
#print axioms born_prob_mem_Icc
#print axioms dagger_form_trace_pos
#print axioms dot_bounds

end BornRule
