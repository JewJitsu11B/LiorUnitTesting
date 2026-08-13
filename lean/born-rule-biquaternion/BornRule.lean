import BornRule.Carrier
import BornRule.Basic
import BornRule.L41_Complementarity
import BornRule.L43_NullIdempotents
import BornRule.L45_Interference
import BornRule.P52_Closure
import BornRule.T61_Born
import BornRule.Corollaries

/-!
# BornRule: a machine-checked derivation of the projective Born rule at α=2

Formalizes the null-idempotent source-closure core of
"A Biquaternionic Source-Closure Derivation of the Born Rule ..." (Leizerman, Aug 2026),
in the carrier `C ⊗ H = M₂(ℂ)`. See `BornRule/Audit.lean` for the axiom gate.
-/
