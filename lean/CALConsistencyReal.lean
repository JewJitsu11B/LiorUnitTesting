/-
CALConsistencyReal -- Mathlib-dependent band checks for the irrational/transcendental
predictions. These are the constants we were unsure Mathlib could handle: algebraic
irrationals (sqrt) and the tight-band FSC. Each theorem proves the derived value lands
in its independently-cited experimental band, with no `sorry` and no axioms beyond Mathlib.
-/
import CALConsistencyReal.Ns
import CALConsistencyReal.FineStructureIR
