/-
CALConsistency -- internal-consistency ledger for the Causal Accumulation Law
derivations. A green `lake build` means the arithmetic, dimension-counting, and
(later) constraint-satisfaction of the derivations are internally consistent. It is
NOT a claim of physical validity. See README.md.

Core (this library): Mathlib-free, every check a `decide` over Nat. The Mathlib-
dependent real-number band enclosures live in a separate library.
-/
import CALConsistency.Primitives.Dimensions
import CALConsistency.Constants.RationalChecks
