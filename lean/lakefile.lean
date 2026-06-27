import Lake
open Lake DSL

/-
CALConsistency: a Lean 4 internal-consistency ledger for the Causal Accumulation
Law derivations. If this compiles, the arithmetic, dimension-counting, and
constraint-satisfaction of the derivations are internally consistent -- NOT a claim
of physical validity. See README.md for what a green build does and does not prove.

The core library (`CALConsistency`) is intentionally Mathlib-free: the dimension
identities are `decide`-checked Nat facts and the constant checks are `decide`-checked
Rat facts. The Mathlib-dependent real-number band enclosures (sqrt2, phi) live in the
separate `CALConsistencyReal` library so the fast core stays dependency-light.
-/

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.14.0"

package «CALConsistency» where
  leanOptions := #[⟨`autoImplicit, false⟩]

/-- Dependency-free core: dimension identities + rational band checks (decide). -/
@[default_target]
lean_lib «CALConsistency» where

/-- Mathlib-dependent: real-number band enclosures for the irrational/transcendental
    predictions we were unsure Mathlib could handle (FSC tight band, transcendental tail). -/
@[default_target]
lean_lib «CALConsistencyReal» where
