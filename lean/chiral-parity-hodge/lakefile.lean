import Lake
open Lake DSL

package ChiralParity where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib ChiralParity where
  srcDir := "."

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.14.0"
