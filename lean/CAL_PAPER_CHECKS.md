# CAL paper checks

Machine checks of the Causal Accumulation Law papers, one folder per paper. Each folder has its
own README saying which test is which.

| Folder | Paper | Lean | Python |
|---|---|---|---|
| `Born_Lorentz_Killing_sheet/` | "Field Theoretic Self-Attention: A Canonicalization of Born, Lorentz, and Killing" (matrix and Clifford editions) | 14 files, 334 theorems | sympy: 67 checks, D1-D18, 104 matrix-free, N1-N3, W1-W4, S1-S8, T1-T7, Q1-Q9 |
| `Schrodinger_edge_alpha1/` | "Schrodinger evolution from the bare Causal Accumulation Law at alpha -> 1^-" and its sheet | `SchrodingerEdge.lean`, 28 theorems | sympy (E), mpmath (A), direct iteration (N), sheet checks G1-G4 |
| `Feynman_alpha2/` | "An Axiomatic Derivation of the Feynman Path Sum at the Light-Cone Kernel Width and Closure Order alpha = 2 of the Causal Accumulation Law" and its sheet | `FeynmanAlphaTwo.lean`, 38 theorems | sympy (E, 95), mpmath (A, 231), simulation (N), sheet checks G1-G4 |
| `CAL_manual_rho_Lambda/` | CAL Unified Manual, Eqs. 78-81 | none | one mpmath script |

Each `lean/` folder is a complete Lean project (Lean 4.35.0, Mathlib v4.35.0) holding only that
paper's files and whatever they import. The full library, all three papers together, is 400
theorems. The repository `JewJitsu11B/Liorhybrid` holds all 400 under `canonicalization/lean`.

The same Lean files are also in `JewJitsu11B/Liorhybrid` under `canonicalization/lean`, as one library.
