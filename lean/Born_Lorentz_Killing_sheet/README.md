# Born, Lorentz, Killing sheet: checks

Paper: "Field Theoretic Self-Attention: A Canonicalization of Born, Lorentz, and Killing", the
five-page sheet. Two editions that say the same thing block for block:

- `paper/FieldTheoretic_SelfAttention_Born_Lorentz_Killing_Matrix.tex/.pdf` (in M2(C))
- `paper/FieldTheoretic_SelfAttention_Born_Lorentz_Killing_Clifford.tex/.pdf` (in Cl(3,0))

Each step of the sheet ends with a bracketed tag naming the Lean theorems and sympy labels below.

## Layout

| Folder | What it holds |
|---|---|
| `paper/` | Both editions, LaTeX and PDF |
| `lean/` | Lean 4 project: 14 files, 334 theorems |
| `python/` | sympy checks, one script per group of labels |
| `python/threeway/` | Nine independent tests of the three-readout relation, with their briefs and reports |
| `python/clifford_native/` | Briefs and reports for the matrix-free Clifford suite and its Lean file |

## Lean (`lean/CalChecks/`)

Lean 4.35.0, Mathlib v4.35.0, PhysLean. No `sorry`, no `native_decide`, no custom axioms.

| File | Theorems | What it proves |
|---|---|---|
| `Basic.lean` | 42 | Carrier, quaternion relations, Born quadratic, Hermitian slice and Minkowski determinant, purity, Lagrange identity, CHSH, uniqueness of alpha = 2 |
| `Extras.lean` | 12 | Bell correlation E = -a.b, Killing form of gl(2), Cl(3,0) map to the Pauli matrices |
| `RepTheory.lean` | 23 | Characters and weights for Clebsch-Gordan, SO(4) by left and right quaternion multiplication, real Killing signature, fractional integral eigenfunction |
| `Complete.lean` | 49 | Carrier bijection, cone stabilizer, POVM rows, purity from channels, SU(2) to SO(3), Tsirelson bound |
| `Structure.lean` | 11 | Kernel {+1, -1} of the spinor map, Mathlib `killingForm` over C and R, positive cone as one congruence orbit |
| `Clifford.lean` | 28 | Cl(3,0) isomorphic to M2(C) as real algebras, dimension 8 |
| `Final.lean` | 7 | 1 + 3 split under conjugation, boost dependence of purity, the Born rule as a kinematic theorem (`born_kinematic`) |
| `SO4.lean` | 32 | (Sp(1) x Sp(1)) / {+1, -1} isomorphic to SO(4): every rotation of the four channels is x -> p x q for unit quaternions, kernel exactly (1,1) and (-1,-1) |
| `Crosscheck.lean` | 3 | Agreement with PhysLean's Pauli matrices and SL(2,C) results; prints the axioms of the theorems in the nine files above |
| `ThreeWay.lean` | 12 | T1 in Lean: `det X = (tr X)^2/4 + (1/8) kappa(iX, iX)` with Mathlib's `killingForm`, for all complex 2x2 over C and for Hermitian X with the real form on su(2) |
| `CliffordNative.lean` | 72 | The matrix-to-Clifford dictionary as theorems (reversion is dagger, Clifford conjugate is adjugate, `q * conj q` is the determinant, scalar part is half the real trace), then the Clifford-form statements: closure, norm, sign flip under the pseudoscalar, real Killing form, three-readout relation |
| `CliffordPhyslib.lean` | 3 | PhysLean's own Cl(3,0): same quadratic form, same map to the Pauli matrices |
| `Biquaternion.lean` | 29 | Biquaternions, M2(C) and Cl(3,0) are one algebra: `C (x) H = H[C] = M2(C) = Cl(3,0)` as algebra isomorphisms, with the map `H[C] -> M2(C)` being the carrier itself. Dictionary: quaternion conjugate is adjugate is Clifford conjugate; coefficient conjugate is grade involution; both together are dagger and reversion; quaternion norm is det is `N`; `i, j, k` are the unit bivectors; the complex unit is the pseudoscalar; which coefficient part lands in which grade |
| `Naturality.lean` | 11 | The commuting squares in Mathlib's category-theory types: the closure `c(J) = J J^dagger` as a `NatTrans` between `GL(2,C)`-sets (`closureNat`) and as a morphism in `Action (Type) GL(2,C)` (`closureAction`); trace, det and the Killing quadratic as `NatTrans` into the constant functor for `U(2)`, `SL(2,C)` and `GL(2,C)` under Ad |

`lean/PhyslibDraft/` is a separate draft (quaternion isometries and SO(4)) written toward an
upstream PhysLean contribution, with `AUTHOR_NOTES.md`. It is not part of the 334.

The sheet's own tags refer to the first 294; `Biquaternion.lean` and `Naturality.lean` were added afterwards.

To build:

```
cd lean
lake exe cache get
lake build CalChecks
```

## Python (`python/`)

All exact sympy unless stated. Each script prints one line per check and exits non-zero on failure.

| Script | Labels | What it tests |
|---|---|---|
| `check_onesheet_sympy.py` | 67 checks | Every algebra and representation-theory step of the matrix edition, in M2(C) |
| `check_clifford_dictionary.py` | D1-D18 | Each Clifford operation against its matrix counterpart, in the Pauli representation |
| `check_clifford_native.py` | 104 checks | The same labels again in Cl(3,0) built from the generator relations, with no matrices until a final block `REP` that ties the product to the Pauli representation |
| `check_naturality.py` | N1-N3 | No linear equivariant map from the carrier to the Hermitian slice; `J J^dagger` is the only quadratic one invariant under right U(2), up to scale |
| `check_area_killing.py` | W1-W4 | The area element of a plane of vectors is the square root of a Killing norm |
| `check_schild.py` | S1-S8 | The squared area element against its root: Schild's integrand, its unit-Jacobian invariance, constant area density, the Nambu-Goto equations, a rotating string |
| `check_threeway.py` | T1-T3 | The three-readout relation `det X = (tr X)^2/4 + (1/8) kappa(iX, iX)` in matrix and Clifford form; runs `threeway/A1_check.py`, `A2_check.py`, `A3_clifford.py` |
| `check_radius.py` | T4-T7 | The relation solved for the radius (T4); rank one, where Born and Killing readouts coincide (T5); conjugation keeps all three readouts (T6); congruence keeps only the determinant, a boost trades trace for Killing form, SU(2) keeps all three (T7). Each in matrix form and in Cl(3,0) without matrices |
| `check_biquaternion.py` | Q1-Q9 | The same chain in sympy with the three algebras built independently (Hamilton's rules, matrices, blades): each map multiplicative and bijective, and involutions, norm, scalar part, units and grades carried to their partners; 18 checks |
| `check_editions_isomorphic.py` | structure | Not mathematics: checks that the two `.tex` editions have the same sequence of blocks and the same tag after each. Takes the two `.tex` paths as arguments |

```
cd python
python check_onesheet_sympy.py
python check_editions_isomorphic.py ../paper/FieldTheoretic_SelfAttention_Born_Lorentz_Killing_Matrix.tex ../paper/FieldTheoretic_SelfAttention_Born_Lorentz_Killing_Clifford.tex
```

### `python/threeway/`: nine tests of the three-readout relation

Written by nine independent runs from `BRIEFS.md`; each report is in `results/`.

| File | Method |
|---|---|
| `A1_check.py` | Exact sympy in coordinates; Killing form built from ad matrices of su(2) |
| `A2_check.py` | Exact sympy starting from the carrier J with eight real parameters, X = J J^dagger |
| `A3_clifford.py` | Exact, in Cl(3,0) built from generator relations, no matrices for algebra elements |
| `B1_check.py` | 10000 random cases at 50 digits with mpmath; also tests the scope (non-Hermitian X) |
| `B2_mutation.py` | Mutation test: changes each constant and sign and confirms the altered relation fails; counterexample hunt |
| `B3_check.py` | Exact Gaussian rationals with the standard library only (no sympy, numpy or mpmath) |
| `C1_proof.md`, `C1_check.py` | Proof by hand, and one assertion per numbered step of it |
| `C2_ThreeWay.lean` | The Lean proof as first written; the library version is `lean/CalChecks/ThreeWay.lean` |
| `C3_proof.md`, `C3_check.py` | Invariant-theory proof, each ingredient checked exactly |

### `python/clifford_native/`

`BRIEFS.md` and `results/` (S: the sympy suite that became `check_clifford_native.py`; L: the Lean
file that became `CliffordNative.lean`), with the run logs.

## What is checked where

- In Lean and sympy: the algebra of the sheet, the dictionary, T1.
- sympy only: N1-N3, W1-W4, S1-S8, T2-T7, the complex Killing form in Clifford form.
- Not machine-checked: Zeeman's theorem; Clebsch-Gordan as a decomposition (characters and
  weights only); the dynamics; the forward lemma; the remarks marked as readings.
