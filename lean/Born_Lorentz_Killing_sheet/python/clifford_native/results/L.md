# Agent L: Lean 4 + Mathlib + PhysLean, native Clifford theorems

## VERDICT

All targets 1a-1e, 2a-2g and the generator part of 3 compile in one new file, with no `sorry`
and no new axioms. Two findings that qualify the paper's wording:

1. Three-way relation (2g): with Mathlib's `killingForm ℝ Cl3` (Cl3 as an 8-dimensional REAL
   Lie algebra) the constant is 1/16, not 1/8 (`threeway_Cl3_real`). The constant 1/8 holds
   for the Killing form of the 3-dimensional bivector Lie algebra `spin3`, evaluated on
   `I * (vector part of X)` (`threeway_spin3`; same kappa as `threeway_su2` in
   `C2_ThreeWay.lean`). The paper's kappa "as a complex Lie algebra" was NOT stated natively:
   that needs a complex structure on Cl3 through the centre span{1, I}; not done.
2. `N q` is not a real number natively: it is an element of the centre,
   `N q = Re(det) + Im(det) I` (`N_eq`). It is a real scalar on paravectors and bivectors.

PhysLean: the images of `γ 0, γ 1, γ 2` are PhysLean's Pauli matrices (proved). PhysLean does
contain a Cl(3,0) development (`Physlib/Relativity/PauliMatrices/CliffordAlgebra.lean`,
`PauliMatrix.ofCliffordAlgebra` over `PauliMatrix.form`), but that module is not built in
this project (no .olean), so no cross-check against it was made. Its other Clifford file
(`Physlib/Relativity/CliffordAlgebra.lean`) is the Dirac algebra Cl(1,3); no connection forced.

## WHAT RAN

From `C:\Users\Sam Leizerman\cal-checks\lean`:

- `lake env lean CalChecks/CliffordNative.lean` : first run 2 errors (fixed), second run
  0 errors, lint warnings only (unused `<;>` at lines 445, 448). All 48 `#print axioms` lines
  read `[propext, Classical.choice, Quot.sound]`; no `sorryAx`.
- `lake env lean <scratch>/PhysCliff.lean` (one line importing
  `Physlib.Relativity.PauliMatrices.CliffordAlgebra`): error, object file
  `...\PauliMatrices\CliffordAlgebra.olean` does not exist.

## WHAT FAILED OR WAS NOT DONE

- Complex Killing form of Cl3 stated natively: not done (see finding 1).
- Cross-check against PhysLean's own Cl(3,0) file: not done, module not built; not rebuilt,
  per the ground rules.
- 2g with the 3-dimensional kappa uses `I * X_1` (vector part), since `I * X` is not a
  bivector when the scalar part is non-zero. With the 8-dimensional kappa `I * X` is used as
  written and the centre drops out.
- 2c, 2e, 2f, 2g are stated on coordinate families (`pre t x y z`, `biv a b c`), not for an
  abstract "paravector"/"bivector" predicate.
- Proofs of part 2 transport through `cliffordToM2` (injective); only 2a, the grade
  decomposition and `N_biv` are proved without matrices.

## FILES

- `C:\Users\Sam Leizerman\cal-checks\lean\CalChecks\CliffordNative.lean` (new)
- `C:\Users\Sam Leizerman\cal-checks\python\clifford_native\results\L.md` (this file)

## DEFINITIONS USED (all intrinsic to Cl3, Mathlib `reverse` and `involute`)

- `cconj q = reverse (involute q)`; `N q = q * cconj q`
- `gradeZero q = (1/4)(q + reverse q + involute q + cconj q)`; `gradeOne` (+,+,-,-),
  `gradeTwo` (+,-,+,-), `gradeThree` (+,-,-,+) likewise
- `par q = (1/2)(q + reverse q)` (grades 0+1); `apar q = (1/2)(q - reverse q)` (grades 3+2)
- `Ips = γ 0 * γ 1 * γ 2`; `cl J = J * reverse J`
- `biv a b c = a γ1γ2 + b γ2γ0 + c γ0γ1`; `pre t x y z` (existing) `= t + x γ0 + y γ1 + z γ2`
- `spin3 = {X | reverse X = -X and involute X = X}` as a `LieSubalgebra ℝ Cl3`

## THEOREMS PROVED (all compiled)

Dictionary
- `toM2_reverse` / `dict_reverse`: cliffordEquiv (reverse q) = (cliffordEquiv q)ᴴ.
- `toM2_involute` / `dict_involute`: cliffordEquiv (involute q) = (adjugate (cliffordEquiv q))ᴴ.
- `toM2_cconj` / `dict_cconj`: cliffordEquiv (cconj q) = adjugate (cliffordEquiv q).
- `toM2_N` / `dict_N`: cliffordEquiv (q * cconj q) = det (cliffordEquiv q) • 1.
- `toM2_gradeZero`, `gradeZero_eq`: gradeZero q = the real scalar (1/2) Re tr (cliffordEquiv q).
- `dict_scalar`: 2 • gradeZero q = algebraMap (Re tr (cliffordEquiv q)).
- `dict_par`: cliffordEquiv (par q) = Hermitian part H.
- `dict_apar`: cliffordEquiv (apar q) = anti-Hermitian part A.
- `grade_sum`: gradeZero + gradeOne + gradeTwo + gradeThree = q.
- `par_eq`, `apar_eq`: par = grade 0 + grade 1; apar = grade 2 + grade 3.
- `toM2_Ips` / `dict_Ips`: Ips maps to Complex.I • 1.
- `Ips_central`, `Ips_mem_center`: Ips commutes with every q; lies in the centre.
- `Ips_sq`: Ips * Ips = -1.

Matrix-free statements
- `cl_reverse`: reverse (cl J) = cl J.
- `cl_par`: par (cl J) = cl J (closure lands in grades 0+1).
- `cl_equivariant`: cl (G * J) = G * cl J * reverse G.
- `N_mul`: N (a * b) = N a * N b.
- `N_central`, `N_mem_center`: N q commutes with everything; lies in the centre.
- `N_eq`, `N_mem_span`: N q = Re det + (Im det) • Ips; N q is in span{1, Ips}.
- `N_pre`: N (t + x γ0 + y γ1 + z γ2) = t^2 - x^2 - y^2 - z^2.
- `N_Ips`: N Ips = -1.
- `N_Ips_mul`: N (Ips * X) = -N X, for every X.
- `toM2_biv`, `biv_eq`: biv a b c = Ips * (a γ0 + b γ1 + c γ2).
- `biv_reverse`, `biv_involute`, `biv_cconj`: reverse = -, involute = +, cconj = - on bivectors.
- `N_biv`: N A = -(A * A) on bivectors.
- `N_biv_val`: N (biv a b c) = a^2 + b^2 + c^2.
- `killing_Cl3`: killingForm ℝ Cl3 a b = killingForm ℝ M2 (image a) (image b)
  (via `cliffordLieEquiv` and Mathlib `killingForm_of_equiv_apply`).
- `killing_Cl3_vector`: real Killing form on vectors = +16 (x^2 + y^2 + z^2).
- `killing_Cl3_bivector`: on bivectors = -16 (a^2 + b^2 + c^2).
- `killing_Cl3_orthogonal`: vectors and bivectors are Killing-orthogonal.
- `killingForm_real_gen` (matrix side helper): real Killing form on gen t a b c = -16 |a|^2.
- `killing_Cl3_Ips_pre`: kappa(Ips X, Ips X) = -16 (x^2+y^2+z^2) for paravector X (centre drops out).
- `gradeZero_pre`: gradeZero of a paravector is t.
- `threeway_Cl3_real`: N X = gradeZero X ^2 + (1/16) killingForm ℝ Cl3 (Ips X) (Ips X).
- `biv_mem`, `Ips_vector_mem`: bivectors lie in spin3.
- `bcoords`, `bbas`: linear coordinates and a basis of spin3 (so spin3 is exactly the
  3-dimensional span of the biv a b c).
- `killing_spin3`: killingForm ℝ spin3 = -8 y·z, computed from tr(ad ad) in the basis.
- `killing_spin3_biv`: killingForm ℝ spin3 (biv a b c) (biv a b c) = -8 (a^2+b^2+c^2).
- `N_biv_killing`: N A = -(1/8) kappa_spin3 (A, A) on bivectors.
- `threeway_spin3`: N X = gradeZero X ^2 + (1/8) kappa_spin3 (Ips X_1, Ips X_1), X_1 the vector part.

PhysLean
- `toM2_γ0_physlib`, `toM2_γ1_physlib`, `toM2_γ2_physlib`, `cliffordEquiv_γ_physlib`:
  image of γ i is `PauliMatrix.pauliMatrix (Sum.inr i)`.
