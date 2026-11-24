# Clifford edition: native checks. Briefs for two agents.

Durable copy so a cancelled run can be relaunched. Keep until the work is reported.

## COMMON

Context. A paper exists in two editions that say the same thing line by line: a Matrix edition
in M2(C) and a Clifford edition in Cl(3,0). So far every machine check (Lean 4 + Mathlib, and a
sympy script) was done on the MATRIX edition, and the Clifford edition was covered only through
a dictionary (reversion = dagger, Clifford conjugate = adjugate, N(q) = q qbar = det,
2<q>_0 = Re tr, paravectors = Herm(2), bivectors = su(2), pseudoscalar I = i) that is checked
in sympy only, and there by writing Cl(3,0) as Pauli matrices. The goal now: check the
Clifford edition NATIVELY, so its statements stand without going through matrices, and prove
the dictionary itself.

Sources to read (read-only):
- `C:\Users\Sam Leizerman\cal-checks\paper\FieldTheoretic_SelfAttention_Born_Lorentz_Killing_Clifford.tex`
  (the statements, in Clifford notation) and `..._Matrix.tex` (same blocks, matrix notation).
- `C:\Users\Sam Leizerman\cal-checks\python\check_onesheet_sympy.py` (67 matrix checks, labels
  Q1, B0-B14, L1-L17, P1-P6, K1-K10, C1-C6, U1-U4, R1-R7, X1),
  `check_clifford_dictionary.py` (D1-D18), `check_naturality.py` (N1-N3),
  `check_area_killing.py` (W1-W4), `threeway\A3_clifford.py` (a native Cl(3,0) implementation
  you may reuse).
- Lean project: `C:\Users\Sam Leizerman\cal-checks\lean` (Lean 4.35.0, Mathlib v4.35.0,
  PhysLean as dependency `Physlib`, already built). `CalChecks\Clifford.lean` defines
  `Cl3 := CliffordAlgebra Q3`, generators `γ i`, `cliffordToM2`, `cliffordEquiv : Cl3 ≃ₐ[ℝ] M2`.

Ground rules.
- Do not assume anything is true. A statement that fails natively, needs an extra hypothesis,
  or cannot be stated natively is a valuable result: report it plainly.
- Create NEW files only. Do not edit or delete any existing file. Do not touch the paper
  sources. No git commands that change history. Do not work in WSL and do not run
  `lake update`, `lake build` of dependencies, or anything that downloads packages.
- Windows machine; run `python` from Git Bash or PowerShell; sympy and mpmath installed.
- No emojis anywhere.
- Say only what you actually ran. Final message: a report under 400 words with
  VERDICT, WHAT RAN (commands and key output), WHAT FAILED OR WAS NOT DONE, FILES.
  Also write that report to the results file named in your section.

## S: native Cl(3,0) sympy suite

Write `C:\Users\Sam Leizerman\cal-checks\python\check_clifford_native.py`.

Implement Cl(3,0) as an 8-dimensional real algebra directly from the generator relations
(multiplication table on basis blades computed by your own code). NO 2x2 matrices anywhere in
the checks. Implement reversion, grade involution, Clifford conjugation, grade projection,
N(q) = q qbar, the commutator, the pseudoscalar I, the channel units e_0 = 1, e_j = -I s_j
(bivectors), and the tensor product Cl3 (x) Cl3 (64-dimensional) for the Bell block. Killing
forms must be computed from tr(ad ad) on a basis, not from a shortcut.

Port EVERY check of the matrix suite to its Clifford form, keeping the same label, using the
Clifford edition's own wording of each statement (dagger -> reversion, det -> N,
(1/2) Re tr -> scalar part, Hermitian -> paravector, anti-Hermitian -> grades 3+2,
i -> I, U(2) -> elements with U Utilde = 1, SL(2,C) -> elements with N = 1, and so on).
Also port N1-N3, W1-W4 and the three-way relation N(X) = <X>_0^2 + (1/8) kappa(IX, IX).
Where N(q) is a complex number in the matrix edition it is an element of the centre
span{1, I} natively: say how you handle that.
Checks that do not mention the algebra at all (R1-R5 character arithmetic, R7, C4-C6
if purely numeric) should be rerun unchanged and marked `algebra-independent` in the output.
If a check has no native Clifford statement, mark it `NOT PORTED` with the reason; do not
fake it.

Then a separate, clearly labelled final block `REP`: the only place matrices may appear.
Verify that your native product agrees with the Pauli representation on all 64 blade pairs
(that is, the map blades -> Pauli words is an algebra homomorphism and the 8 images are
real-independent). This ties the native suite to the matrix suite.

Output one line per check `PASS|FAIL|NOT PORTED <label> <text>`, a final count, exit non-zero
on any FAIL. Results file: `C:\Users\Sam Leizerman\cal-checks\python\clifford_native\results\S.md`.

## L: Lean 4 + Mathlib + PhysLean, native Clifford theorems

Work in `C:\Users\Sam Leizerman\cal-checks\lean`. Put everything in ONE new file
`CalChecks\CliffordNative.lean` (namespace `CalChecks`), importing `Mathlib` and the existing
`CalChecks` modules you need. Check with `lake env lean CalChecks\CliffordNative.lean` run from
the `lean` folder (the existing modules are already built; if an import is reported missing,
report that instead of rebuilding dependencies). No `sorry`, no new axioms; end the file with
`#print axioms` for each main theorem and report the output.

Use Mathlib's own `CliffordAlgebra.reverse` and `CliffordAlgebra.involute` on `Cl3`. Define
Clifford conjugation as reverse of involute. Define grade parts intrinsically from the two
involutions (for example scalar part = the grade-0 component obtained from
(q + reverse q)/2 and (q + conj q)/2), not by pulling back matrix operations; say exactly
which definitions you used.

Targets, in priority order. Do as many as compile; report the rest as not done.

1. The dictionary, as theorems about `cliffordEquiv` (or `cliffordToM2`):
   a. `cliffordEquiv (reverse q) = (cliffordEquiv q)ᴴ`.
   b. `cliffordEquiv (conj q) = adjugate (cliffordEquiv q)`.
   c. `cliffordEquiv (q * conj q) = det (cliffordEquiv q) • 1`.
   d. scalar part: `2 * <q>_0 = Re tr (cliffordEquiv q)`; grades 0+1 = Hermitian part,
      grades 3+2 = anti-Hermitian part.
   e. pseudoscalar `I = γ 0 * γ 1 * γ 2` is central, `I * I = -1`, maps to `Complex.I • 1`.
2. Clifford-form statements whose STATEMENT mentions no matrix (the proof may transport
   through `cliffordEquiv`, which is fine, since it is injective):
   a. closure `c J = J * reverse J` is reverse-fixed (lands in grades 0+1), and
      `c (G * J) = G * c J * reverse G`.
   b. `N (a * b) = N a * N b` with `N q = q * conj q`, `N q` central.
   c. paravector norm: `N (t + x γ0 + y γ1 + z γ2) = t^2 - x^2 - y^2 - z^2` (as a scalar).
   d. `N (I * X) = - N X`.
   e. bivectors: `N A = - A * A` for A in the span of γiγj.
   f. real Killing form of `Cl3` under the commutator (Mathlib `killingForm ℝ`), its value on
      vectors (positive) and bivectors (negative); the project already has `killingForm_real`
      and related theorems on the matrix side in other `CalChecks` files, reuse them.
   g. three-way relation in Clifford form: for a paravector X,
      `N X = <X>_0^2 + (1/8) κ(I X, I X)` with the same κ as in `C2_ThreeWay.lean`
      (see `C:\Users\Sam Leizerman\cal-checks\python\threeway\C2_ThreeWay.lean`), or state
      precisely which κ you used.
3. PhysLean. `CalChecks\Crosscheck.lean` cross-checks the project's Pauli matrices against
   PhysLean's. Extend that to the Clifford side in your new file: the images of `γ 0, γ 1, γ 2`
   under `cliffordToM2` are PhysLean's Pauli matrices. Then look in
   `.lake\packages` for any Clifford-algebra development inside PhysLean itself; if there is
   one that covers Cl(3,0) or can be related to it, cross-check against it; if PhysLean only
   has a different Clifford algebra (for example the Dirac/spacetime one), say so plainly and
   do not force a connection.

Results file: `C:\Users\Sam Leizerman\cal-checks\python\clifford_native\results\L.md`.
List every theorem name you proved with a one-line plain statement.
