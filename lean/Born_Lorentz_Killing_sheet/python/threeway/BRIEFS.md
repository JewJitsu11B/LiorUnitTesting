# Three-way identity test: briefs for nine agents

Durable copy of the briefs, so a cancelled run can be relaunched agent by agent.
Each agent reads COMMON and its own section only. Results go in `results/<ID>.md`.

## COMMON (every agent)

You are one of nine independent testers. Your job is to find out whether the claim below is
true, under exactly which hypotheses, and with exactly which constants. Do not assume it is
true. A failure, a needed extra hypothesis, or a wrong constant is a valuable result; report
it plainly. Do not coordinate with other agents and do not read other agents' files in this
folder: independence is the point.

### The claim under test

Setting: 2x2 complex matrices. Pauli matrices sx, sy, sz; `x.s` means x1 sx + x2 sy + x3 sz.

1. Three-way identity. Let X be a Hermitian 2x2 matrix (in particular X = J J^dagger for any
   complex 2x2 J). Put X0 = X - (1/2) tr(X) 1 (the traceless part) and A = i X0, which lies in
   su(2). Let kappa be the Killing form of su(2), kappa(A, B) = tr(ad_A ad_B). Then

       det X = (tr X)^2 / 4 + (1/8) kappa(i X0, i X0).

2. Supporting readouts. With X = t 1 + x.s (t real, x real 3-vector):
   tr X = 2t, det X = t^2 - |x|^2, kappa(i X0, i X0) = -8 |x|^2.

3. Normalized form. For tr X nonzero, rho = X / tr X = (1/2)(1 + r.s). Then
   det rho = (1 - |r|^2)/4, tr(rho^2) = (1 + |r|^2)/2, and
   -(1/8) kappa(i(rho - 1/2), i(rho - 1/2)) = |r|^2 / 4.

4. Clifford form (same claim in Cl(3,0), generators s1, s2, s3 with sj sk + sk sj = 2 delta_jk,
   pseudoscalar I = s1 s2 s3). For a paravector X = t + x (scalar plus vector), with
   N(q) = q qbar (qbar the Clifford conjugate) and <q>_0 the scalar part:

       N(X) = <X>_0^2 + (1/8) kappa(I x, I x),

   where I x is a bivector and kappa is the Killing form of the bivector Lie algebra under
   the commutator.

### Ground rules

- The Killing form must be computed from its definition, tr(ad_A ad_B), built from
  commutators in a basis, unless your section says otherwise. Do not start from a shortcut
  formula such as 4 tr(AB); if you use one, derive or verify it first.
- Work only inside `C:\Users\Sam Leizerman\cal-checks\python\threeway\`. Name every file you
  create with your agent ID as a prefix (for example `A1_check.py`). Do not edit or delete
  any existing file anywhere else, do not touch the paper sources, and do not run git
  commit, push, or any command that changes a repository's history.
- Windows machine. Run Python as `python` from Git Bash or PowerShell; sympy and mpmath are
  installed. Do not install packages unless your section says you may.
- No emojis anywhere, in files or in your report.
- Write your result to `C:\Users\Sam Leizerman\cal-checks\python\threeway\results\<ID>.md`
  and also return it as your final message. Keep it under 300 words. Format:

      VERDICT: HOLDS | FAILS | HOLDS WITH CONDITIONS
      STATEMENT TESTED: (the exact statement, with your conventions)
      METHOD: (two or three sentences)
      EVIDENCE: (the command you ran and its key output)
      SCOPE: (what you found about where it holds and where it does not)
      CAVEATS: (anything not checked, anything you are unsure of)
      FILES: (paths you created)

  Say only what you actually ran. If something did not run, say so.

## TEAM A: exact computer algebra (sympy)

### A1: coordinates, Killing form from ad matrices

Write a sympy script. Build a real basis of su(2), compute the 3x3 matrices ad_E by solving
[E, E_k] = sum c_k^j E_j, and define kappa(A, B) = tr(ad_A ad_B) by expanding A and B in that
basis. With symbolic real t, x1, x2, x3 verify parts 1, 2 and 3 of the claim as exact
polynomial or rational identities. Report the value of kappa on your basis elements so the
normalization is visible.

### A2: start from the carrier J, eight real parameters

Write a sympy script. Take J a general complex 2x2 matrix with eight real symbols, form
X = J J^dagger, and verify part 1 as an exact polynomial identity in the eight symbols, with
kappa from tr(ad ad) built from commutators (your own construction). Also verify that X is
Hermitian and positive semidefinite in the sense tr X >= 0 and det X = |det J|^2 >= 0, and
report what part 1 says in terms of J alone (express each of the three terms in J).

### A3: Clifford algebra, no matrices

Write a sympy script that implements Cl(3,0) directly as an 8-dimensional real algebra from
the generator relations (a multiplication table on basis blades, your own code). Do not
represent elements as 2x2 matrices anywhere. Implement reversion, Clifford conjugation,
grade projection, N(q) = q qbar, and the Killing form of the bivector Lie algebra from
tr(ad ad) on a bivector basis. Verify part 4 symbolically. Check also that N(X) is a pure
scalar on paravectors, and report what kappa is on unit bivectors.

## TEAM B: numerical and adversarial testing

### B1: high-precision random testing and scope

Write an mpmath script at 50 digits. Test part 1 on at least 10,000 random inputs in each of
these classes, with kappa from tr(ad ad): (a) X = J J^dagger for random complex J;
(b) random Hermitian X, including indefinite and negative ones; (c) rank-one and zero X;
(d) very large and very small scales. Report the largest residual per class. Then test
whether the same formula holds for (e) random non-Hermitian complex X, using the complex
bilinear Killing form of sl(2,C) on i X0, and say clearly whether Hermitian is needed.

### B2: mutation testing and counterexample hunt

Write a sympy or mpmath script that first confirms or refutes part 1, then tries to break
it. Mutations: change 1/8 to 1/4 and 1/16; flip the sign of the kappa term; change
(tr X)^2/4 to (tr X)^2/2; use X instead of X0 inside kappa; use the trace form tr(AB)
in place of the Killing form. For each mutation give a concrete numeric counterexample or
report that it survives. Then test the natural 3x3 analogue (Hermitian 3x3, su(3) Killing
form, same shape of formula) and report whether any choice of constants makes it true.
State what is special about 2x2.

### B3: independent toolchain, exact rationals, no sympy or numpy

Write a pure-Python script using only the standard library (`fractions.Fraction` for exact
Gaussian rationals; write your own 2x2 complex-rational matrix arithmetic). Do not import
sympy, numpy or mpmath. Build ad matrices from commutators, compute kappa as tr(ad ad), and
check parts 1 and 3 exactly on at least 5,000 random rational inputs, plus the specific
points X = 1, X = sz, X = 1 + sz, X = 2 - 3 sx + sy. Report any nonzero residual exactly.

## TEAM C: proof

### C1: derivation by hand, every step justified

Produce a written proof of part 1 (file `C1_proof.md`) from first principles: Cayley-Hamilton
for 2x2, det X = ((tr X)^2 - tr(X^2))/2, the traceless decomposition, and a derivation (not
a citation) of kappa(A, A) = 4 tr(A^2) on su(2) from tr(ad_A ad_A). Number the steps and name
the hypothesis each step uses. State precisely which hypotheses are needed: Hermitian?
2x2? field? Then check each numbered step with a short sympy script, one assertion per step,
and report which steps passed.

### C2: Lean 4 with Mathlib

A Lean 4 project with Mathlib already exists in WSL (Ubuntu) at
`/home/sam_leizerman/Liorhybrid-canonicalization/canonicalization` (look for the lakefile
and existing `.lean` files; existing theorems include names like `det_su2_killing`,
`det_I_smul`, `killingForm_real`). Run WSL commands from Git Bash by writing a script file to
`\\wsl.localhost\Ubuntu\home\sam_leizerman\C2_<name>.sh` and running
`MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -l /home/sam_leizerman/C2_<name>.sh`
(inline quoting through wsl is unreliable; backslashes get mangled).

State and prove part 1 in Lean for Hermitian `Matrix (Fin 2) (Fin 2) Complex`, reusing the
project's own definition of the Killing form if there is one, and say exactly which
definition you used. Put your work in ONE NEW file whose name starts with `C2_`; do not edit
any existing file, do not add it to any import list, do not commit. Check it with
`lake env lean <file>` from the project directory. The result must have no `sorry` and you
must run `#print axioms` on the final theorem and report its output. If the project's
Killing form is a shortcut definition (for example 4 tr(AB)) say so plainly, because then
the Lean theorem proves the identity relative to that definition only. If you cannot get it
to compile, report the exact error and how far you got; do not claim a proof you did not
compile. Copy the final `.lean` file into the threeway folder as `C2_ThreeWay.lean` too.

### C3: invariant-theory proof

Prove part 1 by symmetry instead of by expansion, and verify each ingredient by machine:
(i) both sides are real quadratic polynomials on Herm(2) invariant under X -> U X U^dagger
for U in SU(2); (ii) the space of such invariant quadratics is exactly 2-dimensional
(compute this dimension with a sympy linear-algebra calculation over the infinitesimal
action, do not just assert it); (iii) therefore two independent evaluations fix the
identity; do them at X = 1 and X = sz, with kappa from tr(ad ad). Write the argument in
`C3_proof.md`, and say what this proof shows that a direct expansion does not (for example,
why no other combination of the three readouts could appear).
