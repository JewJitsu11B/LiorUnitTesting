# Feynman alpha = 2 paper: independent checks. Briefs for four agents.

Durable copy so a cancelled run can be relaunched agent by agent. Keep until reported.

## COMMON (every agent)

A paper takes a causal memory law at "closure order alpha = 2" with a light-cone memory kernel
and claims: (i) Lorentz invariance of the kernel's symbol holds exactly at alpha = 2; (ii) the
balanced law is the Klein-Gordon equation; (iii) its Neumann series is a sum over piecewise-null
histories that resums to the retarded massive propagator and, in 1+1 dimensions, matches
Feynman's checkerboard term by term; (iv) each sign of frequency evolves unitarily, and the
nonrelativistic limit of the positive-frequency part is Feynman's free kernel with error
O(1/c^2). Your job is to test its mathematics independently.

Do not assume any claim is true. A failed claim, a missing hypothesis, a wrong constant, sign,
factor or normalization, or a proof step that does not follow is a valuable result: report it
plainly with the smallest example that shows it. Equally, do not manufacture problems: if a
claim holds, say it holds. Report mathematical status only (holds, fails, holds with conditions,
formal but not proved, not checked). Do not comment on whether the result "is physics".

Read-only sources, in `C:\Users\Sam Leizerman\cal-checks\feynman\reference\`:
- `paper_snapshot.tex`: the paper (732 lines). Definitions are in Sec. 2 (lines 107-156),
  axioms in Sec. 3, results in Secs. 4-8, numerical tables in Sec. 9 (from line 502),
  numerical methods in the appendix (from line 694). Read what your section names, plus the
  definitions.
- `cal_feynman_alpha2.py`: the author's numerical script. Agents E, A and L must NOT import
  or reuse it; write your own code so the check is independent. Agent N runs it.
- `addendum_snapshot.tex`: a short addendum; out of scope unless your section says otherwise.

Ground rules.
- Work only inside `C:\Users\Sam Leizerman\cal-checks\feynman\`. Prefix every file you create
  with your agent letter (for example `E_exact.py`). Do not edit or delete any existing file
  anywhere, do not touch the paper, no git commands that change history, no package installs,
  no work in WSL.
- Windows; run `python` from Git Bash or PowerShell. sympy, mpmath, numpy, scipy if present.
- No emojis anywhere.
- Label every check with the paper's own name for the claim (for example `Thm lorentz`,
  `Prop checker`, `Table 3 row 2`). Scripts must exit non-zero if any check fails, and print
  one `PASS|FAIL <label> <text>` line per check.
- Say only what you actually ran. Final message: a report under 450 words with
  VERDICT (per claim: HOLDS / FAILS / HOLDS WITH CONDITIONS / NOT CHECKED),
  WHAT RAN (commands, key output), PROBLEMS FOUND (each with evidence), FILES.
  Also write it to `results\<letter>.md`.

## E: exact and symbolic (sympy)

Check symbolically every statement that is exact mathematics. At least:
- Def profile / Def memory / Def symbol: internal consistency (normalizations, dimensions).
- Prop symbol: derive the symbol of the order-alpha light-cone memory yourself from the
  definitions and compare with (tau_c R)^(-alpha) sin((alpha-1) phi)/((alpha-1) sin phi),
  s = R cos phi, c|k| = R sin phi. State the conditions (range of alpha, of s) under which
  your derivation is valid.
- Thm lorentz: the symbol depends on s^2 + c^2 k^2 alone exactly when alpha = 2 (and alpha = 0);
  treat alpha = 1 separately; verify the leading defect alpha (2 - alpha) phi^2 / 6 and get
  the next term.
- Cor green: at alpha = 2 the kernel is the retarded Green's function of the wave operator
  (Kirchhoff shell in 3+1): check normalization and sign.
- Prop nulllegs: in 1+1 the order-2 kernel is the product of one order-one accumulation along
  each null leg.
- Prop descent and Prop descent3.
- Thm KG: from the balanced law T = 2J + K_2 J = 0 get (box + mu^2) J = 0 with
  mu tau_c = 1/sqrt 2; check the sign convention of box and the claim that an added factor -1
  on the memory makes the mass tachyonic.
- Thm near2: the rates (pi/4)(2 - alpha) mu^2 / omega just below and just above alpha = 2.
- Thm pathsum: the Neumann series terms, the absolute convergence bound, and the resummation
  to the retarded massive propagator.
- Prop checker: the explicit n-th term in 1+1; verify against the series of the retarded
  Klein-Gordon propagator, and verify the claimed equality with the (2n+1)-corner amplitude
  of Feynman's checkerboard divided by 2 i c mu (do the corner counting yourself, exactly,
  for small lattices, and the continuum limit symbolically).
- Prop pair, Thm unitary (the generator, the conserved quantity on one sign of frequency and
  its failure on mixed frequencies), Remark two norms.
- Thm NR and Cor feynman: the expansion in 1/c, the claimed error order, the composition law.
Report the exact form of anything that differs from the paper.

## A: high precision and special functions (mpmath, your own code)

Work at 30 digits or more.
- Prop symbol: evaluate the symbol by direct quadrature of the definition for several alpha
  in (0, 3], several (s, k), and compare with the closed form. Find the true range of validity.
- Thm lorentz: tabulate the Lorentz defect against alpha and phi; confirm it vanishes only at
  alpha = 2 (and 0) and that the leading coefficient is alpha (2 - alpha)/6.
- Thm near2: solve the dispersion relation of the balanced order-alpha law exactly for
  alpha = 2 -+ delta, delta from 1e-2 down to 1e-8, and extract the decay or growth rate;
  compare with (pi/4)(2 - alpha) mu^2/omega; report the next correction and how fast the
  formula converges.
- Thm pathsum, Prop checker, Prop descent3: sum the Neumann series numerically in 1+1 and in
  3+1 and compare with the retarded massive propagators (Bessel J0 inside the cone in 1+1;
  the delta shell plus the J1 tail in 3+1). Check term-by-term agreement with your own
  checkerboard sum on lattices of increasing size, including the factor 1/(2 i c mu).
- Thm NR: measure the error of the positive-frequency propagator against Feynman's free
  kernel as c grows; confirm or refute O(1/c^2) and give the coefficient.
- Reproduce every numerical table of Sec. 9 from your own code and say how many printed
  digits agree, row by row.
- Look for anything the paper does not mention: extra roots of the dispersion relation,
  convergence failures, ranges of parameters where a stated formula stops holding.

## N: direct simulation, then the author's script

First write your own code, without opening `cal_feynman_alpha2.py`:
- Discretize the balanced alpha = 2 law in 1+1 (light-cone kernel as defined in the paper)
  and evolve a packet; compare with Klein-Gordon evolution computed spectrally; measure the
  mass from the dispersion and compare with mu tau_c = 1/sqrt 2.
- Unitarity: evolve positive-frequency, negative-frequency and mixed data; measure the
  conserved quantities the paper names (Thm unitary, Remark two norms) and the drift of the
  trace readout in each case.
- Near alpha = 2: iterate the order-alpha law for alpha slightly below and above 2 and
  measure decay or growth; compare with Thm near2.
- Nonrelativistic limit: compare the positive-frequency evolution with free Schrodinger
  evolution as c grows (Thm NR).
Then run the author's script unchanged (from a copy in your own working folder if it writes
files) and compare its printed numbers with the paper's tables and with yours. Report every
number that differs beyond the printed digits, any claim in the paper that has no code behind
it, and any place where the paper's description of a method does not match what the script
does (rescaling, truncation, fitting windows, seeds, parameter values not stated).

## L: Lean 4 with Mathlib

Lean project: `C:\Users\Sam Leizerman\cal-checks\lean` (Lean 4.35.0, Mathlib v4.35.0, already
built). The one exception to "work only inside the feynman folder": put everything in ONE new
file `CalChecks\FeynmanAlphaTwo.lean` (namespace `CalChecks.AlphaTwo`), `import Mathlib` plus
any existing `CalChecks` module you reuse. Check with
`lake env lean CalChecks\FeynmanAlphaTwo.lean` from the `lean` folder (it can take minutes; use
a long timeout). Do not edit existing files, do not run `lake update` or anything that
downloads. No `sorry`, no new axioms; end with `#print axioms` for each main theorem.
Do not claim anything you did not compile.

Read the paper's statements and choose what can honestly be proved. Suggested targets, in
priority order; do as many as compile and report the rest as not done:
1. Thm lorentz, algebraic core: for real a != 0, (for all phi, sin (a phi) = a sin phi) iff
   a = 1 or a = -1. With a = alpha - 1 this is alpha = 2 or alpha = 0. Also the alpha = 2
   value: sin phi / sin phi = 1 where sin phi != 0.
2. Thm KG, algebraic core: from 2 + (tau_c R)^(-2) = 0 the relation the paper states between
   mu and tau_c (mu tau_c = 1/sqrt 2), with the sign conventions written out; and that the
   opposite sign on the memory gives the opposite sign of mu^2.
3. Prop checker: (-mu^2)^n = (i mu)^(2n) in the complex numbers.
4. Thm pathsum: an absolute convergence statement for the series the paper bounds (state
   precisely which series and which bound you proved).
5. Thm unitary: for real omega, the multiplier exp(-i omega t) has norm 1, so the L2 norm of
   each Fourier mode is conserved; for a 2x2 carrier J(t) = exp(-i omega t) J0,
   J(t) * J(t)ᴴ = J0 * J0ᴴ; for a sum of two modes with opposite signs of frequency the
   closure is not constant (give the cross term).
6. Thm NR: for m > 0, c > 0 and real p,
   0 <= p^2/(2m) - (sqrt(m^2 c^4 + p^2 c^2) - m c^2) <= p^4/(8 m^3 c^2),
   which is the O(1/c^2) error of the dispersion relation.
7. Anything else in the paper that is a clean finite statement.
List every theorem proved with a one-line plain statement, exactly how each relates to the
paper's statement (same, weaker, a core lemma), and which targets were not done.
