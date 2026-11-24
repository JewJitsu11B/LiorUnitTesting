# Schrodinger-edge paper: independent checks. Briefs for four agents.

Durable copy so a cancelled run can be relaunched agent by agent. Keep until reported.

## COMMON (every agent)

A paper claims that a discrete feedback law, iterated, gives free Schrodinger evolution in a
limit. Your job is to test its mathematics independently. Do not assume any claim is true.
A failed claim, a missing hypothesis, a wrong constant or sign, or a proof step that does not
follow is a valuable result: report it plainly, with the smallest example that shows it.
Equally, do not manufacture problems: if a claim holds, say it holds.

Read-only sources, in `C:\Users\Sam Leizerman\cal-checks\schrodinger\reference\`:
- `paper_snapshot.tex`: the paper (749 lines). Read the parts your section names.
- `cal_edge_transducer.py`: the author's own numerical script (NumPy). Agents E, A and L must
  NOT import or reuse it; write your own code so the check is independent. Agent N runs it.

The objects (paper sections 2 to 4), in plain text:
- alpha in (0,1), eps = 1 - alpha. Memory weights w_j = (j^alpha - (j-1)^alpha)/Gamma(alpha+1),
  j >= 1. Generating function W(z) = sum_{j>=1} w_j z^(-j).
- Spatial factor ghat in (0,1], ghat(k) = 1 - <r^2> k^2/6 + O(k^4); Gaussian case
  ghat = exp(-<r^2> k^2/6).
- Update rule (eq. update): J_{n+1} = alpha J_n - (1-alpha) ghat sum_{j>=1} w_j J_{n-j}.
- Characteristic function F(z) = z - alpha + (1-alpha) ghat W(z). A mode z0^n solves the update
  rule when F(z0) = 0. Write z = e^{i phi}: Re phi is rotation per update, Im phi decay per update.

Ground rules.
- Work only inside `C:\Users\Sam Leizerman\cal-checks\schrodinger\`. Prefix every file you
  create with your agent letter (for example `E_exact.py`). Do not edit or delete any existing
  file anywhere, do not touch the paper, no git commands that change history, no package
  installs, no work in WSL (agent L: see its section).
- Windows; run `python` from Git Bash or PowerShell. sympy, mpmath, numpy are installed.
- No emojis anywhere.
- Label every check with the paper's own name for the claim (for example `Lemma flat (1)`,
  `Thm rotation`, `Table modes row 3`).
- Say only what you actually ran. Final message: a report under 450 words with
  VERDICT (per claim: HOLDS / FAILS / HOLDS WITH CONDITIONS / NOT CHECKED),
  WHAT RAN (commands, key output), PROBLEMS FOUND (each with evidence), FILES.
  Also write it to `results\<letter>.md`.

## E: exact algebra (sympy, exact arithmetic only)

Check every claim of the paper that is exact algebra, symbolically. At least these:
- Def memory: w_j is the integral of tau^(alpha-1)/Gamma(alpha) over [j-1, j].
- Lemma W, first line of the proof: sum (j^alpha - (j-1)^alpha) x^j = (1-x) sum j^alpha x^j
  (as formal power series; check coefficientwise to high order).
- Lemma modes (1), (3), (4): substitute and verify the algebra (for (3), the z-transform of
  the zero-past solution; for (4), the split of the memory sum at j = n).
- Lemma flat: with w_j = 1, W = 1/(z-1), (z-1)F(z) = z^2 - (1+alpha) z + alpha + (1-alpha) ghat;
  the three root cases at ghat = 1; the claim phi (or s) = sqrt|1-alpha| (1 + O(|1-alpha|)):
  get the actual next coefficient.
- Thm rotation, the formal expansion: starting from the k = 0 term of eq. jonquiere, derive
  the displayed expansion of F in phi and eps yourself, then solve order by order and compare
  with the paper's phi_0, phi_1, phi_2 and with the displayed theorem. Check the claim "the
  other terms of order eps^(3/2) are imaginary in F, so they shift only the real part of phi".
  Check the paper's statement that F'(phi_0) = 2i.
- Cor dispersion: 1 - sqrt(ghat) = <r^2> k^2/12 + O(k^4), and the Gaussian closed form.
- Thm unitary: derive both displayed exponents from Thm rotation, and the "ratio to the
  kinetic phase" expression (1/2) sqrt(eps) (1 + sqrt(ghat)) - (pi/4) eps.
- Thm feedback: (1); the flat-weight roots for alpha > 1; (3) the recurrence
  J_{n+1} - (1+alpha) J_n + J_{n-1} = 0 and its two-sided bounded Green's function
  e^{-s|n-n0|}/(2 sinh s) (verify the jump condition and the normalization);
  (4) F(1) with brake b and fading lambda.
- Thm normalizable: check each inequality in the proof chain and the equality case
  F(1) = 2(1-alpha). Try to find a counterexample with exact rational weights.
- Prop accumulator: the Laplace representation of tau^(alpha-1)/Gamma(alpha), the mass
  fraction (delta/S)^(1-alpha).
- Prop carrier (1)-(4), Prop lorentz (1), (3), Lemma minors, Prop composition (1)-(3),
  including the examples.
- Prop born (1): the decay rate 2[(pi/4)(1-alpha) + sqrt(1-alpha)(hbar/2m)<k^2>] from Thm unitary.
Report the exact form of anything that differs from the paper.

## A: special functions and asymptotics (mpmath, high precision, your own code)

Work at 40 digits or more.
- Lemma weyl: the improper integral of s^(alpha-1) e^{i omega s} over (0, inf) equals
  Gamma(alpha) (-i omega)^(-alpha) for 0 < alpha < 1; check several alpha and both signs of
  omega; check the symbol and phase stated in Def weyl; show what happens at alpha >= 1.
- Lemma W (eq. jonquiere): W(e^{i phi}) = (1 - e^{-i phi}) sum_{k in Z} (i(phi + 2 pi k))^(-alpha-1)
  with principal powers. Test (a) against direct summation for |z| > 1 (the formula continued
  to z = r e^{i phi}), (b) on the unit circle against an independently accelerated sum,
  (c) against mpmath.polylog. State which branch conventions make it true and whether the
  paper's are those.
- Thm rotation: find the co-rotating root of the EXACT F (through the continuation above)
  for ghat in {1, 0.9, 0.5} and eps = 1e-2 down to 1e-12. Compare with
  phi = sqrt(eps ghat)(1 + O(eps ln eps)) + i[(1/2) eps (1-ghat) + (pi/4) eps^(3/2) sqrt(ghat)]
  + O(eps^2 ln eps). Extract the coefficients numerically (differences across decades,
  Richardson where it helps) and say whether pi/4 and 1/2 are right and whether the stated
  error orders are right. Reproduce Table modes (all six rows) and say how many digits agree.
  Check the paper's remark that the measured value 0.78667 at eps = 1e-7 tends to pi/4.
- Cor dispersion and Table mass: reproduce the five rows from exact roots.
- Thm unitary and Table diffusion: reproduce the table; check the predicted value
  (1/2) sqrt(eps)(1 + sqrt(ghat)) - (pi/4) eps against it.
- Thm feedback and Table feedback: for alpha = 1.01 and 1.001 find the real root z* > 1 and the
  growth per update with and without brake b and fading lambda; check the criterion's
  thresholds b > 0.095 (lambda = 0.1) and b > 0.501 (lambda = 0.01); check that the criterion
  is sufficient as well as necessary on a grid.
- Thm normalizable and Table normalizable: reproduce largest |z| for the four kernels.
- Is there any OTHER root of F on or outside the unit circle for 0 < alpha < 1 (a mode the
  paper does not mention)? Use the argument principle numerically on a contour.

## N: dynamics by direct iteration (numpy, independent implementation, then the author's script)

First write your own code, without reading `cal_edge_transducer.py`:
- Iterate the update rule directly and measure rotation and decay per update at
  alpha = 0.999, k = 0 (paper: 24000 updates give 0.031570 and 2.652e-5).
- Lemma modes (4): implement the exact one-way infinite past and check that a pure mode
  z0^n continues as z0^n.
- Prop orientation (1)-(4): flat weights exact continuation; power-law counter-rotating
  fraction at eps = 1e-4; zero past excites both; two-cycle truncated history sends 70 to 97
  percent of the norm counter-rotating.
- Table packet: packet exp(-x^2/2 + i x) on a periodic box of length 40 with 128 points,
  evolved to Schrodinger time (hbar/2m) t = 0.5 with hbar/2m = <r^2>/12, compared with free
  Schrodinger evolution after aligning the global phase. Reproduce as many of the five rows
  as run time allows (state which), including the floor and counter-rotating columns, and the
  quoted norm loss of 0.9 to 1.5 percent at eps = 1e-4.
- Table closure (Prop born): Tr Q drift per radian and readout drift for one-way and zero past.
- Table readout: the nine rows.
Then run the author's script unchanged (`python reference\cal_edge_transducer.py`, from a
copy in your own working folder if it writes files) and compare its printed numbers with the
paper's tables and with yours. Report every number that differs beyond the printed digits,
and any place where the paper's description of the method does not match what the script does.

## L: Lean 4 with Mathlib

Lean project: `C:\Users\Sam Leizerman\cal-checks\lean` (Lean 4.35.0, Mathlib v4.35.0, already
built). Put everything in ONE new file `CalChecks\SchrodingerEdge.lean`
(namespace `CalChecks.Edge`), `import Mathlib` plus any existing `CalChecks` module you reuse.
Check with `lake env lean CalChecks\SchrodingerEdge.lean` from the `lean` folder (it can take
minutes; use a long timeout). Do not edit existing files, do not run `lake update` or anything
that downloads. No `sorry`, no new axioms; end with `#print axioms` for each main theorem.
Do not claim anything you did not compile.

Targets, in priority order; do as many as compile and report the rest as not done:
1. Thm normalizable, as a real theorem: for w : N -> R with w j >= 0, Summable w, tsum = 1,
   alpha in (0,1), ghat in [0,1], and complex z with 1 <= |z|,
   z - alpha + (1-alpha) * ghat * tsum (w j * z^(-(j+1))) is nonzero. State exactly the
   indexing and hypotheses you used, and whether the paper's proof goes through as written.
2. Lemma flat: (z-1) * (z - alpha + (1-alpha) * ghat / (z-1)) = z^2 - (1+alpha) z + alpha
   + (1-alpha) ghat for z != 1; for alpha in (0,1), z = exp(i phi) with cos phi = (1+alpha)/2
   is a root at ghat = 1 and has norm 1; alpha = 1 gives the double root 1; for alpha > 1 the
   roots are exp(s), exp(-s) with cosh s = (1+alpha)/2.
3. Thm feedback (1): at alpha = 1 the update rule gives J_{n+1} = J_n. Thm feedback (3):
   G n = exp(-s |n - n0|) satisfies G (n+1) - (1+alpha) G n + G (n-1) = 0 for n != n0 and the
   stated jump at n0.
4. Prop accumulator, discrete part: for each fixed j >= 1,
   (j^alpha - (j-1)^alpha)/Gamma(alpha+1) tends to 1 as alpha -> 1 from below.
5. Prop carrier and Prop lorentz, for J a 2x2 complex matrix: det (J * Jᴴ) = |det J|^2;
   with Q = J * Jᴴ = x0 + x.sigma, (2 |det J| / tr Q)^2 = 1 - |x|^2/x0^2; C = sech eta for
   Q = rho * exp(eta n.sigma) stated in coordinates (x0 = rho cosh eta, |x| = rho sinh eta).
   Existing `CalChecks` files already have related theorems (purity, boosts): reuse them and
   name which.
6. Lemma minors: 2 • J = J * Jᴴ implies Jᴴ = J and J * J = 2 • J; the diag(2,2,0) example in
   3x3 has det 0 and trace of J/2 equal to 2.
List every theorem proved with a one-line plain statement, and which targets were not done.
