# Agent E: exact algebra

## VERDICT
- Def memory: HOLDS.
- Lemma W, proof line 1: HOLDS (coefficientwise to x^60, symbolic alpha).
- Lemma modes (1), (3), (4): HOLDS.
- Lemma flat: HOLDS. Actual next coefficient: phi = sqrt(eps)(1 + eps/24 + 3 eps^2/640 + ...), s = sqrt(d)(1 - d/24 + 3 d^2/640 - ...), d = alpha-1. At alpha=1 the double root belongs to (z-1)F; F itself is z-1.
- Thm rotation (formal expansion): HOLDS. Displayed F expansion, phi_0, phi_1, phi_2, F'(phi_0)=2i all reproduced. The non-log eps^(3/2) terms of F are i(eps g phi/12 - phi^3/6), imaginary, so Im phi at eps^(3/2) is exactly (pi/4) sqrt(g). Derived next terms:
  Re phi = sqrt(eps g)[1 + eps(ln sqrt(eps g)/2 + 1/2 - g/3 - 1/(8g))] + (pi/4) g eps^2;
  Im phi = eps(1-g)/2 + (pi/4) eps^(3/2) sqrt(g) + eps^2[-(g/2) ln sqrt(eps g) + (1-g)(2-g)/4].
- Cor dispersion: HOLDS (k^4 term for Gaussian: -<r^2>^2 k^4/288).
- Thm unitary: first exponent and the ratio expression HOLD exactly; second exponent HOLDS WITH CONDITIONS (see problem 2).
- Thm feedback (1), flat roots, (4) F(1): HOLDS. (3): recurrence HOLDS; Green's function normalization FAILS (sign).
- Thm normalizable: HOLDS; no counterexample.
- Prop accumulator: HOLDS.
- Prop carrier (1)-(4), Prop lorentz (1), (3), Lemma minors, Prop composition (1)-(3): HOLD (conditions in problem 4).
- Prop born (1): HOLDS at leading order in k^2 and eps.
- NOT CHECKED: Lemma weyl, Jonquiere formula itself, Prop lorentz (2), (4), Prop orientation.

## WHAT RAN
- `python E_exact.py` -> `E_exact.log`: 61 checks, 60 PASS, 1 FAIL (the Green's function jump).
- `python E_rotation.py`: all 9 checks PASS. Its mpmath sanity block (numeric, 40 digits) finds exact roots matching my series with error about eps^(5/2) ln^2, and reproduces Table modes column 6: 0.901459, 0.840053, 0.808426, 0.7945, 0.788852, 0.786672. The eps^2 log term explains 0.78667 vs pi/4.
- Thm normalizable: exact Schur-Cohn test on 4075 rational kernels, none with a root at |z|>=1. Total weight 1/(1-alpha) (Dirac, alpha=1/2, mass 2) puts roots on the circle; signed weights (3,-2) fail.
- `python E_side_numeric.py` (numpy, numeric side note).

## PROBLEMS FOUND
1. Thm feedback (3): G(n0+1) - (1+alpha)G(n0) + G(n0-1) = -1 for G = e^{-s|n-n0|}/(2 sinh s). The Green's function of the displayed operator is the negative. Also, in the update rule itself (J_{n+1} = T_n + S_n), e^{-s|n|} needs the source -2 sinh s at every n>=0 (exact: alpha=3/2, J_n=2^{-|n|}, residual -3/2 for all n>=0): a step, not an impulse. The bounded impulse response is -(G_n - G_{n-1}): still advanced, but with different amplitude and opposite sign on the two sides (-1/3 at n0, +1/3 at n0+1), so not globally proportional to e^{-s|n-n0|}.
2. Thm unitary, second display: the exact O(k^2) coefficient is -(i + sqrt(eps) - (pi/4)eps); the paper drops -(pi/4)eps, though its own ratio formula keeps it. Same term in Prop born (1).
3. Table normalizable, "exponential, width 100": the infinite exponential kernel gives exactly |z| = 0.990075 (closed-form quadratic), not 0.99205. 0.99205 is the 400-lag truncated kernel (numeric). Width 10 and Dirac rows match (0.97677, 0.97979). Theorem unaffected.
4. Minor: Prop carrier (2) needs J != 0; Prop composition (3) gives Tr_B Q = |eta|^2 I/2, equal to I/2 only for unit eta.

## FILES
`E_exact.py`, `E_exact.log`, `E_rotation.py`, `E_side_numeric.py`, `results\E.md` (all in `C:\Users\Sam Leizerman\cal-checks\schrodinger\`).
