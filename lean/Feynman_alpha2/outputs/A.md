# Agent A report: high precision and special functions

Own code, mpmath at 40-90 digits (numpy only for grid tables, winding counts, time stepping). 230 PASS, 1 FAIL.

## VERDICT
- Prop symbol: HOLDS, for every alpha > 0 tested (0.05 to 7), real and complex s with Re s > 0, k = 0 included; quadrature vs closed form agree to 1e-35 or better. On Re s = 0 the integral itself converges only for alpha < 2 (checked to 1e-26); for alpha >= 2 the carrier value exists only as the limit.
- Thm lorentz: HOLDS. Defect zero only at alpha = 0, 2; positive below 2, negative above; leading coefficient alpha(2-alpha)/6 confirmed; next term alpha(2-alpha)(7-3(alpha-1)^2) phi^4/360.
- Thm KG, Table disp, Remark phase: HOLDS (mu tau_c = 1/sqrt2 is sharp).
- Thm near2: HOLDS WITH CONDITIONS (first order only; see problems 1-2).
- Thm pathsum, Prop checker, Prop descent3: HOLDS (terms by direct convolution quadrature, sums vs J0/J1/I0/I1 to 1e-40, Fourier symbols to 1e-46, corner counts exact, factor 1/(2ic mu) confirmed).
- Prop pair, Thm unitary, Thm NR, Cor feynman: HOLDS. NR error is exactly order 1/c^2: error*c^2 -> (t/8)||k^4 psi^|| = 3.67151, next term -5.0651/c^2.

## WHAT RAN
`python A_symbol.py` (96 PASS), `A_near2.py` (52 PASS, 1 FAIL, exit 1), `A_pathsum.py` (53 PASS), `A_nr.py` (29 PASS).
Tables: every printed digit reproduced in Table symbol, defect row, Table disp, Table near2 (roots, Im omega, formula column, mode counts), Table pathsum, Table checker, Table descent, Table NR and the 3.672 limit.
Near 2, delta = +-1e-2..+-1e-8, k = 0, 0.1, 1, 5: Im omega = (pi/4) delta mu^2/|omega| (1 + c1 delta), convergence linear in delta; c1 = -0.500, -0.503, -0.688, -0.955 (root's own mu, omega, as the table uses); with alpha = 2 values c1 = (ln2-1)/4, -0.074, +0.131, +1.208. Exact at k = 0: Im omega = rho cos(pi/alpha). Checkerboard: error O(eps), term n relative error -n(n+1)(1/u+1/v) eps.

## PROBLEMS FOUND
1. Thm near2, below 2: the stated rate is that of resonance poles only. Branch points at omega = +-ck give a tail ~ t^(-alpha) cos(ckt). At alpha = 1.9, k = 1 my time-stepped run follows the exponential to t ~ 150, then the power law (fitted exponent -1.902; coefficient matches my branch-point formula to 3%); at t = 600 |J| = 1e-6 versus 2e-15 from the exponential. "Every mode decays at rate ..." is not the long-time law.
2. Table near2 caption: growth rate at alpha = 2.01, k = 5 is 1.5587e-3 (rounds to 1.6e-3); the printed 1.5e-3 is the first-order value. True growing-mode count there is 2 (the printed 0 is the disclosed threshold).
3. Table descent, t = 8: the grid reaches only r <= 4, so 376 of the 751 points in the stated window; over the full window max|exact| is 2.32e-2, not 2.330e-3. The printed error 2.46e-6 comes from the single grid-edge point r = 4 (one-sided difference); elsewhere 8.4e-8. The appendix says central differences.
4. Prop symbol proof: the Gamma integral with beta = alpha - 1 needs alpha > 1; for 0 < alpha <= 1 the two pieces diverge separately (result still true by continuation).
5. Not reproducible from the text: time-stepped amplitudes (source, k unstated), mixed-solution spread 1.6e-1 (state unstated).
6. Outside the theorem's range: for 0 < alpha < 1 the same law has a real growing mode.

## FILES
`A_common.py`, `A_symbol.py`, `A_near2.py`, `A_pathsum.py`, `A_nr.py`, and their `.out` logs, in `C:\Users\Sam Leizerman\cal-checks\feynman\`.
