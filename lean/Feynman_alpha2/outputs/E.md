# Agent E: exact and symbolic checks (sympy 1.14, mpmath 1.3)

## VERDICT
- Def profile / memory / symbol: HOLDS (masses 1, all three transforms sinc, projections, bound, dimensions).
- Prop symbol: HOLDS for all alpha > 0, Re s > 0. The printed proof is valid only for alpha > 1 (see problem 1).
- Thm lorentz: HOLDS. Leading defect alpha(2-alpha)/6 confirmed. Next term: alpha(2-alpha)(4+6alpha-3alpha^2) phi^4/360. alpha=1: h = 1 + phi^2/6 + 7phi^4/360.
- Cor green: HOLDS (sign +, normalization 1, Kirchhoff form).
- Prop nulllegs, Prop descent, Prop descent3: HOLD.
- Thm KG: HOLDS (mu tau_c = 1/sqrt 2; box symbol s^2+c^2k^2; factor -1 gives tachyon).
- Thm near2: HOLDS WITH CONDITIONS (rate correct to first order; see problem 2).
- Thm pathsum: HOLDS (terms, bound attained by constant S, cosh, resummation).
- Prop checker: HOLDS, with a normalization reading (see problem 3).
- Prop pair, Thm unitary (1),(2),(4): HOLD. Thm unitary (3): HOLDS WITH CONDITIONS (problem 4).
- Thm NR: HOLDS; error exactly order 1/c^2, constant 1/8 sharp. Cor feynman: HOLDS (see problem 5).
- Remark two norms (frame dependence): NOT CHECKED.

## WHAT RAN
`python E_exact.py` -> exit 0, 95 PASS, 0 FAIL, about 40 s. Output in `E_output.txt`.
Key output: symbol closed form = termwise Gamma series (symbolic alpha, to b^12) and = 30-digit quadrature
(alpha 0.3 to 3, real and complex s, max diff below 1e-25); near2 exact roots at b=0.75, mu=1:
Im omega/prediction = 1.00045 (delta=1e-2), 1.0000051 (1e-4), 0.99943 (-1e-2); second-order
coefficients match to O(delta^3); brute-force corner count over all 2^N paths, N=2..16, 1630 classes.

## PROBLEMS FOUND
1. Prop symbol, proof gap. It uses int tau^(beta-1) e^(-z tau) = Gamma(beta) z^(-beta) with beta = alpha-1, which diverges for 0 < alpha <= 1. The formula is still right there (series and quadrature at alpha = 0.3, 0.5); the proof needs analytic continuation.
2. Thm near2. The proof says the symbol of T changes by -i pi delta only. Its real part also changes at first order, so Re omega shifts at O(delta) (d omega/d delta = 0.4007 + 0.6283i at b=0.75, mu=1). The stated Im rate is unaffected. Relative correction to the rate is +0.0509 delta there. "mu^2 = omega^2 - c^2k^2 at the root" is complex for alpha != 2.
3. Prop checker. Phi as printed is a leg-duration volume, not a density in x; 2ic mu = (Jacobian 2c) x (one corner i mu). L_eps counts one fixed (first, last) orientation: C(N_R-1,n)C(N_L-1,n). All corner-odd paths number twice that.
4. Thm unitary (3) is false as a universal statement. J^ = E11 e^{-i omega t} + E22 e^{+i omega t} has both signs and constant Q(t) = identity. Condition needed: int tr(a_+^dagger a_-) e^{2i omega t} dk not identically zero.
5. Cor feynman. The rescaled [G_mu]^+ has multiplier (mc^2/hbar omega) e^{-i(omega - mc^2/hbar)t}; it is not a group at finite c. The group law holds for the operator of Thm unitary (1), as cited.

No wrong constant, sign or factor was found in any displayed formula.

## FILES
- `E_exact.py` (script), `E_output.txt` (full output), `results\E.md` (this report).
