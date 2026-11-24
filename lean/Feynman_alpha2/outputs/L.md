# Agent L report: Lean 4 / Mathlib

## VERDICT (mathematical status of what was machine-checked)

- Thm lorentz: HOLDS, conditional on Prop symbol (the closed form of h_alpha is taken as the definition, not derived). Proved: `sin_mul_eq_mul_sin_iff` (a != 0: sin(a phi) = a sin phi for all phi iff a = 1 or -1; target 1, same). `lorentz_selects_two`: for alpha > 0, h_alpha constant on (0, pi/2) iff alpha = 2, with alpha = 1 (phi/sin phi) handled separately (`hfun_one_not_const`); this is stronger than target 1 (any constant, only the paper's interval). `hfun_two` (h_2 = 1), `hfun_symm` (h_{2-alpha} = h_alpha). The phi^2 expansion itself was NOT proved (only the coefficient identity).
- Thm KG (3) and Remark phase: HOLDS (algebraic core). `kg_symbol_zero`, `kg_dispersion`: at s = -i omega, 2 + tau^-2/(s^2 + c^2k^2) = 0 iff omega^2 = c^2k^2 + 1/(2 tau^2). `mu_tau`: mu^2 = 1/(2 tau^2) iff mu tau = 1/sqrt 2 (mu, tau > 0). `kg_dispersion_flipped`: sign -1 on memory gives omega^2 = c^2k^2 - 1/(2 tau^2). Parts (1), (2) NOT CHECKED beyond `kg_part2_algebra`.
- Prop checker: core lemmas HOLD. `neg_mu_sq_pow` ((-mu^2)^n = (i mu)^(2n), same as target 3); `checker_factor` ((i mu)^(2n+1)/(2 i c mu) = (-mu^2)^n/(2c)); `checker_step` (the stated n-th term maps to the (n+1)-th under (1/4c^2) int_0^u int_0^v); `checker_binomial_bound` (eps^n C(N-1,n) <= (N eps)^n/n!). Bessel sums and the lattice limit NOT CHECKED.
- Thm pathsum (1), (3): HOLDS WITH CONDITIONS (weaker). `rl_two_factorial`: int_0^T tau (T-tau)^(2n)/(2n)! = T^(2n+2)/(2n+2)!. `pathsum_majorant`: for a real sequence with |a_n| <= T^(2n)/(2n)! M, (1/2) sum (-mu^2)^n a_n converges absolutely with bound (1/2) cosh(mu T) M. `pathsum_symbol_geometric`: sum (-mu^2)^n X^(-n-1) = 1/(X+mu^2) for |mu^2/X| < 1. The convolution inequality for G*f and parts (2), (4) NOT CHECKED.
- Prop pair: HOLDS. `Mk_add` (group law, omega != 0), `Mk_det`.
- Thm unitary: core HOLDS. `norm_mode`, `closure_const` (J J^H constant for J = e^{-i omega t} J0), `closure_two_modes` (cross terms conj(z)^2 A B^H + z^2 B A^H, z = e^{i omega t}), `mixed_mode_not_const` (|e^{-i omega t} + e^{i omega t}| = |2 cos omega t|), `charge_ratio_bounds` (1 <= sqrt(1+x) <= 1 + x/2). Stone's theorem, L2 statement, charge conservation NOT CHECKED.
- Thm NR: mode-level HOLDS. `sqrt_expansion_bounds`, `nr_dispersion_bounds` (target 6, same), `nr_mode_bound` (phase error <= |t| p^4/(8 m^3 c^2 hbar), matching the paper's constant with p = hbar k). Plancherel step NOT CHECKED. Cor feynman: `feynman_amplitude_bounds` only.
- Prop nulllegs: symbol identities only. Prop symbol, Cor green, Prop descent, Prop descent3, Thm near2: NOT CHECKED.

## WHAT RAN

`lake env lean CalChecks\FeynmanAlphaTwo.lean` in `cal-checks\lean`: exit 0, no errors, no sorry; 5 deprecation warnings (`if_neg`/`if_pos`). All 27 `#print axioms` lines report exactly `[propext, Classical.choice, Quot.sound]`.

## PROBLEMS FOUND

None in the statements checked; every constant, sign and factor tested matched the paper.
Notes: `kg_dispersion` needs omega^2 != c^2k^2 (the paper states this). `Mk_add` needs omega != 0 (true since mu > 0).

## FILES

- `C:\Users\Sam Leizerman\cal-checks\lean\CalChecks\FeynmanAlphaTwo.lean`
- `C:\Users\Sam Leizerman\cal-checks\feynman\L_lean_output.txt`
- `C:\Users\Sam Leizerman\cal-checks\feynman\results\L.md`
