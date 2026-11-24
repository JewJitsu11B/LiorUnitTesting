# Agent N: direct simulation, then the author's script

## VERDICT
- Thm KG (mu tau_c = 1/sqrt 2): HOLDS. Real-space null-lattice run of 2J + K_2 J = S against the spectral sourced KG solution: relative L2 error 7.1e-3, 1.8e-3, 4.4e-4 at h = 0.08, 0.04, 0.02 (ratio 4.00). Measured mu tau_c extrapolates to 0.7071068 (also at c = 2, tau_c = 0.5).
- Thm unitary, Remark two norms: HOLDS. Spread of ||J||^2: 6.9e-13 on either sign, 6.3e-4 mixed (per-mode oscillation at 2 omega). KG charge constant to 2e-14 in all three; q(+) > 0 > q(-); the two-norms inequality holds; Q and p_m constant to 2e-9.
- Thm near2: HOLDS WITH CONDITIONS. Time-stepped rates match the exact roots (2.05: 2.7902e-2; 2.10: 5.5902e-2; 1.95: -2.760e-2 vs -2.753e-2). First-order formula off by relative -0.69 delta at k = 1. Condition: Problem 1.
- Thm NR, Cor feynman: HOLDS. Error x c^2 tends to 3.6712 (bound 3.6715); bounds hold for c = 2..128.
- Sec. 9 tables: the author's script reproduces every printed number except norm-drift roundoff.

## WHAT RAN
- `python N_sim.py` (68 s, exit 0, 29 PASS).
- `python N_tables.py`, my rebuild from Appendix B written before opening the script (exit 1: Table 5 and Table 6 t=8 differ, Problems 3-4; all other rows match).
- Author's script unchanged, from `N_author_run\` (16 s, exit 0).
- `python N_compare.py` (exit 0).

## PROBLEMS FOUND
1. Below alpha = 2 the sourced solution stops decaying exponentially: it crosses over to a power law t^-alpha oscillating at c|k|, not omega. At alpha = 1.9, k = 1: exponent 1.91, frequency 1.0000; |J| = 1.6e-6 at t = 600 where the pole alone gives 5e-15. Same in the author's stepper; not in the paper.
2. Table 3 caption: the |J| values are single-instant samples of an oscillating signal, not envelopes (sample rates 2.68e-2, 3.02e-2 against roots 2.75e-2, 2.79e-2). Source centre 10 and width 1 unstated.
3. Table 6, t = 8: the 2.46e-6 is one grid-edge point where `np.gradient` is one-sided (paper says central differences); interior maximum 8.44e-8. The grid stops at r = 4 of 7.75.
4. Table 5: reproducible only with the first step forced right (start at site 1, N - 1 updates); starting at the origin gives 3.891e-2, not 3.697e-2. The script's recursion equals the binomial formula of Prop checker(3) exactly.
5. Caption "1.5e-3" at alpha = 2.01, k = 5: the root gives 1.559e-3; no code prints it.
6. No code behind: the law in spacetime (Thm KG(1)-(2)); term-by-term checkerboard equality; I_0, I_1 sums; convergence bound; charge inequality; Cor feynman's bound; tachyon column (formula only). Section 5c evaluates closed-form mode phases with an unstated mix.
7. Unstated: winding rectangle |Re| <= 12; numpy >= 2 needed.

## FILES
In `C:\Users\Sam Leizerman\cal-checks\feynman\`: `N_sim.py`, `N_tables.py`, `N_compare.py`, `N_sim_out.txt`, `N_tables_out.txt`, `N_compare_out.txt`, `N_author_out.txt`, `N_nrlattice_out.txt`, `N_author_run\`, `results\N.md`.
