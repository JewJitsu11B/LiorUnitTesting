# Closure at alpha = 2 (the Feynman path sum): checks

Paper: "An Axiomatic Derivation of the Feynman Path Sum at the Light-Cone Kernel Width and
Closure Order alpha = 2 of the Causal Accumulation Law" (`paper/cal_feynman_alpha2.tex/.pdf`,
17 pages) and its five-page canonical sheet (`paper/cal_feynman_canonicalization.tex/.pdf`).
Both are the corrected versions.

The checks were written by four independent runs from `BRIEFS.md`: E (exact and symbolic),
A (high precision), N (direct simulation), L (Lean). None of them reuses the paper's own script,
except `N_compare.py`, which loads it unchanged to compare numbers.

## Layout

| Folder or file | What it holds |
|---|---|
| `paper/` | Corrected paper and sheet, LaTeX and PDF |
| `lean/` | Lean 4 project: `FeynmanAlphaTwo.lean` |
| `python/` | The check scripts (run them from inside this folder) |
| `outputs/` | Saved output of every run and the four reports `E.md`, `A.md`, `N.md`, `L.md` |
| `reference/` | The paper and addendum as they stood before correction, and the paper's own script |
| `BRIEFS.md` | What each of the four runs was asked to check |
| `apply_corrections.py` | Turns `reference/paper_snapshot.tex` into `paper/cal_feynman_alpha2.tex`; every replacement must match exactly once |

## Lean (`lean/CalChecks/`)

| File | Theorems | What it proves |
|---|---|---|
| `FeynmanAlphaTwo.lean` | 38 | The light-cone profile `h_alpha` is constant on (0, pi/2) exactly at alpha = 2; the Klein-Gordon dispersion relation with `mu tau_c = 1/sqrt 2` and its sign-flipped form; `(-mu^2)^n = (i mu)^(2n)` and the step from one term of the path sum to the next; an absolute-convergence majorant for the path sum; constancy of the closure on one sign of frequency and the cross terms on two; the two-sided bound on the nonrelativistic dispersion error |

Lean 4.35.0, Mathlib v4.35.0, no `sorry`, standard axioms only; the file ends with its own
`#print axioms` lines. Build with `lake exe cache get` then `lake build CalChecks` from `lean/`.
Saved compiler output: `outputs/L_lean_output.txt`.

## Python (`python/`)

Prefix tells the method: `E_` exact (sympy), `A_` mpmath at 40 digits or more, `N_` direct
simulation (numpy), `G_` checks written for the sheet, `T_` a test at one chosen alpha.
Each prints `PASS` or `FAIL` per check and exits non-zero on failure.

| Script | Method | What it tests |
|---|---|---|
| `E_exact.py` | sympy | 95 checks labelled by the paper's statement |
| `G_check.py` | sympy, exact | G1 a boost on null coordinates leaves uv and du dv invariant; G2 the pair matrix has determinant 1 and trace 2 cos(omega t), elliptic in SL(2,R); G3 the next coefficient of the Lorentz defect and the phi^2, phi^4 coefficients of `h_alpha`; G4 rotating (s, c\|k\|) fixes R and shifts phi, so `h_2 = 1` is the invariant case |
| `A_common.py` | - | Shared helpers for the `A_` scripts. Not a test |
| `A_symbol.py` | mpmath | The symbol of the light-cone memory, the Lorentz theorem, the profile, and the symbol, defect and dispersion tables |
| `A_near2.py` | mpmath | Exact roots near alpha = 2 against the near-2 theorem and its table |
| `A_pathsum.py` | mpmath (numpy, scipy for the grid tables) | The path-sum terms, the checkerboard, the descent to 3+1, Bessel resummations and their tables |
| `A_nr.py` | mpmath | Klein-Gordon from the law, the pair matrix, unitarity, the nonrelativistic limit, the Feynman corollary and the Section 9 numbers |
| `N_sim.py` | numpy | Simulation on a null lattice with independent code, 29 checks. Sections: `lattice`, `near2`, `nr`, `nrlattice` (`python N_sim.py lattice`; no argument runs all) |
| `N_tables.py` | numpy, scipy | The Section 9 tables rebuilt from the paper's description of its method, before reading the paper's script |
| `N_compare.py` | numpy | Runs the paper's script unchanged and pins down where its method differs from the description. Loads `N_author_run/cal_feynman_alpha2.py` |
| `N_author_run/cal_feynman_alpha2.py` | - | Unchanged copy of the paper's script, used by `N_compare.py` |
| `T_alpha_test.py` | mpmath | The law at any one alpha near 2: roots, decay rates, and the crossover from exponential decay to the algebraic tail (`python T_alpha_test.py 1.9 long`) |

The four `A_` scripts together make 231 checks.

Failures by design:

- `A_near2.py`, one: it tests a figure in the caption of the near-2 table as the paper first
  gave it. The paper now carries the corrected figure.
- `N_tables.py`, five: its independent rebuild starts the checkerboard walk with a free first
  step (four rows) and differences the 3+1 tail in the interior of the grid (one row), where the
  paper's script fixes the first step and includes the grid edge. Both differences are now
  stated in the paper's captions; `N_compare.py` confirms the grid-edge cause.

## Outputs (`outputs/`)

| File | From |
|---|---|
| `E.md`, `A.md`, `N.md`, `L.md` | The four reports: verdict per statement, what ran, problems found |
| `E_output.txt`, `E_myrun.txt` | Runs of `E_exact.py` |
| `A_symbol.out`, `A_near2.out`, `A_pathsum.out`, `A_nr.out`, and the `_myrun.txt` of each | Runs of the `A_` scripts |
| `N_sim_out.txt`, `N_sim_myrun.txt`, `N_nrlattice_out.txt` | Runs of `N_sim.py` |
| `N_tables_out.txt`, `N_compare_out.txt` | Runs of `N_tables.py` and `N_compare.py` |
| `N_author_out.txt` | Output of the paper's own script, unchanged |
| `L_lean_output.txt`, `L_myrun.txt` | Lean compiler output for `FeynmanAlphaTwo.lean` |

## What the checks changed in the paper

No displayed formula was wrong. The proof of the symbol formula now covers 0 < alpha <= 1 by
continuation; the first-order rate near 2 is stated for the imaginary part only, with its
measured next-order coefficient; "not constant" on mixed frequencies became "in general not
constant", with the exact condition and a counterexample; the rescaled nonrelativistic kernel is
said not to be a group at finite c; the checkerboard normalization is spelled out; three table
captions now say what the grid and the counting threshold actually do.

New in the paper, found by the checks: below alpha = 2 the exponential decay belongs to the
poles only. The symbol also has branch points at omega = +-c|k|, and their contribution falls
as t^(-alpha) at the massless frequency and dominates at late times. This is measured, with the
origin identified; it is not derived.

Not proved: the symbol formula itself; the retarded Green function and the descent to 3+1; the
Bessel resummations; the lattice limit of the checkerboard; the near-2 theorem; the step from
one Fourier mode to L2. All of these are checked symbolically or numerically only.
