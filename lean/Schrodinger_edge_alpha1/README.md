# Schrodinger edge (alpha -> 1 from below): checks

Paper: "Schrodinger evolution from the bare Causal Accumulation Law at alpha -> 1^-"
(`paper/cal_schrodinger_collapse.tex/.pdf`) and its five-page canonical sheet
(`paper/cal_schrodinger_canonicalization.tex/.pdf`). Both are the corrected versions.

The checks were written by four independent runs from `BRIEFS.md`: E (exact algebra),
A (high precision), N (direct iteration), L (Lean). None of them reuses the paper's own script,
except `N_followup.py`, which loads it to compare numbers.

## Layout

| Folder or file | What it holds |
|---|---|
| `paper/` | Corrected paper and sheet, LaTeX and PDF |
| `lean/` | Lean 4 project: `SchrodingerEdge.lean` and the one file it imports |
| `python/` | The check scripts (run them from inside this folder) |
| `outputs/` | Saved output of every run and the four reports `E.md`, `A.md`, `N.md`, `L.md` |
| `reference/` | The paper as it stood before correction (`paper_snapshot.tex`) and the paper's own script |
| `BRIEFS.md` | What each of the four runs was asked to check |
| `apply_corrections.py` | Turns `reference/paper_snapshot.tex` into `paper/cal_schrodinger_collapse.tex`; every replacement must match exactly once |

## Lean (`lean/CalChecks/`)

| File | Theorems | What it proves |
|---|---|---|
| `SchrodingerEdge.lean` | 28 | A normalizable memory has no root on or outside the unit circle; flat-weight roots below, at and above alpha = 1; the weights tend to 1 as alpha -> 1; the homogeneous recurrence above the edge; concurrence equals 1/gamma and sech of the rapidity; the self-normalization lemma |
| `Basic.lean` | 42 | Imported by the file above (purity and boost results from the Born, Lorentz, Killing sheet) |

Lean 4.35.0, Mathlib v4.35.0, no `sorry`, standard axioms only. Build with
`lake exe cache get` then `lake build CalChecks` from `lean/`. Saved compiler output:
`outputs/L_lean_output.txt`.

## Python (`python/`)

Prefix tells the method: `E_` exact (sympy), `A_` mpmath at 40 digits or more, `N_` direct
iteration (numpy), `G_` checks written for the sheet.

| Script | Method | What it tests |
|---|---|---|
| `E_exact.py` | sympy, exact | 61 checks (70 together with the 9 of `E_rotation.py`) labelled by the paper's statement: memory weights, Lemma W, Lemma modes, Lemma flat, dispersion, unitarity exponents, feedback, the normalizable-memory inequalities, accumulator, carrier, Lorentz, minors, composition, Born decay rate |
| `E_rotation.py` | sympy, exact | The rotation theorem: the formal expansion derived from scratch and solved order by order, with the next-order terms. Its last block is a numerical sanity check and is labelled so |
| `E_side_numeric.py` | numpy | Side note: why the "width 100" row of the normalizable-kernel table depends on the 400-lag cut |
| `A_common.py` | - | Shared special-function code for the `A_` scripts (weights, the Jonquiere continuation). Not a test |
| `A_weyl_lemmaW.py` | mpmath | Lemma weyl (the Weyl integral) and Lemma W (Jonquiere's formula) against direct summation and `polylog` |
| `A_rotation.py` | mpmath, 80 digits | Exact co-rotating roots down to 1 - alpha = 1e-40 against the rotation theorem; the modes, mass and diffusion tables |
| `A_feedback.py` | mpmath | Above the edge: the growing root, brake and fading thresholds, the feedback table |
| `A_normalizable.py` | mpmath, numpy | Largest root for the four normalizable kernels (the table) |
| `A_norm_trunc.py` | numpy | How rows 2 and 3 of that table move with the lag cut (400 to 3200) |
| `A_argp.py` | mpmath | Argument principle: counts roots on or outside the unit circle for 0 < alpha < 1 |
| `N_own.py` | numpy, mpmath | Independent iteration of the update rule, written before reading the paper's script. Sections: `modes`, `orient`, `packet`, `closure`, `readout`, `all` (`python N_own.py modes`) |
| `N_followup.py` | numpy | The same iteration with the paper's script's own metric definitions, to explain where numbers differ. Loads `N_work/cal_edge_transducer.py` |
| `N_work/cal_edge_transducer.py` | - | Unchanged copy of the paper's script, used by `N_followup.py` |
| `G_check.py` | sympy, exact | G1 the corrected impulse response above the edge; G2 the SL(2,R) class of the flat-weight recurrence; G3 the long-wavelength unitarity exponent; G4 the diffusive ratio |

`E_exact.py` reports one failure by design: it tests the impulse response as the paper first
stated it, `exp(-s|n-n0|)/(2 sinh s)`, which has the wrong sign and answers a source held on.
The paper and the sheet now carry the corrected form, which `G_check.py` (G1) verifies.

## Outputs (`outputs/`)

| File | From |
|---|---|
| `E.md`, `A.md`, `N.md`, `L.md` | The four reports: verdict per statement, what ran, problems found |
| `E_exact.log`, `E_exact_myrun.out`, `E_rotation_myrun.out` | Runs of the `E_` scripts |
| `A_*.out`, `A_rotation_myrun.out` | Runs of the `A_` scripts |
| `N_out_modes.txt`, `N_out_orient.txt`, `N_out_packet.txt`, `N_out_closure.txt`, `N_out_readout.txt` | Runs of `N_own.py`, one per section |
| `N_out_followup.txt` | Run of `N_followup.py` |
| `N_author_output.txt` | Output of the paper's own script, unchanged |
| `L_lean_output.txt`, `L_myrun.txt` | Lean compiler output for `SchrodingerEdge.lean` |

## What the checks changed in the paper

The impulse response above the edge; the wavepacket table now shows the norm lost and the
distance without rescaling; "70 to 97 percent of the norm" is a ratio of amplitudes; the brake
thresholds are for the simulated weights (0.097, 0.513); the long-wavelength exponent keeps its
-(pi/4)(1 - alpha) term; two rows of the normalizable-kernel table depended on a 400-lag cut;
the frozen carrier at alpha = 1 is stated for convergent sums.

Not proved: the rotation theorem (formal expansion, reproduced symbolically and matched by exact
roots to 34 digits); Jonquiere's formula (cited, verified numerically); the two-carrier
"no interaction" remark, which no check covers.
