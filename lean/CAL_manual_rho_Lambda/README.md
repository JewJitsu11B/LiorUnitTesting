# CAL Unified Manual, cosmological-constant relation: one test

Source: the CAL Unified Manual, Eqs. 78-81:

    rho_Lambda = M_Pl^4 * (7/135) * (t_Pl / t)^alpha,   alpha = 2 - 1/280 + 1/14000 + 1/140000

There is no Lean file and no paper copy in this folder; this relation was tested once, with one
script, alongside the alpha = 2 work.

| File | Method | What it tests |
|---|---|---|
| `T_rho_lambda.py` | mpmath at 40 digits, exact fractions for the exponent | See below |

What the script does, in order:

1. Eq. 79: the exponent as an exact fraction (279511/140000) and the two small terms
   `(1/280)(1/50) = 1/14000` and `1/(14 (3+7)^4) = 1/140000`.
2. Eq. 81: inverts the relation for the age t with the manual's rounded inputs
   (manual: 4.351e17 s = 13.79 Gyr), then again with unrounded CODATA constants.
3. The same inversion at alpha = 2 exactly (10.79 Gyr), to show what the correction to 2 does.
4. Sensitivity of t to alpha, to rho_Lambda and to M_Pl; the alpha that returns 13.787 Gyr
   exactly; the age for each partial sum of the exponent.
5. The input: rho_Lambda recomputed from Planck 2018 (H0 = 67.36, Omega_Lambda = 0.6847)
   and compared with the manual's 2.888e-47 GeV^4; the age that follows from the Planck value;
   the same for two other choices of H0.

Only steps 1 to 3 are PASS or FAIL checks (arithmetic). Steps 4 and 5 print numbers to read;
they are not pass or fail.

Open: where the manual's value 2.888e-47 GeV^4 comes from has not been traced.

```
python T_rho_lambda.py
```
