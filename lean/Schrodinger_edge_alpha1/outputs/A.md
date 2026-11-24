# Agent A report (mpmath, 40 to 140 digits, own code)

## VERDICT
- Lemma weyl, Def weyl symbol and phase: HOLDS. Quadrature matches the formula to 1e-45 for five alpha in (0,1), both signs of omega. At alpha = 1 the partial integrals oscillate without limit; at alpha = 1.5 they grow like sqrt(T).
- Lemma W (eq. jonquiere): HOLDS with principal powers, the paper's convention. Agreement: direct sums for |z|>1, 4e-44; unit circle (Levin-accelerated series, independent Bose-type integral), 1e-45; mpmath.polylog, 7e-46. Imprecision: for complex phi the right side has a cut on Re phi = 0 mod 2 pi, Im phi > 0.
- Thm rotation: HOLDS. pi/4 recovered to 34 digits (eps down to 1e-40) and 1/2 exactly, for ghat = 1, 0.9, 0.5. Error orders are right. Measured next terms: Re phi = sqrt(eps g)[1 + eps((1/4) ln(eps g) + g/24 + (1-g)/4 - (1-g)^2/(8g))]; Im phi continues with -(g/4) eps^2 ln eps.
- Table modes (six rows), Table mass, Table diffusion: every printed digit reproduced. The 0.78667 remark holds; the gap to pi/4 is -(1/4) sqrt(eps) ln eps.
- Thm unitary predicted ratio: HOLDS at leading order (0.09214 predicted, 0.09187 exact at eps = 1e-2, k = 0.1).
- Thm feedback (2), (4): HOLDS; criterion necessary and sufficient on 480 grid points plus 30 argument-principle counts. Table feedback: 14 of 15 entries agree. Caption thresholds: FAIL.
- Thm normalizable: HOLDS (3000 random kernels). Table normalizable: reproduced, WITH CONDITIONS.
- Other roots: none. 54 (alpha, ghat) cases, alpha 0.05 to 0.999999: zero roots with 1 <= |z| < e^6. Inside, on the principal sheet, exactly two (the conjugate pair) for alpha 0.3 to 0.9999.

## PROBLEMS FOUND
1. Table feedback thresholds. With the alpha = 1.01 weights, W_lambda(1) = 9.7302 and 104.19, so the criterion gives b > 0.0969 (lambda = 0.1) and b > 0.5126 (lambda = 0.01). The paper's 0.095 and 0.501 are the flat-weight values (0.09509, 0.50125). At b = 0.505, lambda = 0.01 root and direct iteration both give growth +3.03e-4.
2. Table feedback, b = 0, lambda = 0.01: root and iteration give +0.09635; paper prints +0.0964.
3. Table normalizable rows 2 and 3 reflect the 400-lag cut. Exponential width 100: 0.99205 at 400 lags, 0.99008 untruncated. Power law j^-1.5: 0.98656, 0.99217, 0.99566, 0.99766 at 400, 800, 1600, 3200 lags, tending to 1; the untruncated kernel has a branch point at z = 1 and decays algebraically.
4. Thm rotation converges slowly: exact Im phi exceeds the stated value by 13.6% at eps = 1e-2 and 2.9% at 1e-4 (ghat = 1).

## WHAT RAN
`python A_weyl_lemmaW.py`, `A_rotation.py`, `A_feedback.py`, `A_argp.py`, `A_normalizable.py`, `A_norm_trunc.py`; all completed.

## FILES
`A_common.py`, the six scripts, `results\A.md`, `results\A_*.out`.
