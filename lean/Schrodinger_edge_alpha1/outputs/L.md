# Agent L report: Lean 4 + Mathlib

## VERDICT
- Thm normalizable: HOLDS. Proved for w : N -> R (w j = paper's w_{j+1}), w j >= 0, Summable w, tsum = 1, 0 < alpha < 1, 0 <= ghat <= 1, 1 <= |z|: z - alpha + (1-alpha) ghat sum' w j z^(-(j+1)) != 0. The paper's proof goes through as written (inequality chain, equality forces z = 1, then F(1) != 0).
- Lemma flat, identity and root cases (1)-(3): HOLDS. The closing claim phi, s = sqrt|1-alpha|(1+O(|1-alpha|)): NOT CHECKED.
- Thm feedback (1): HOLDS (trivially; see Problems). Thm feedback (3): HOLDS WITH CONDITIONS (sign, see Problems). (2), (4): NOT CHECKED.
- Prop accumulator, discrete part: HOLDS. Continuous (Laplace) part: NOT CHECKED.
- Prop carrier (3), (4); Prop lorentz (1), (3): HOLDS. Carrier (1), (2), lorentz (2), (4): NOT CHECKED.
- Lemma minors (proof step and diag(2,2,0) example): HOLDS. Prop composition: NOT CHECKED.

## WHAT RAN
`lake env lean CalChecks\SchrodingerEdge.lean` in `cal-checks\lean`: exit 0, no errors, no warnings, no `sorry`. All 22 `#print axioms` lines read `[propext, Classical.choice, Quot.sound]`. Output saved in `L_lean_output.txt`.

Theorems (namespace `CalChecks.Edge`):
- `normalizable`: Thm normalizable as above. `norm_memory_le_one`: |W(z)| <= 1.
- `F_one`: F(1) = (1-alpha)(1+ghat) (= 2(1-alpha) at ghat = 1).
- `alpha_zero_root`: at alpha = 0, ghat = 1, Dirac kernel, z = i is a root of norm 1.
- `flat_W`: sum_{j>=1} z^(-j) = 1/(z-1) for |z| > 1.
- `flat_factor`: (z-1)F(z) identity, z != 1.
- `flat_lt_factor`, `flat_lt`: 0 < alpha < 1: phi > 0 exists with cos phi = (1+alpha)/2; quadratic = (z - e^{i phi})(z - e^{-i phi}); both roots have norm 1.
- `flat_eq_one`: alpha = 1 gives (z-1)^2.
- `flat_gt_factor`, `flat_gt`: alpha > 1: s > 0 with cosh s = (1+alpha)/2; roots are exactly e^s, e^{-s}.
- `feedback_frozen`: at alpha = 1 the update's right side is J n.
- `green_homogeneous`, `green_jump`, `green_normalized`, `green_bounded`: G n = e^{-s|n-n0|} solves the recurrence for n != n0; jump at n0 is -2 sinh s; G/(2 sinh s) gives -1 at n0; 0 < G <= 1.
- `weight_tendsto_one`: (j^alpha - (j-1)^alpha)/Gamma(alpha+1) -> 1 as alpha -> 1-.
- `det_closure`: det(J J^H) = |det J|^2.
- `conc_sq`, `conc_eq_sqrt`: C^2 = 1 - |x|^2/x0^2; C = sqrt(...) for x0 > 0.
- `conc_sech`: C = 1/cosh eta. `schmidt_weights`: p+- = e^{+-eta}/(2 cosh eta).
- `lorentz_invariant_density`: |det(GJ)| = |det J|, Q -> G Q G^H, det Q unchanged.
- `self_normalized`: 2J = J J^H implies J^H = J and J J = 2J (any size).
- `minors_example`: diag(2,2,0): 2J = J J^H, det 0, tr(J/2) = 2.

Reused from `CalChecks.Basic`: `herm`, `herm_eq`, `det_herm`, `det_congruence_SL`, `M2`.

## PROBLEMS FOUND
1. Thm feedback (3), sign. With the operator as displayed, L[G] = J_{n+1} - (1+alpha)J_n + J_{n-1}, the paper's e^{-s|n-n0|}/(2 sinh s) gives L = -delta, not +delta (`green_normalized`). Correct only up to sign convention for the source.
2. Thm normalizable: alpha > 0 is essential; alpha = 0 has a unit-circle root (`alpha_zero_root`). Not an error: the paper states alpha in (0,1). The proof's "F(1) = 2(1-alpha)" is the ghat = 1 value; in general (1-alpha)(1+ghat).
3. Thm feedback (1): `feedback_frozen` is 0 times the memory sum. Lean's `tsum` is 0 when divergent, so this does not address whether the sum converges at alpha = 1 (flat weights, non-decaying past). Not checked further.

## FILES
- `C:\Users\Sam Leizerman\cal-checks\lean\CalChecks\SchrodingerEdge.lean`
- `C:\Users\Sam Leizerman\cal-checks\schrodinger\L_lean_output.txt`
- `C:\Users\Sam Leizerman\cal-checks\schrodinger\results\L.md`
