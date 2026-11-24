# C1: proof of the three-way identity (part 1)

Claim. For X a Hermitian 2x2 complex matrix, X0 = X - (1/2) tr(X) 1, and kappa the Killing
form of su(2), kappa(A, B) = tr(ad_A ad_B):

    det X = (tr X)^2 / 4 + (1/8) kappa(i X0, i X0).

Each step names the hypothesis it uses. [2x2] = size is 2; [char] = 2 is invertible in the
ground ring; [Herm] = X is Hermitian; [none] = pure algebra over any commutative ring.
Steps are checked one by one in `C1_check.py` (same numbering).

## Part I: the determinant side (no Hermitian hypothesis)

1. Cayley-Hamilton. [2x2] For X = [[p, q], [r, w]], direct multiplication gives
   X^2 - (p + w) X + (pw - qr) 1 = 0, that is X^2 - tr(X) X + det(X) 1 = 0.
   (Entries: (1,1): p^2 + qr - p^2 - pw + pw - qr = 0; (1,2): pq + qw - pq - wq = 0; the
   others are symmetric.) Valid over any commutative ring.

2. Take the trace of step 1. [2x2, char] tr(1) = 2, so tr(X^2) - (tr X)^2 + 2 det X = 0 and
   det X = ((tr X)^2 - tr(X^2)) / 2.

3. Traceless decomposition. [2x2, char] X0 = X - (tr X / 2) 1 has tr X0 = tr X - (tr X/2) 2 = 0.
   Squaring X = X0 + (tr X / 2) 1: X^2 = X0^2 + tr(X) X0 + (tr X)^2/4 1. Taking traces and
   using tr X0 = 0: tr(X^2) = tr(X0^2) + (tr X)^2 / 2.

4. Substitute 3 into 2. [2x2, char]
   det X = ((tr X)^2 - tr(X0^2) - (tr X)^2/2)/2 = (tr X)^2/4 - (1/2) tr(X0^2).

   Everything so far holds for every 2x2 matrix over any commutative ring in which 2 is
   invertible. The identity is now equivalent to: tr(X0^2) = -(1/4) kappa(i X0, i X0).

## Part II: the Killing form side

5. Where A = i X0 lives. [Herm] If X is Hermitian then tr X is real, X0 is Hermitian and
   traceless, so A = i X0 satisfies A^dagger = -A and tr A = 0: A is in su(2). Concretely
   X = t 1 + x.s with t = tr(X)/2 and x_k = tr(X0 s_k)/2 real. This is the only place
   Hermiticity enters: it is what makes "Killing form of su(2)" applicable to i X0.

6. A basis and its brackets. [none] Put E_k = -(i/2) s_k. From s_j s_k = delta_jk 1 +
   i eps_jkl s_l one gets [s_j, s_k] = 2 i eps_jkl s_l, hence
   [E_j, E_k] = (-1/4)(2i) eps_jkl s_l = eps_jkl (-(i/2) s_l) = eps_jkl E_l.
   The E_k are a real basis of su(2) (3 linearly independent anti-Hermitian traceless
   matrices in a 3-dimensional real space).

7. The matrix of ad_A. [none] For A = a.E (a real 3-vector) and V = v.E,
   [A, V] = a_j v_k eps_jkl E_l = (a x v).E. So in the basis E, ad_A is the cross-product
   matrix [a]_x, with entries ([a]_x)_{lk} = eps_{ljk} a_j.

8. Trace of ad_A squared. [none] a x (a x v) = a (a.v) - |a|^2 v, so
   ad_A^2 = a a^T - |a|^2 1_3 and
   kappa(A, A) = tr(ad_A ad_A) = |a|^2 - 3 |a|^2 = -2 |a|^2.

9. Compare with the trace form. [none] A^2 = (-1/4)(a.s)^2 = (-1/4)|a|^2 1 (using
   s_j s_k + s_k s_j = 2 delta_jk), so tr(A^2) = -|a|^2 / 2. Comparing with step 8:

       kappa(A, A) = 4 tr(A^2)   for all A in su(2).

   Both sides are quadratic forms of symmetric bilinear forms, so by polarization
   kappa(A, B) = 4 tr(AB). On the basis: kappa(E_j, E_k) = -2 delta_jk. This is derived,
   not cited; the constant 4 = 2n with n = 2.

10. Apply to A = i X0. [Herm, for A in su(2)] tr(A^2) = i^2 tr(X0^2) = -tr(X0^2), so by 9
    tr(X0^2) = -(1/4) kappa(i X0, i X0). In coordinates X0 = x.s, so
    i X0 = i x.s = (-2x).E, that is a = -2x, hence kappa(i X0, i X0) = -2 |a|^2 = -8 |x|^2 (the part 2 readout).

11. Conclusion. Substitute 10 into 4:
    det X = (tr X)^2/4 - (1/2)(-(1/4)) kappa(i X0, i X0) = (tr X)^2/4 + (1/8) kappa(i X0, i X0).
    In coordinates: t^2 - |x|^2 = (2t)^2/4 + (1/8)(-8 |x|^2). QED.

## Which hypotheses are needed

- 2x2: essential. Steps 1-4 use the 2x2 Cayley-Hamilton and tr(1) = 2; det is quadratic only
  for n = 2. For n = 3, X = 2 * 1 has X0 = 0, det X = 8, (tr X)^2/4 = 9, so no constant on the
  kappa term repairs it (step 12b). Also the Killing constant is 2n (6 on su(3), checked).
- Hermitian: needed only so that i X0 lies in su(2) and the real Killing form of su(2) is the
  form being evaluated. It is not needed for the algebra: for every complex 2x2 X the same
  formula holds with the complex-bilinear Killing form of sl(2,C) (step 12a), because steps
  6-9 are polynomial identities in a and remain true for complex a. It fails if one uses a
  sesquilinear or realified form on sl(2,C) instead (not tested here; this is by the
  argument, since the realified Killing form is 2 Re of the complex one).
- X = J J^dagger and positivity: not needed. Indefinite and negative Hermitian X are covered.
- Field: 2 must be invertible (the formula itself contains 1/4 and 1/8). The factor i is a
  convention: over any commutative ring with 1/2, det X = (tr X)^2/4 - (1/8) kappa(X0, X0)
  with kappa the Killing form of sl(2), since kappa(iA, iA) = -kappa(A, A). This last general
  ring statement follows from steps 1-4 and 6-9 being polynomial identities; the script
  checks it with free commuting symbols for the entries (steps 1-4, 12a), not in positive
  characteristic.
- Constants: 1/4 and 1/8 are forced, given kappa = tr(ad ad) with no extra normalization.

## Machine check

`python C1_check.py` (sympy): steps 1-11 and scope checks 12a, 12b, 12c all passed. In the
script the ad matrices are built by solving [A, E_k] = sum_j c_j E_j for the coordinates, and
step 7 asserts that this agrees with the cross-product matrix used in the hand derivation.
