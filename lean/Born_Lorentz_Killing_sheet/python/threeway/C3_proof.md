# C3: proof of the three-way identity by invariant theory

Claim (part 1). For X in Herm(2), X0 = X - (1/2) tr(X) 1, and kappa the Killing form of su(2),
kappa(A, B) = tr(ad_A ad_B):

    det X = (tr X)^2 / 4 + (1/8) kappa(i X0, i X0).

Conventions. Coordinates X = t 1 + x1 sx + x2 sy + x3 sz, (t, x) in R^4. Real basis of su(2):
E_k = i sigma_k. ad matrices are obtained by solving [E_j, E_k] = sum_l c^l_{jk} E_l, and
kappa is tr(ad ad) extended bilinearly. Machine result: kappa(E_j, E_k) = -8 delta_jk.
All "checked" statements below are assertions in `C3_check.py` (exact sympy, no floats).

## The argument

Let V be the 10-dimensional real space of homogeneous quadratic polynomials on Herm(2) = R^4.
G = SU(2) acts on Herm(2) by X -> U X U^dagger, hence on V. Write V^G for the invariants and
V^g for the quadratics killed by the infinitesimal action X -> [A, X], A in su(2).

Step 1 (both sides lie in V). det X, (tr X)^2 and kappa(i X0, i X0) are homogeneous real
quadratics in (t, x). For kappa this uses: X Hermitian implies i X0 is traceless
anti-Hermitian, so it is in su(2) and kappa is defined on it. Checked (i-a).

Step 2 (both sides lie in V^G). Checked exactly (i-b): with U = a1 + i(a2 sx + a3 sy + a4 sz),
which is sqrt(n) times a general element of SU(2), n = a1^2+a2^2+a3^2+a4^2, the polynomial
identity f(U X U^dagger) = n^2 f(X) holds in the eight symbols (a, t, x) for
f = det, tr^2, kappa(i X0, i X0). On n = 1 this is invariance. (Conceptually: det and tr are
conjugation invariant, U X0 U^dagger = (U X U^dagger)0, and kappa is Ad-invariant.)

Step 3 (dim V^G = 2). Differentiating f(exp(sA) X exp(-sA)) = f(X) at s = 0 gives
V^G subset V^g. V^g is the nullspace of a linear system: for each E_k the vector field
X -> [E_k, X] is written in coordinates and applied to a general quadratic with 10 unknown
coefficients. Checked (ii): 21 equations, rank 8, nullspace dimension 2, basis
{t^2, x1^2 + x2^2 + x3^2}. By Step 2, tr^2 = 4 t^2 and kappa = -8 |x|^2 are in V^G and they
are independent, so V^G = V^g and dim V^G = 2 exactly. (No appeal to connectedness of SU(2)
is needed, because the reverse inclusion is supplied by Step 2.)

Step 4 (two evaluations suffice). The evaluation map V^G -> R^2, f -> (f(1), f(sz)), has
rank 2 (checked (iii)), so it is injective: an invariant quadratic vanishing at 1 and at sz
is zero.

Step 5 (the evaluations). With kappa from tr(ad ad):

    X = 1 :  det = 1,  tr^2 = 4, kappa = 0   ->  rhs = 4/4 + 0      = 1
    X = sz:  det = -1, tr^2 = 0, kappa = -8  ->  rhs = 0 + (-8)/8   = -1

So D = det X - (tr X)^2/4 - (1/8) kappa(i X0, i X0) is in V^G and vanishes at 1 and sz;
by Step 4, D = 0 on Herm(2). QED.

## What this shows that a direct expansion does not

1. Uniqueness of the relation and of the constants. Three invariant quadratics (det, tr^2,
   kappa) live in a 2-dimensional space, so there is exactly one linear relation among them up
   to scale; any two of the three are independent (checked), so the relation involves all
   three. Solving det = alpha tr^2 + beta kappa at the two points gives alpha = 1/4,
   beta = 1/8 and nothing else. The mutated constants (1/4, 1/16, sign flip, tr^2/2) cannot
   hold, without testing them one by one.
2. Completeness. Every SU(2)-invariant quadratic readout of X (tr(X^2), tr(rho-type purities
   before normalization, |x|^2, and so on) is a combination of any two of the three. No fourth
   independent quadratic invariant exists.
3. The constants are normalization data only. 1/4 is fixed by X = 1 (tr 1 = 2) and 1/8 by
   X = sz, that is by kappa(i sz, i sz) = -8. A different normalization of kappa changes 1/8
   and nothing else.
4. Why 2x2. For Herm(3) under su(3) the same computation gives invariant quadratic dimension 2
   as well (45 unknowns, rank 43), but det is cubic there, so it is not in that space and no
   choice of constants gives an analogue of this shape. The identity exists for n = 2 because
   det is quadratic exactly then.
5. The choice of test points matters. {1, sz} and {1, 1 + sz} separate V^G; {1, 2*1} and
   {sz, sx} do not (rank 1, checked), so those pairs would not prove the identity.

## Hypotheses

- 2x2: essential (item 4).
- Hermitian: used so that i X0 is in su(2) and so that the SU(2) argument applies. Not
  machine-checked here, but argued: both sides are polynomial in the entries of X and
  Herm(2) is a real form of M_2(C), so the identity extends to all complex 2x2 X when kappa
  is replaced by the complex-bilinear Killing form of sl(2, C).
- X = J J^dagger (positive semidefinite) is not needed; X = sz is indefinite and is one of
  the test points.
