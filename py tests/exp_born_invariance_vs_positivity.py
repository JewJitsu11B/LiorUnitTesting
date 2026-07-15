"""
Positive XOR invariant: what a QUADRATIC form on C (x) H can and cannot be.

The claim under test is NEGATIVE. For a quadratic form the biquaternion algebra offers
POSITIVE or INVARIANT, never both, and the reason is structural rather than a matter of
picking a better form:

    N(q) = q qbar = sum_mu q_mu^2      BILINEAR (a pure squaring; no conjugation anywhere),
                                       equals det X_q, SL(2,C)-invariant,
                                       but COMPLEX and INDEFINITE.
    <q^dag q>_0  = sum_mu |q_mu|^2     SESQUILINEAR (conjugates one slot), real and
                                       positive definite, but NOT boost-invariant.

Over an algebraically closed field every nondegenerate quadratic form in n >= 2 variables is
ISOTROPIC. So no complex-BILINEAR form on C (x) H can be positive definite, whatever basis or
sign convention is chosen. Positivity requires conjugation; invariance requires squaring.

The obvious escape is to conjugate twice: |N(W)|^2 = N(W) conj(N(W)) is both non-negative and
SL(2,C)-invariant, so it does defeat positive-XOR-invariant as literally stated. This script
prices that escape, and the price is fatal: |N(W)|^2 is QUARTIC not quadratic, it is DEGENERATE
(identically zero on the whole null cone, hence non-negative but never positive DEFINITE), on
the paper's own central W = w*1 it returns |w|^4 rather than the Born target |w|^2, and it does
not normalize.

CLAIMS VERIFIED (sympy/numpy in float64; cross-checked against cal.biquaternion in complex64):
  Q1  N(q) = q qbar is a pure SQUARING: grade-0 = sum_mu q_mu^2 with the other three grades
      identically zero, and no conjugation appears anywhere in the product rule. Cross-checked
      against cal's quat_norm_sq and det(biquat_to_matrix).
  Q2  Real-coordinate expansion q_mu = x_mu + i y_mu, verified symbolically:
          N(q)        = sum_mu (x_mu^2 - y_mu^2) + 2i sum_mu x_mu y_mu    <- both signs, complex
          <q^dag q>_0 = sum_mu (x_mu^2 + y_mu^2)                          <- all plus, always
  Q3  <q^dag q>_0 is SESQUILINEAR, real, and positive definite (zero only at q = 0).
  Q4  Under SL(2,C) boosts X -> L X L^dag (det L = 1): N is invariant and |N|^2 is invariant,
      while <q^dag q>_0 = (1/2)||X||_F^2 is NOT. Control: <q^dag q>_0 IS invariant under the
      SU(2) rotation subgroup, so the failure is specifically boosts, not sloppiness.
  Q5  ISOTROPY, exhibited: (1 + i*i_q), (1 + i*j_q), (i_q + i*j_q) are nonzero with N(q) = 0
      but <q^dag q>_0 = 2. A positive definite form cannot vanish on a nonzero vector.
      => POSITIVE XOR INVARIANT holds for quadratic forms.
  E1  The escape works on its own terms: |N(W)|^2 is non-negative AND boost-invariant.
  E2  But it is QUARTIC: sympy.Poly total degree 4 in the real coordinates, versus 2 for both
      N and the Born form.
  E3  And it is DEGENERATE: identically zero on the whole null cone, so it is positive
      SEMI-definite only. Positive-DEFINITE XOR invariant survives the escape untouched.
  E4  DECISIVE: for the paper's own CENTRAL amplitude W = w*1 (born_axiomatic.tex abstract,
      W = <psi|psi_0> = g + i t_p), det W = w^2, so |N(W)|^2 = |w|^4 -- not the Born target
      |w|^2. Solved symbolically: the two agree only at |w| in {0, 1}.

NEGATIVE FINDINGS (reported and kept as negatives, not reframed):
  N1  Probabilities must sum to 1. Parseval gives sum_k |c_k|^2 = 1 exactly for random
      normalized states in dims 2, 3, 5, 10; sum_k |c_k|^4 does not, and the deficit grows
      with dimension. The quartic rule is not a probability distribution.
  N2  Spin-1/2 at angle theta: cos^4(th/2) + sin^4(th/2) = 1 - (1/2) sin^2(theta), equal to 1
      only at theta = 0, pi. At theta = pi/2 the two outcomes total 0.5 -- half the
      probability is simply gone.
  N3  The repair divisor is STATE-DEPENDENT (1.0, 2.0, 3.0 for (1,0), (1,1)/sqrt2,
      (1,1,1)/sqrt3), so no algebraic constant fixes it. Restoring normalization means
      dividing by sum_k |c_k|^4, i.e. re-introducing by hand the external sum over outcomes
      that Thm 6.1 claims to have eliminated ("no Gleason-type premise, external partition
      integral, or auxiliary Hilbert-space postulate is invoked").
  D1  DOCUMENTATION DEFECT in born_axiomatic.tex: the abstract (line 38) says "the algebra's
      split (2,2) norm", Lemma 4.2 (line 139) says "the split-signature (2,2) carrier", and
      the scope section (line 54) writes the selection as "(2,2) -> (1,3)". But Re N(q) =
      sum_mu (x_mu^2 - y_mu^2) on R^8 has eigenvalues [-1,-1,-1,-1,+1,+1,+1,+1]: signature
      (4,4). (2,2) is what appears on particular real 4-dim slices, e.g. the real span of
      {1, i_q, i*j_q, i*k_q}, giving x0^2 + x1^2 - x2^2 - x3^2. The Lemma 4.2 basis
      {1, h i, h j, h k} is a DIFFERENT slice and is (1,3), which is the slice actually used.
      The defect is in naming the 8-dim carrier's norm (2,2); the (1,3) conclusion stands.

Not covered here (see exp_two_closures.py, C1-C7): multiplicativity of the reduced norm, the
bar-vs-dagger interference contrast, and the basic null-cone complementarity. This script adds
the symbolic bilinear-vs-sesquilinear separation, the SL(2,C) boost behaviour of each form, the
degree/degeneracy pricing of the |N|^2 escape, the normalization test, and D1.
"""
import os
import sys

import numpy as np
import sympy as sp
import torch

# cal/ lives at the repo root, one level above "py tests"; this script is run from inside
# "py tests" as: python exp_born_invariance_vs_positivity.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cal.biquaternion import (quat_mul, quat_conj, hermitian_conj, quat_norm_sq,
                              biquat_to_matrix, matrix_to_biquat, CDTYPE)

torch.manual_seed(0)
rng = np.random.default_rng(23)
np.set_printoptions(precision=6, suppress=True)

TOL = 1e-4         # cal is complex64 (float32 precision)
TOL_EXACT = 1e-10  # pure numpy/sympy float64 sections


def ok(b):
    return "PASS" if b else "**FAIL**"


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def to_np(t):
    return t.detach().numpy().astype(np.complex128)


def mat(q):
    """cal biquaternion (...,4) -> numpy (...,2,2) complex128."""
    return to_np(biquat_to_matrix(q))


def rand_biquat(n):
    return (torch.randn(n, 4) + 1j * torch.randn(n, 4)).to(CDTYPE)


# ---------------------------------------------------------------------------
# Symbolic biquaternion product, identical to cal.quat_mul's rule (paper Eq. 3).
# Kept symbolic so that "no conjugation appears anywhere" is a readable fact about the
# multiplication rule, not an assertion.
# ---------------------------------------------------------------------------
def biquat_mul(A, B):
    a0, a1, a2, a3 = A
    b0, b1, b2, b3 = B
    return [sp.expand(c) for c in [
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0]]


def bar(q):
    """The BAR: negate the vector part, do NOT conjugate coefficients."""
    return [q[0], -q[1], -q[2], -q[3]]


def dag(q):
    """The DAGGER: conjugate coefficients AND negate the vector part."""
    return [sp.conjugate(q[0]), -sp.conjugate(q[1]),
            -sp.conjugate(q[2]), -sp.conjugate(q[3])]


# ---------------------------------------------------------------------------
# SL(2,C) elements. Boosts are Hermitian with det 1; rotations are unitary with det 1.
# ---------------------------------------------------------------------------
S0 = np.eye(2, dtype=np.complex128)
S1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
S2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
S3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def boost(eta, axis=S3):
    """exp(eta/2 * sigma_axis): Hermitian, det = 1, a pure SL(2,C) boost of rapidity eta."""
    return np.cosh(eta / 2) * S0 + np.sinh(eta / 2) * axis


def rotation(theta, axis=S3):
    """exp(-i theta/2 * sigma_axis): unitary, det = 1, an SU(2) rotation."""
    return np.cos(theta / 2) * S0 - 1j * np.sin(theta / 2) * axis


def act(L, X):
    """The Lorentz action X -> L X L^dag on M2(C)."""
    return L @ X @ L.conj().T


def born_scalar(X):
    """<q^dag q>_0 = (1/2) Re tr(X^H X) = (1/2) ||X||_F^2."""
    return 0.5 * np.real(np.trace(X.conj().T @ X))


def main():
    q = [sp.Symbol(f'q_{i}') for i in range(4)]           # complex, unconstrained
    x = [sp.Symbol(f'x_{i}', real=True) for i in range(4)]
    y = [sp.Symbol(f'y_{i}', real=True) for i in range(4)]
    xy = x + y
    sub = {q[i]: x[i] + sp.I * y[i] for i in range(4)}

    # -----------------------------------------------------------------------
    banner("Q1. N(q) = q qbar is a pure SQUARING -- no conjugation in the product rule")
    N = biquat_mul(q, bar(q))
    N0 = N[0]
    vec_zero = all(sp.simplify(c) == 0 for c in N[1:])
    is_sum_sq = sp.simplify(N0 - sum(qi ** 2 for qi in q)) == 0
    print(f"  N(q) grade-0 = {N0}")
    print(f"  grades 1,2,3 identically zero (N is a pure scalar):     {ok(vec_zero)}")
    print(f"  N(q) == sum_mu q_mu^2:                                  {ok(is_sum_sq)}")
    has_conj = N0.has(sp.conjugate)
    print(f"  any conjugate() anywhere in N(q):                       {has_conj}"
          f"   -> BILINEAR: {ok(not has_conj)}")
    print("  >> The coefficients are SQUARED, never conjugated. This falls straight out of")
    print("     the multiplication rule; no convention was inserted.")

    # Cross-check the symbolic form against the package under test.
    qq = rand_biquat(200)
    Xq = mat(qq)
    N_cal = to_np(quat_norm_sq(qq))
    det_cal = np.linalg.det(Xq)
    qqbar = to_np(quat_mul(qq, quat_conj(qq)))
    cal_scalar = np.abs(qqbar[..., 1:]).max() < TOL
    cal_sumsq = np.allclose(N_cal, (to_np(qq) ** 2).sum(-1), atol=TOL, rtol=1e-4)
    cal_det = np.allclose(N_cal, det_cal, atol=TOL, rtol=1e-4)
    print(f"\n  cal cross-check on 200 random biquaternions:")
    print(f"    quat_mul(q, quat_conj(q)) is a pure scalar:           {ok(cal_scalar)}"
          f"   max|vec|={np.abs(qqbar[..., 1:]).max():.2e}")
    print(f"    quat_norm_sq(q) == sum_mu q_mu^2:                     {ok(cal_sumsq)}")
    print(f"    quat_norm_sq(q) == det(biquat_to_matrix(q)):          {ok(cal_det)}"
          f"   max|diff|={np.abs(N_cal - det_cal).max():.2e}")

    # -----------------------------------------------------------------------
    banner("Q2. Real-coordinate expansion: q_mu = x_mu + i y_mu")
    N_real = sp.expand(N0.subs(sub))
    ReN = sum(xi ** 2 - yi ** 2 for xi, yi in zip(x, y))
    ImN = 2 * sum(xi * yi for xi, yi in zip(x, y))
    matches = sp.simplify(N_real - (ReN + sp.I * ImN)) == 0
    print(f"  N(q) = {sp.simplify(N_real)}")
    print(f"\n  claimed form: sum(x_mu^2 - y_mu^2) + 2i sum(x_mu y_mu)")
    print(f"  symbolically equal:                                     {ok(matches)}")
    print(f"    Re N = {ReN}")
    print(f"    Im N = {ImN}")

    D = biquat_mul(dag(q), q)
    D0 = sp.simplify(D[0])
    D_real = sp.simplify(sp.expand(D0.subs(sub)))
    born_form = sum(xi ** 2 + yi ** 2 for xi, yi in zip(x, y))
    d_matches = sp.simplify(D_real - born_form) == 0
    print(f"\n  <q^dag q>_0 = {D0}")
    print(f"  <q^dag q>_0 = {D_real}")
    print(f"  == sum(x_mu^2 + y_mu^2):                                {ok(d_matches)}")
    print("  >> N squares COMPLEX numbers: x^2 - y^2 terms, signs of both kinds, plus a")
    print("     surviving imaginary part. The dagger form squares REAL numbers: all plus.")

    # -----------------------------------------------------------------------
    banner("Q3. <q^dag q>_0 is SESQUILINEAR, real, and positive definite")
    dag_has_conj = D[0].has(sp.conjugate)
    print(f"  conjugate() appears in <q^dag q>_0:                     {dag_has_conj}"
          f"   -> SESQUILINEAR: {ok(dag_has_conj)}")
    print("  >> Conjugation is an EXTRA operation laid on top of the product rule. It is not")
    print("     'multiplying biquaternions' in the bilinear sense at all.")

    born_cal = to_np(quat_mul(hermitian_conj(qq), qq))[..., 0]
    sumsq = (qq.real ** 2 + qq.imag ** 2).sum(-1).numpy().astype(np.float64)
    real_ok = np.abs(born_cal.imag).max() < TOL
    pos_ok = bool((born_cal.real > 0).all())
    eq_ok = np.allclose(born_cal.real, sumsq, atol=TOL, rtol=1e-4)
    zero_only = born_scalar(np.zeros((2, 2), dtype=np.complex128)) == 0.0
    print(f"\n  cal cross-check on 200 random biquaternions:")
    print(f"    <q^dag q>_0 is real:                                  {ok(real_ok)}"
          f"   max|imag|={np.abs(born_cal.imag).max():.2e}")
    print(f"    <q^dag q>_0 > 0 on every nonzero sample:              {ok(pos_ok)}"
          f"   min={born_cal.real.min():.4f}")
    print(f"    <q^dag q>_0 == sum_mu |q_mu|^2:                       {ok(eq_ok)}")
    print(f"    <q^dag q>_0 == 0 at q = 0 (definite, not just >= 0):  {ok(zero_only)}")

    # -----------------------------------------------------------------------
    banner("Q4. Under SL(2,C) boosts: N invariant, |N|^2 invariant, <q^dag q>_0 NOT")
    w = rand_biquat(1)
    X = mat(w)[0]
    print(f"  A single random biquaternion W, acted on by X -> L X L^dag with det L = 1.")
    print(f"  N(W) = det X = {np.linalg.det(X):+.6f}")
    print(f"\n  {'rapidity eta':>13}{'Re N':>12}{'Im N':>12}{'|N|^2':>12}"
          f"{'<W^dag W>_0':>15}")
    etas = [0.0, 0.25, 0.5, 1.0, 1.5]
    dets, mods, borns = [], [], []
    for eta in etas:
        L = boost(eta)
        Xb = act(L, X)
        d = np.linalg.det(Xb)
        dets.append(d)
        mods.append(abs(d) ** 2)
        borns.append(born_scalar(Xb))
        print(f"  {eta:>13.2f}{d.real:>12.6f}{d.imag:>12.6f}{abs(d)**2:>12.6f}"
              f"{born_scalar(Xb):>15.6f}")
    det_inv = max(abs(d - dets[0]) for d in dets) < TOL_EXACT
    mod_inv = max(abs(m - mods[0]) for m in mods) < TOL_EXACT
    born_var = max(abs(b / borns[0] - 1) for b in borns)
    born_not_inv = born_var > 1e-2
    print(f"\n  N is boost-invariant:                                   {ok(det_inv)}"
          f"   max|drift|={max(abs(d - dets[0]) for d in dets):.2e}")
    print(f"  |N|^2 is boost-invariant:                               {ok(mod_inv)}"
          f"   max|drift|={max(abs(m - mods[0]) for m in mods):.2e}")
    print(f"  <W^dag W>_0 is NOT boost-invariant:                     {ok(born_not_inv)}"
          f"   max|ratio-1|={born_var:.3f}")

    print("\n  Control -- the SU(2) rotation subgroup (unitary, det 1). If the dagger form")
    print("  drifted under everything, the finding would be vacuous:")
    print(f"  {'angle theta':>13}{'|N|^2':>12}{'<W^dag W>_0':>15}")
    rot_borns = []
    for th in [0.0, 0.5, 1.0, 2.0]:
        Xr = act(rotation(th), X)
        rot_borns.append(born_scalar(Xr))
        print(f"  {th:>13.2f}{abs(np.linalg.det(Xr))**2:>12.6f}{born_scalar(Xr):>15.6f}")
    rot_inv = max(abs(b / rot_borns[0] - 1) for b in rot_borns) < 1e-8
    print(f"\n  <W^dag W>_0 IS rotation-invariant:                      {ok(rot_inv)}")
    print("  >> The dagger form is an SU(2) invariant, not an SL(2,C) invariant. It fails")
    print("     specifically on boosts -- exactly the non-compact directions a relativistic")
    print("     probability would have to survive.")

    # Package-level restatement, round-tripping through cal.
    Lb = boost(1.0)
    wb = matrix_to_biquat(torch.tensor(act(Lb, X), dtype=CDTYPE).unsqueeze(0))
    N_before = quat_norm_sq(w).item()
    N_after = quat_norm_sq(wb).item()
    b_before = quat_mul(hermitian_conj(w), w)[..., 0].real.item()
    b_after = quat_mul(hermitian_conj(wb), wb)[..., 0].real.item()
    cal_N_inv = abs(N_before - N_after) < 1e-3
    cal_b_inv = abs(b_before - b_after) < 1e-3
    print(f"\n  Same check at package level (cal, complex64, eta = 1.0):")
    print(f"    quat_norm_sq  before -> after: {N_before:+.5f} -> {N_after:+.5f}")
    print(f"      invariant, as it must be:                           {ok(cal_N_inv)}")
    print(f"    <W^dag W>_0   before -> after: {b_before:+.5f} -> {b_after:+.5f}")
    print(f"      NOT invariant, as claimed:                          {ok(not cal_b_inv)}"
          f"   drift = {b_after - b_before:+.5f}")

    # -----------------------------------------------------------------------
    banner("Q5. ISOTROPY over C: no complex-bilinear form on C (x) H is positive definite")
    print("  Over an algebraically closed field every nondegenerate quadratic form in n >= 2")
    print("  variables is ISOTROPIC -- it has nonzero null vectors. C is algebraically closed")
    print("  and N is a nondegenerate quadratic form in 4 complex variables. So N must have")
    print("  null vectors, and a positive definite form cannot. Exhibiting them:\n")
    print(f"  {'q':<16}{'N(q)':>10}{'<q^dag q>_0':>16}{'q nonzero?':>13}")
    null_vecs = [("1 + i*i_q", [1, sp.I, 0, 0]),
                 ("1 + i*j_q", [1, 0, sp.I, 0]),
                 ("i_q + i*j_q", [0, 1, sp.I, 0])]
    isotropy_ok = True
    for name, v in null_vecs:
        val = sp.simplify(biquat_mul(v, bar(v))[0])
        dval = sum(abs(complex(c)) ** 2 for c in v)
        nonzero = any(c != 0 for c in v)
        isotropy_ok &= (val == 0 and abs(dval - 2.0) < TOL_EXACT and nonzero)
        print(f"  {name:<16}{str(val):>10}{dval:>16.1f}{str(nonzero):>13}")
    print(f"\n  all three are nonzero with N = 0 but <q^dag q>_0 = 2:   {ok(isotropy_ok)}")

    # Same elements through cal, so this is a fact about the package too.
    zc = torch.tensor([[1.0 + 0j, 1j, 0, 0]], dtype=CDTYPE)
    print(f"  cal: quat_norm_sq(1 + i*i_q)  = {quat_norm_sq(zc).item().real:+.3e}"
          f"   <q^dag q>_0 = {quat_mul(hermitian_conj(zc), zc)[..., 0].real.item():.3f}")
    print("\n  >> Positivity REQUIRES conjugation. Invariance REQUIRES squaring. For a")
    print("     QUADRATIC form the two are mutually exclusive: POSITIVE XOR INVARIANT.")

    # -----------------------------------------------------------------------
    banner("E1. The escape: |N(W)|^2 conjugates twice -- positive AND invariant")
    print("  |N(W)|^2 = N(W) conj(N(W)) is non-negative by construction and inherits N's")
    print("  boost-invariance (verified in Q4 above). So it DOES defeat positive-XOR-invariant")
    print("  as literally stated. Conceding that. Now the price.")
    escape_pos = all(m >= 0 for m in mods)
    print(f"\n  |N|^2 >= 0 across the boost sweep:                       {ok(escape_pos)}")
    print(f"  |N|^2 boost-invariant (from Q4):                        {ok(mod_inv)}")

    # -----------------------------------------------------------------------
    banner("E2. But |N(W)|^2 is QUARTIC, not quadratic")
    mod_sq = sp.expand(ReN ** 2 + ImN ** 2)
    prod_form = sp.expand(N_real * sp.conjugate(N_real))
    consistent = sp.simplify(prod_form - mod_sq) == 0
    print(f"  N conj(N) == (Re N)^2 + (Im N)^2 symbolically:          {ok(consistent)}")
    deg_N = sp.Poly(N_real, *xy).total_degree()
    deg_born = sp.Poly(born_form, *xy).total_degree()
    deg_mod = sp.Poly(mod_sq, *xy).total_degree()
    print(f"\n  sympy.Poly total degree in the 8 real coordinates (x_0..x_3, y_0..y_3):")
    print(f"    {'N(q)':<28}{deg_N:>3}   {'(quadratic)' if deg_N == 2 else '(?)'}")
    print(f"    {'<q^dag q>_0':<28}{deg_born:>3}   {'(quadratic)' if deg_born == 2 else '(?)'}")
    print(f"    {'|N(q)|^2':<28}{deg_mod:>3}   {'(QUARTIC)' if deg_mod == 4 else '(?)'}")
    degrees_ok = (deg_N == 2 and deg_born == 2 and deg_mod == 4)
    print(f"\n  degrees are 2, 2, 4 as claimed:                         {ok(degrees_ok)}")
    print("  >> The escape leaves the category. The claim was about QUADRATIC forms -- the")
    print("     closure order alpha = 2 is what the parent paper says fixes the form. A")
    print("     quartic is not a quadratic closure; it is a different theory.")

    # -----------------------------------------------------------------------
    banner("E3. And |N(W)|^2 is DEGENERATE: identically zero on the whole null cone")
    print("  Positive SEMI-definite is not positive definite. |N|^2 vanishes wherever N does,")
    print("  which is a nonzero variety, so it cannot separate states there:\n")
    print(f"  {'q':<16}{'|N(q)|^2':>12}{'<q^dag q>_0':>16}")
    degen_ok = True
    for name, v in null_vecs:
        val = complex(sp.simplify(biquat_mul(v, bar(v))[0]))
        dval = sum(abs(complex(c)) ** 2 for c in v)
        degen_ok &= abs(abs(val) ** 2) < TOL_EXACT
        print(f"  {name:<16}{abs(val)**2:>12.1f}{dval:>16.1f}")
    print(f"\n  |N|^2 = 0 on nonzero vectors:                           {ok(degen_ok)}")
    print("  >> So the sharpened statement survives the escape untouched:")
    print("     POSITIVE DEFINITE XOR INVARIANT, at every degree. |N|^2 buys invariance and")
    print("     non-negativity by giving up definiteness -- it assigns probability 0 to a")
    print("     whole cone of distinct nonzero states.")

    # -----------------------------------------------------------------------
    banner("E4. DECISIVE: on the paper's own CENTRAL W = w*1, |N(W)|^2 = |w|^4, not |w|^2")
    print("  born_axiomatic.tex, abstract: 'the closed two-channel amplitude is the transition")
    print("  overlap W = <psi|psi_0> = g + i t_p', and the target closure is")
    print("  W^dag W = g^2 + t_p^2 = |<psi|psi_0>|^2. That W is a complex SCALAR: W = w*1,")
    print("  central, with vector part zero. So the escape is fully determined:\n")
    g, tp = sp.symbols('g t_p', real=True)
    wsym = g + sp.I * tp
    Wc = [wsym, 0, 0, 0]
    N_W = sp.simplify(biquat_mul(Wc, bar(Wc))[0])
    D_W = sp.simplify(biquat_mul(dag(Wc), Wc)[0])
    modN_W = sp.simplify(sp.expand(N_W * sp.conjugate(N_W)))
    born_W = sp.simplify(sp.expand(D_W))
    print(f"    N(W)   = det W  = {N_W}                 (= w^2)")
    print(f"    |N(W)|^2        = {modN_W}     (= |w|^4)")
    print(f"    <W^dag W>_0     = {born_W}         (= |w|^2)  <- the Born target")

    is_w4 = sp.simplify(modN_W - (g ** 2 + tp ** 2) ** 2) == 0
    is_w2 = sp.simplify(born_W - (g ** 2 + tp ** 2)) == 0
    print(f"\n  |N(W)|^2 == (g^2 + t_p^2)^2 = |w|^4:                     {ok(is_w4)}")
    print(f"  <W^dag W>_0 == g^2 + t_p^2 = |w|^2:                     {ok(is_w2)}")

    m = sp.Symbol('m', nonnegative=True)     # m = |w|
    sols = sp.solve(sp.Eq(m ** 4, m ** 2), m)
    sols = sorted([s for s in sols if s.is_real and s >= 0])
    print(f"\n  Solving |w|^4 == |w|^2 for |w| >= 0:  |w| in {sols}")
    agree_ok = (sols == [0, 1])
    print(f"  the two forms agree ONLY at |w| in {{0, 1}}:              {ok(agree_ok)}")
    print(f"\n  {'|w|':>8}{'|N(W)|^2 = |w|^4':>20}{'Born = |w|^2':>16}{'agree?':>9}")
    for mv in (0.0, 0.5, 1.0, 1.5, 2.0):
        print(f"  {mv:>8.2f}{mv**4:>20.6f}{mv**2:>16.6f}"
              f"{str(np.isclose(mv**4, mv**2)):>9}")
    print("\n  >> The escape does not return the Born rule on the paper's own amplitude. It")
    print("     returns its SQUARE. It coincides at |w| = 1 -- which is precisely the")
    print("     normalized case the derivation quietly assumes, so the disagreement is")
    print("     invisible exactly where it is being checked and appears everywhere else.")

    # -----------------------------------------------------------------------
    banner("N1. NEGATIVE: probabilities must sum to 1. The quartic rule does not.")
    print("  Parseval/completeness: for an orthonormal basis {|k>} and normalized |psi>, the")
    print("  amplitudes c_k = <k|psi> satisfy sum_k |c_k|^2 = 1. Always. The 'both conjugates'")
    print("  rule would instead report sum_k |c_k^2|^2 = sum_k |c_k|^4.\n")
    print(f"  {'dim':>5}{'sum |c_k|^2 (Born)':>22}{'sum |c_k|^4 (|N|^2)':>22}{'valid?':>9}")
    born_all_one = True
    quart_any_one = False
    quart_means = {}
    for d in (2, 3, 5, 10):
        psi = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi /= np.linalg.norm(psi)
        b = np.sum(np.abs(psi) ** 2)
        qv = np.sum(np.abs(psi) ** 4)
        born_all_one &= bool(np.isclose(b, 1.0, atol=1e-12))
        quart_any_one |= bool(np.isclose(qv, 1.0, atol=1e-6))
        print(f"  {d:>5}{b:>22.12f}{qv:>22.12f}{str(np.isclose(qv, 1)):>9}")
    print(f"\n  Born sums to 1 in every dimension:                       {ok(born_all_one)}")
    print(f"  quartic rule never sums to 1:                           {ok(not quart_any_one)}")

    print("\n  Averaged over 2000 random normalized states per dimension, the deficit grows:")
    print(f"  {'dim':>5}{'mean sum |c_k|^2':>20}{'mean sum |c_k|^4':>20}{'mean deficit':>15}")
    for d in (2, 3, 5, 10):
        psis = rng.normal(size=(2000, d)) + 1j * rng.normal(size=(2000, d))
        psis /= np.linalg.norm(psis, axis=1, keepdims=True)
        bm = np.mean(np.sum(np.abs(psis) ** 2, axis=1))
        qm = np.mean(np.sum(np.abs(psis) ** 4, axis=1))
        quart_means[d] = qm
        print(f"  {d:>5}{bm:>20.12f}{qm:>20.6f}{1 - qm:>15.6f}")
    degrades = all(quart_means[a] > quart_means[b]
                   for a, b in zip((2, 3, 5), (3, 5, 10)))
    print(f"\n  deficit strictly worsens with dimension (2 > 3 > 5 > 10):{ok(degrades)}")
    print("  >> It is not a probability distribution at all. It does not normalize, and the")
    print("     failure gets worse the larger the system.")

    # -----------------------------------------------------------------------
    banner("N2. NEGATIVE: spin-1/2 at angle theta -- half the probability vanishes")
    th = sp.Symbol('theta', real=True)
    p_sum = sp.simplify(sp.cos(th / 2) ** 2 + sp.sin(th / 2) ** 2)
    q_sum = sp.simplify(sp.cos(th / 2) ** 4 + sp.sin(th / 2) ** 4)
    target = 1 - sp.Rational(1, 2) * sp.sin(th) ** 2
    identity_ok = sp.simplify(q_sum - target) == 0
    print(f"  Born:     P(up) = cos^2(theta/2), P(down) = sin^2(theta/2)")
    print(f"            sum   = {p_sum}                     <- exactly 1, all theta")
    print(f"  Quartic:  P(up) = cos^4(theta/2), P(down) = sin^4(theta/2)")
    print(f"            sum   = {q_sum}")
    print(f"\n  quartic sum == 1 - (1/2) sin^2(theta), verified symbolically: {ok(identity_ok)}")
    roots = sp.solve(sp.Eq(target, 1), th)
    print(f"  solving 1 - (1/2) sin^2(theta) == 1 over theta:          roots {roots}")
    print(f"  i.e. equal to 1 only at theta = 0, pi:                   "
          f"{ok(sp.simplify(target.subs(th, 0) - 1) == 0 and sp.simplify(target.subs(th, sp.pi) - 1) == 0)}")
    print(f"\n  {'theta':>10}{'Born sum':>12}{'quartic sum':>14}")
    half_at_pi2 = False
    for tv in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi):
        b = np.cos(tv / 2) ** 2 + np.sin(tv / 2) ** 2
        qv = np.cos(tv / 2) ** 4 + np.sin(tv / 2) ** 4
        if np.isclose(tv, np.pi / 2):
            half_at_pi2 = bool(np.isclose(qv, 0.5))
        print(f"  {tv:>10.4f}{b:>12.6f}{qv:>14.6f}")
    print(f"\n  at theta = pi/2 the two outcomes total 0.5:              {ok(half_at_pi2)}")
    print("  >> Half the probability has vanished. And the deficit depends on theta, so no")
    print("     single divisor repairs it across states.")

    # -----------------------------------------------------------------------
    banner("N3. NEGATIVE: the repair divisor is STATE-DEPENDENT -- no constant fixes it")
    print(f"  {'state':>18}{'sum |c|^4':>12}{'needed divisor':>17}")
    states = [("(1,0)", np.array([1, 0], dtype=complex)),
              ("(1,1)/sqrt2", np.array([1, 1], dtype=complex) / np.sqrt(2)),
              ("(2,1)/sqrt5", np.array([2, 1], dtype=complex) / np.sqrt(5)),
              ("(1,1,1)/sqrt3", np.array([1, 1, 1], dtype=complex) / np.sqrt(3))]
    divisors = []
    for label, psi in states:
        s = np.sum(np.abs(psi) ** 4)
        divisors.append(1 / s)
        print(f"  {label:>18}{s:>12.6f}{1/s:>17.6f}")
    expected = [1.0, 2.0, 3.0]
    got = [divisors[0], divisors[1], divisors[3]]
    div_ok = all(np.isclose(a, b) for a, b in zip(got, expected))
    print(f"\n  divisors for (1,0), (1,1)/sqrt2, (1,1,1)/sqrt3 are 1.0, 2.0, 3.0: {ok(div_ok)}")
    print(f"  divisor is not constant across states:                   "
          f"{ok(len(set(np.round(divisors, 6))) > 1)}")
    print("\n  >> There is no algebraic constant that fixes it. Restoring normalization means")
    print("     dividing by sum_k |c_k|^4 -- re-introducing by hand exactly the external sum")
    print("     over outcomes that Thm 6.1 boasts of not needing: 'Normalization closes")
    print("     through the algebra's own conjugation; no Gleason-type premise, external")
    print("     partition integral, or auxiliary Hilbert-space postulate is invoked.'")

    # -----------------------------------------------------------------------
    banner("D1. DOCUMENTATION DEFECT: born_axiomatic.tex says (2,2); the carrier is (4,4)")
    print("  born_axiomatic.tex says, in three places:")
    print("    line  38 (abstract):  '... selects the Lorentzian (1,3) sector from the")
    print("                           algebra's split (2,2) norm'")
    print("    line  54 (scope):     'The Lorentzian signature selection (2,2) -> (1,3)'")
    print("    line 139 (Lemma 4.2): 'maps the split-signature (2,2) carrier onto a real")
    print("                           four-dimensional slice of Lorentzian signature (1,3)'")
    print("\n  The carrier is C (x) H: 8 real dimensions. Its norm form is Re N(q). Taking the")
    print("  Hessian of the symbolic Re N directly, rather than asserting a matrix:\n")
    H = sp.hessian(ReN, xy) / 2
    Hn = np.array(H).astype(np.float64)
    ev = np.linalg.eigvalsh(Hn)
    npos = int((ev > 1e-9).sum())
    nneg = int((ev < -1e-9).sum())
    print(f"  Re N(q) = {ReN}")
    print(f"  Hessian/2 in coords (x_0..x_3, y_0..y_3) is diagonal:    "
          f"{ok(np.allclose(Hn, np.diag(np.diag(Hn))))}")
    print(f"  diagonal: {np.diag(Hn)}")
    print(f"  eigenvalues: {ev}")
    print(f"  signature: ({npos}, {nneg})")
    sig_ok = (npos == 4 and nneg == 4)
    print(f"\n  the full carrier's norm form is (4,4), NOT (2,2):        {ok(sig_ok)}")

    print("\n  (2,2) is what appears on particular real 4-dim SLICES. The real span of")
    print("  {1, i_q, i*j_q, i*k_q}, i.e. q = (x_0, x_1, i x_2, i x_3):")
    sl = [x[0], x[1], sp.I * x[2], sp.I * x[3]]
    Nsl = sp.simplify(biquat_mul(sl, bar(sl))[0])
    Hsl = np.array(sp.hessian(Nsl, x)).astype(np.float64) / 2
    evsl = np.linalg.eigvalsh(Hsl)
    sl_ok = (sp.simplify(Nsl - (x[0] ** 2 + x[1] ** 2 - x[2] ** 2 - x[3] ** 2)) == 0
             and int((evsl > 1e-9).sum()) == 2 and int((evsl < -1e-9).sum()) == 2)
    print(f"    N = {Nsl}")
    print(f"    eigenvalues {evsl}  -> signature "
          f"({int((evsl > 1e-9).sum())}, {int((evsl < -1e-9).sum())}):  {ok(sl_ok)}")

    print("\n  And the slice Lemma 4.2 actually names, basis {1, h i, h j, h k} with h the")
    print("  split unit (h^2 = +1, so each h i_n squares to -1), is a DIFFERENT slice:")
    t_, a1, a2, a3 = sp.symbols('t a_1 a_2 a_3', real=True)
    herm = [t_, sp.I * a1, sp.I * a2, sp.I * a3]
    Nh = sp.simplify(biquat_mul(herm, bar(herm))[0])
    Hh = np.array(sp.hessian(Nh, [t_, a1, a2, a3])).astype(np.float64) / 2
    evh = np.linalg.eigvalsh(Hh)
    h_ok = (int((evh > 1e-9).sum()) == 1 and int((evh < -1e-9).sum()) == 3)
    print(f"    N = {Nh}")
    print(f"    eigenvalues {evh}  -> signature "
          f"({int((evh > 1e-9).sum())}, {int((evh < -1e-9).sum())}):  {ok(h_ok)}")
    print("\n  >> So THREE different signatures are in play and the paper calls two of them")
    print("     by one name: the 8-dim carrier is (4,4), the {1, i_q, i*j_q, i*k_q} slice is")
    print("     (2,2), and the Hermitian slice Lemma 4.2 maps ONTO is (1,3). The (1,3)")
    print("     conclusion is correct and unaffected. The defect is the label on the SOURCE")
    print("     of the projection: 'the algebra's split (2,2) norm' names the 8-dim carrier,")
    print("     which is (4,4). Recommend: say (4,4) in the abstract and Lemma 4.2, or say")
    print("     explicitly that (2,2) refers to a chosen 4-dim slice and name which.")

    # -----------------------------------------------------------------------
    banner("SUMMARY")
    print(f"  {'form':<24}{'positive def':>14}{'sums to 1':>11}{'SL(2,C) inv':>13}"
          f"{'degree':>8}   verdict")
    rows = [
        ("<W^dag W>_0 = |w|^2", "yes", "yes", "NO", "2", "the Born rule"),
        ("|N(W)|^2    = |w|^4", "NO (degen)", "NO", "yes", "4", "not a probability"),
        ("N(W)        = w^2", "NO", "n/a", "yes", "2", "complex, indefinite"),
    ]
    for r in rows:
        print(f"  {r[0]:<24}{r[1]:>14}{r[2]:>11}{r[3]:>13}{r[4]:>8}   {r[5]}")

    quadratic_xor = (isotropy_ok and det_inv and born_not_inv and rot_inv
                     and cal_N_inv and not cal_b_inv)
    escape_priced = degrees_ok and degen_ok and is_w4 and is_w2 and agree_ok
    normalization = born_all_one and (not quart_any_one) and degrades and identity_ok \
        and half_at_pi2 and div_ok
    print(f"\n  Q1-Q5  POSITIVE XOR INVARIANT holds for quadratic forms:  {ok(quadratic_xor)}")
    print("     N is bilinear, invariant, indefinite. The dagger form is sesquilinear,")
    print("     positive definite, and drifts under boosts. Isotropy over C forbids any")
    print("     complex-bilinear form on C (x) H from being positive definite.")
    print(f"  E1-E4  the |N|^2 escape is real but priced out of the claim: {ok(escape_priced)}")
    print("     It is non-negative and invariant, so it beats the literal XOR. But it is")
    print("     QUARTIC (degree 4 vs 2), DEGENERATE on the null cone (so positive-DEFINITE")
    print("     XOR invariant still holds), and on the paper's own central W = w*1 it returns")
    print("     |w|^4 instead of |w|^2, agreeing only at |w| in {0, 1}.")
    print(f"  N1-N3  and it is not a probability at all:                {ok(normalization)}")
    print("     sum_k |c_k|^4 does not sum to 1 in any dimension, the deficit worsens with")
    print("     dimension, spin-1/2 loses half its probability at theta = pi/2, and the")
    print("     repair divisor is state-dependent (1.0, 2.0, 3.0), so the fix is the external")
    print("     sum over outcomes the theorem claims to have removed.")
    print(f"  D1     documentation defect confirmed, (4,4) not (2,2):   {ok(sig_ok and sl_ok)}")
    print("     born_axiomatic.tex lines 38, 54, 139. The (1,3) result is unaffected.")


if __name__ == "__main__":
    main()
