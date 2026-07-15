"""
Why the holomorphic dual exp maps are parameterised into QUATERNION CHANNELS.

Short answer: a complex scalar has no null cone, and the null cone is where J^2 = 2J
lives. The exp maps carry the dynamics; the quaternion direction carries the nullity.

The object under test (paper Eq. 1 shape, one exp map per channel pair):

    J_mu = A_mu g + B_mu s,    g = e^{-H/T},  s = e^{iS/hbar},   mu = 0..3,  J_mu in C
    det J = SUM_mu J_mu^2 = N(A) g^2 + 2<A,B> g s + N(B) s^2
    with N(A) = SUM_mu A_mu^2  and  <A,B> = SUM_mu A_mu B_mu    (BILINEAR, not sesquilinear)

Sections:
  0. HANDEDNESS GUARD          i*j = k and ijk = -1, on both the numpy mirror and cal.
  1. ONE vs TWO vs FOUR        a SCALAR current can never reach the null cone.
  2. MINIMAL NULL ELEMENT      J = 1 + h*i_q = (1, i, 0, 0). Lemma 4.1(c) of born_axiomatic.
  3. SYMPY IDENTITY            det J = N(A) g^2 + 2<A,B> g s + N(B) s^2, residual exactly 0.
  4. IDENTICAL VANISHING       det J == 0 for ALL g,s  <=>  N(A) = 0, <A,B> = 0, N(B) = 0.
  5. THE HEADLINE              H and S never enter the null condition.
  6. CROSS-CHECK vs cal        float32-limited tolerance, stated honestly.
  7. THE READING

PRECISION NOTE. cal is torch.complex64 (eps ~ 1e-7). The null-cone checks are CANCELLING
differences (det J = SUM J_mu^2 = 0 by cancellation), so complex64 CANNOT resolve them to
1e-15. All FINDINGS below are computed in numpy complex128 using cal's own representation
formula, written out explicitly. Section 6 then calls the real cal.biquaternion functions
on the same inputs and confirms agreement at a tolerance that is honest for complex64.
No 1e-15 identity is ever asserted through complex64.

Run:  python exp_null_cone_needs_channels.py     (from the "py tests" directory)
"""
import os
import sys

import numpy as np
import sympy as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cal.biquaternion import (quat_mul, hermitian_conj, quat_norm_sq,  # noqa: E402
                              biquat_to_matrix, CDTYPE)

RNG = np.random.default_rng(31)
RULE = "=" * 78
RESULTS = []


def ok(b):
    return "PASS" if b else "**FAIL**"


def record(name, passed):
    RESULTS.append((name, bool(passed)))
    return bool(passed)


# -- numpy complex128 mirror of cal's algebra -------------------------------
# quat_mul: cal/biquaternion.py Eq. 3, transcribed exactly.
def qmul(P, Q):
    p0, p1, p2, p3 = P
    q0, q1, q2, q3 = Q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], dtype=np.complex128)


# biquat_to_matrix: M(q) = [[q0 + h q3, q1 + h q2], [-q1 + h q2, q0 - h q3]], h = 1j.
# Gives tr M(q) = 2 q0 and det M(q) = q0^2+q1^2+q2^2+q3^2 = quat_norm_sq(q). Three lines:
def bqmat(q):
    h = 1j
    q0, q1, q2, q3 = q
    return np.array([[q0 + h*q3, q1 + h*q2],
                     [-q1 + h*q2, q0 - h*q3]], dtype=np.complex128)


def Nq(c):
    """Reduced norm = det M(c) = SUM_mu c_mu^2. BILINEAR."""
    return np.sum(np.asarray(c, dtype=np.complex128)**2)


def bilin(a, b):
    """<a,b> = SUM_mu a_mu b_mu. BILINEAR, no conjugation."""
    return np.sum(np.asarray(a, dtype=np.complex128) * np.asarray(b, dtype=np.complex128))


E0 = np.array([1, 0, 0, 0], dtype=np.complex128)
E1 = np.array([0, 1, 0, 0], dtype=np.complex128)
E2 = np.array([0, 0, 1, 0], dtype=np.complex128)
E3 = np.array([0, 0, 0, 1], dtype=np.complex128)


# ===========================================================================
def sec0_handedness():
    print(RULE)
    print("0. HANDEDNESS GUARD -- the basis used below is RIGHT-HANDED")
    print(RULE)

    ij = qmul(E1, E2)
    ijk = qmul(qmul(E1, E2), E3)
    np_ij = np.allclose(ij, E3, atol=1e-15)
    np_ijk = np.allclose(ijk, -E0, atol=1e-15)
    print(f"  numpy mirror   i*j = {np.real_if_close(ij)}   == k ?  {ok(np_ij)}")
    print(f"  numpy mirror   ijk = {np.real_if_close(ijk)}   == -1 ? {ok(np_ijk)}")

    t1 = torch.tensor([0, 1, 0, 0], dtype=CDTYPE)
    t2 = torch.tensor([0, 0, 1, 0], dtype=CDTYPE)
    t3 = torch.tensor([0, 0, 0, 1], dtype=CDTYPE)
    tk = torch.tensor([0, 0, 0, 1], dtype=CDTYPE)
    tm1 = torch.tensor([-1, 0, 0, 0], dtype=CDTYPE)
    c_ij = bool(torch.allclose(quat_mul(t1, t2), tk, atol=1e-6))
    c_ijk = bool(torch.allclose(quat_mul(quat_mul(t1, t2), t3), tm1, atol=1e-6))
    print(f"  cal.quat_mul   i*j == k ?  {ok(c_ij)}        ijk == -1 ? {ok(c_ijk)}")

    passed = np_ij and np_ijk and c_ij and c_ijk
    print(f"\n  SECTION 0: {ok(passed)}  (mirror and cal agree, both right-handed)")
    return record("0. handedness guard", passed)


# ===========================================================================
def sec1_channel_count():
    print("\n" + RULE)
    print("1. ONE vs TWO vs FOUR CHANNELS -- the null cone appears exactly at TWO")
    print(RULE)
    print("  det J = SUM_mu J_mu^2 over the first n components. Solve det J = 0, J != 0.\n")

    J0, J1, J2, J3 = sp.symbols('J0 J1 J2 J3')

    # n = 1: J0^2 = 0 has the single root J0 = 0. Sympy-verified, not asserted.
    roots1 = sp.solve(sp.Eq(J0**2, 0), J0)
    n1_only_zero = (roots1 == [0])

    # n = 2: J0^2 + J1^2 = 0  <=>  J1 = +- i J0. Sympy-verified.
    roots2 = sp.solve(sp.Eq(J0**2 + J1**2, 0), J1)
    n2_pm_i = set(roots2) == {sp.I*J0, -sp.I*J0}

    # Dimension of the null set. det is a nondegenerate quadratic; grad(det) = 2 J,
    # which is nonzero off the origin, so 0 is a regular value and {det = 0} \ {0} is
    # a smooth complex hypersurface of dimension n - 1. Verified numerically below.
    rows = [
        (1, "J_0^2", False, roots1, 0, "vanishes only at J_0 = 0"),
        (2, "J_0^2 + J_1^2", True, roots2, 1, "J_1 = +- i J_0"),
        (4, "J_0^2 + ... + J_3^2", True, None, 3, "a 3-dim family"),
    ]
    print(f"  {'components':>11}{'det J':>22}{'nonzero null?':>15}"
          f"{'dim null set':>14}{'why':>27}")
    for n, expr, has_null, _r, dim, why in rows:
        print(f"  {n:>11}{expr:>22}{str(has_null):>15}{dim:>14}{why:>27}")

    print(f"\n  sympy: solve(J_0^2 = 0, J_0)          -> {roots1}"
          f"        only zero? {ok(n1_only_zero)}")
    print(f"  sympy: solve(J_0^2 + J_1^2 = 0, J_1)  -> {roots2}   J_1 = +-i J_0? "
          f"{ok(n2_pm_i)}")

    # Regular-value check: sample nonzero null vectors at n = 2 and n = 4, confirm
    # det = 0 to complex128 precision and grad(det) = 2J != 0 there.
    print("\n  regular-value check on sampled NONZERO null vectors (complex128):")
    reg_ok = True
    for n, dim in ((2, 1), (4, 3)):
        worst_det = 0.0
        worst_grad = np.inf
        for _ in range(200):
            if n == 2:
                z = RNG.normal() + 1j*RNG.normal()
                v = np.array([z, 1j*z, 0, 0], dtype=np.complex128)
            else:
                d = RNG.normal(size=3)
                d = d / np.linalg.norm(d)
                z = RNG.normal() + 1j*RNG.normal()
                v = z * np.array([1.0, 1j*d[0], 1j*d[1], 1j*d[2]], dtype=np.complex128)
            worst_det = max(worst_det, abs(Nq(v)))
            worst_grad = min(worst_grad, np.linalg.norm(2*v))
        good = (worst_det < 1e-12) and (worst_grad > 1e-6)
        reg_ok = reg_ok and good
        print(f"    n = {n}: max|det J| = {worst_det:.2e}   min||grad det|| = "
              f"{worst_grad:.2e} (spot-check only)"
              f"   dim = n - 1 = {dim}   {ok(good)}")

    # n = 1 sampled: a scalar current is null only when it IS zero.
    worst_scalar = min(abs(Nq(np.array([RNG.normal() + 1j*RNG.normal(), 0, 0, 0])))
                       for _ in range(200))
    scal_never = worst_scalar > 1e-6
    print(f"    n = 1: min|det J| over 200 random NONZERO scalars = {worst_scalar:.2e} "
          f"(never null)  {ok(scal_never)}")

    # What actually carries "dim = n - 1", stated plainly so the table is not mistaken
    # for the evidence. The numbers above are a spot-check; the ARGUMENT is the proof.
    print("\n  WHAT CARRIES 'dim = n - 1' (the sampling does NOT, and is not claimed to):")
    print("    grad(det) = 2J, which is nonzero for ANY J != 0 -- no sampling needed. So 0")
    print("    is a regular value of det on C^n \\ {0}, and {det = 0} \\ {0} is a smooth")
    print("    complex hypersurface of dim n - 1. That is the regular-value theorem.")
    print("    Caveat kept honest: the n = 4 sampling above draws v = z*(1, i*d) with d a")
    print("    REAL unit 3-vector, spanning only 2 COMPLEX dims of the 3-COMPLEX-dim cone.")
    print("    It is a subset, so it corroborates nullity but cannot exhibit the full")
    print("    3-dim family. Section 5 uses a generic COMPLEX null direction off this slice.")

    passed = n1_only_zero and n2_pm_i and reg_ok and scal_never
    print(f"\n  SECTION 1: {ok(passed)}")
    print("  >> TWO complex components is the minimum. One is never null. So the moment")
    print("     you want a source current that can sit on its own light cone, you cannot")
    print("     use a scalar. That is not a modelling preference, it is arithmetic.")
    return record("1. one vs two vs four channels", passed)


# ===========================================================================
def sec2_minimal_null():
    print("\n" + RULE)
    print("2. THE MINIMAL NULL ELEMENT -- Lemma 4.1(c) of born_axiomatic.tex, exactly")
    print(RULE)

    a = np.array([1.0, 1j, 0.0, 0.0], dtype=np.complex128)
    M = bqmat(a)
    det = Nq(a)
    det_m = M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]
    rank = np.linalg.matrix_rank(M, tol=1e-10)
    nonzero = not np.allclose(M, 0, atol=1e-12)

    comps = ", ".join(f"{c:.0f}" for c in a)
    print(f"    J = 1 + h*i_q, components (q0,q1,q2,q3) = ({comps})")
    print(f"    M(J) = [[{M[0,0]:.0f}, {M[0,1]:.0f}], [{M[1,0]:.0f}, {M[1,1]:.0f}]]")
    print(f"    det J = SUM J_mu^2 = 1^2 + i^2 = {det.real:.1f}      "
          f"det M(J) = {det_m.real:.1f}   (agree: {ok(abs(det - det_m) < 1e-14)})")
    print(f"    J != 0 ? {nonzero}      rank M(J) = {rank}")

    d_ok = abs(det) < 1e-14
    r_ok = (rank == 1)
    print(f"\n    det = 0:  {ok(d_ok)}      rank 1: {ok(r_ok)}      J != 0: {ok(nonzero)}")

    passed = d_ok and r_ok and nonzero
    print(f"\n  SECTION 2: {ok(passed)}  (a nonzero rank-1 null element exists at n = 2)")
    return record("2. minimal null element (Lemma 4.1(c))", passed)


# ===========================================================================
def sec3_sympy_quadratic():
    print("\n" + RULE)
    print("3. det J IS A QUADRATIC IN THE TWO EXP MAPS (g, s) -- sympy residual")
    print(RULE)

    g, s = sp.symbols('g s')
    A = sp.symbols('A0 A1 A2 A3')
    B = sp.symbols('B0 B1 B2 B3')
    J = [A[m]*g + B[m]*s for m in range(4)]
    det = sp.expand(sum(j**2 for j in J))

    NA = sum(A[m]**2 for m in range(4))
    AB = sum(A[m]*B[m] for m in range(4))       # BILINEAR
    NB = sum(B[m]**2 for m in range(4))
    residual = sp.expand(det - (NA*g**2 + 2*AB*g*s + NB*s**2))

    print(f"  det J = {sp.collect(det, [g, s])}\n")
    print("  claim:  det J = N(A) g^2 + 2 <A,B> g s + N(B) s^2")
    print(f"  residual det J - [N(A) g^2 + 2 <A,B> g s + N(B) s^2] = {residual}")

    exact_zero = (residual == 0)
    print(f"  residual is EXACTLY 0 (symbolic, not numeric): {ok(exact_zero)}")

    print(f"\n  SECTION 3: {ok(exact_zero)}")
    print("  >> The null condition is a QUADRATIC FORM in the two exp maps, with")
    print("     coefficients built from the amplitude vectors ALONE. The exp maps never")
    print("     had to be logs, and the condition never needed a branch.")
    return record("3. sympy quadratic identity", exact_zero)


# ===========================================================================
def null_vec(rng):
    """N(c) = 0 with c != 0:  c = (1, i n), n a real unit 3-vector."""
    n = rng.normal(size=3)
    n = n / np.linalg.norm(n)
    return np.array([1.0, 1j*n[0], 1j*n[1], 1j*n[2]], dtype=np.complex128)


def shared_dir_pair(rng):
    """A = B = (1, i n): both null AND pointing the SAME quaternionic direction."""
    v = null_vec(rng)
    return v, v.copy()


def sec4_identical_vanishing():
    print("\n" + RULE)
    print("4. det J == 0 for ALL g,s  <=>  N(A) = 0, <A,B> = 0, N(B) = 0")
    print(RULE)
    print("  Three conditions. Achieve all three by making A and B both null AND")
    print("  pointing the SAME quaternionic direction: A = B = (1, i n), |n| = 1 real.")
    print("  Then <A,B> = N(A) = 0 comes for FREE.\n")

    Ar = RNG.normal(size=4).astype(np.complex128)
    Br = RNG.normal(size=4).astype(np.complex128)
    An, Bn = null_vec(RNG), null_vec(RNG)       # both null, generic (different) dirs
    Ap, Bp = shared_dir_pair(RNG)               # both null, SAME dir

    cases = [
        ("A, B generic real", Ar, Br, False),
        ("A, B both null, generic dirs", An, Bn, False),
        ("A, B null, SAME direction", Ap, Bp, True),
    ]

    print(f"  {'construction':>30}{'|N(A)|':>11}{'|<A,B>|':>11}{'|N(B)|':>11}"
          f"{'max|det J| over 20 (g,s)':>26}")
    passed = True
    for name, Av, Bv, expect_null in cases:
        na, ab, nb = Nq(Av), bilin(Av, Bv), Nq(Bv)
        worst = 0.0
        for _ in range(20):
            # free COMPLEX (g,s): the strongest reading of "for ALL g,s".
            gv = RNG.normal() + 1j*RNG.normal()
            sv = RNG.normal() + 1j*RNG.normal()
            worst = max(worst, abs(Nq(Av*gv + Bv*sv)))
        is_null = worst < 1e-12
        agree = (is_null == expect_null)
        passed = passed and agree
        print(f"  {name:>30}{abs(na):>11.2e}{abs(ab):>11.2e}{abs(nb):>11.2e}"
              f"{worst:>18.2e} {ok(agree):>7}")

    print(f"\n  <A,B> for the SAME-direction pair = {abs(bilin(Ap, Bp)):.2e}  (free, "
          "not imposed)")
    print("  Only the third construction gives ~1e-15. The first two do NOT, and that is")
    print("  reported as a negative, not smoothed over: two null vectors with DIFFERENT")
    print("  directions are not bilinearly orthogonal, so the cross term survives.")

    print(f"\n  SECTION 4: {ok(passed)}")
    return record("4. identical vanishing iff all three conditions", passed)


# ===========================================================================
def sec5_headline():
    print("\n" + RULE)
    print("5. THE HEADLINE -- H and S NEVER ENTER THE NULL CONDITION")
    print(RULE)
    print("  def:J-closed: 'At closure the two coincide, theta_R = theta_I =: phi(x,t)'.")
    print("  The channels sharing ONE quaternionic direction IS what 'the two coincide'")
    print("  says algebraically, and it is exactly what makes det J vanish identically.")
    print("  Closure IS the null cone.\n")

    Ap, Bp = shared_dir_pair(RNG)
    MA = bqmat(Ap)                       # the direction matrix, (H,S)-independent
    T, HBAR = 1.0, 1.0

    sweep = [(0.2, 0.0), (0.7, 0.0), (0.5, 1.1), (1.3, np.pi/2),
             (0.05, np.pi), (2.0, np.pi), (0.0, np.pi)]

    print(f"  {'H':>6}{'S':>10}{'g=e^{-H/T}':>13}{'s=e^{iS/hbar}':>20}"
          f"{'|det J|':>11}{'tr Jn':>9}{'|Jn^2-2Jn|':>13}{'|Jn-M(A)|':>12}")
    det_ok = True
    idem_ok = True
    degen_rows = 0
    degen_gs = None
    for (H, S) in sweep:
        gv = np.exp(-H/T)
        sv = np.exp(1j*S/HBAR)
        Jc = Ap*gv + Bp*sv
        Jm = bqmat(Jc)
        detJ = abs(Nq(Jc))
        det_ok = det_ok and (detJ < 1e-12)

        tr = np.trace(Jm)
        if abs(tr) < 1e-12:
            # A = B and g ~ -s: J = (g+s)A ~ 0, so the trace normalisation is undefined.
            # NOT exact: np.exp(1j*pi) = -1 + 1.22e-16j, so g + s = 1.22e-16j, not 0.
            # J is zero to MACHINE PRECISION only. Reported, not hidden, not counted.
            degen_rows += 1
            degen_gs = abs(gv + sv)
            print(f"  {H:>6.2f}{S:>10.4f}{gv:>13.6f}{str(np.round(sv, 4)):>20}"
                  f"{detJ:>11.2e}{'0.0':>9}{'DEGEN':>13}{'DEGEN':>12}")
            continue

        Jn = 2*Jm/tr                       # normalise to tr = 2
        idem = np.abs(Jn @ Jn - 2*Jn).max()
        same = np.abs(Jn - MA).max()
        idem_ok = idem_ok and (idem < 1e-10)
        print(f"  {H:>6.2f}{S:>10.4f}{gv:>13.6f}{str(np.round(sv, 4)):>20}"
              f"{detJ:>11.2e}{np.trace(Jn).real:>9.4f}{idem:>13.2e}{same:>12.2e}")

    print(f"\n  det J = 0 at EVERY swept point:                 {ok(det_ok)}")
    print("     (threshold is ABSOLUTE 1e-12 on |det J|, which scales as |J|^2. Honest for")
    print("      this sweep, where H in [0,2] gives g in [0.135,1] and |J| ~ O(1). It would")
    print("      NOT be scale-free for large negative H; that regime is not claimed here.)")
    print(f"  Jn^2 = 2 Jn at every non-degenerate point:     {ok(idem_ok)}")
    print(f"  degenerate rows (J ~ 0, g ~ -s):               {degen_rows} "
          "(H = 0, S = pi.)")
    if degen_gs is not None:
        print(f"     NOT exact: np.exp(1j*pi) = -1 + 1.22e-16j, so |g + s| = {degen_gs:.2e}, not 0.")
        print("     J is zero to MACHINE PRECISION, not identically. det J = 0 there too, but")
        print("     trivially, and the tr-normalisation is undefined. Stated, not hidden.")

    # The sharper statement: with A = B, J = (g+s)A, so the whole (H,S) dependence is a
    # single complex scalar prefactor and the normalised current is literally M(A).
    print("\n  Sharper: with A = B the current is J = (g + s) A. All of (H,S) sits in ONE")
    print("  complex prefactor, so after normalising to tr = 2 the current IS M(A) at")
    print("  every point -- the |Jn - M(A)| column above is 0 to complex128 precision.")

    # Do not generalise from the special case: n real puts M(A) on the HERMITIAN branch,
    # where Jn/2 is an ORTHOGONAL projection. A generic COMPLEX null direction still
    # gives det = 0 and Jn^2 = 2 Jn, but Jn is NOT Hermitian, so Jn/2 is idempotent
    # only -- NOT an orthogonal projection. Check both, do not overclaim.
    herm_real = np.abs(MA - MA.conj().T).max()
    d = RNG.normal(size=3) + 1j*RNG.normal(size=3)
    d = d / np.sqrt(np.sum(d**2))                  # complex dir with SUM d_k^2 = 1
    Ac = np.array([1.0, 1j*d[0], 1j*d[1], 1j*d[2]], dtype=np.complex128)
    Mc = bqmat(Ac)
    Mcn = 2*Mc/np.trace(Mc)
    idem_c = np.abs(Mcn @ Mcn - 2*Mcn).max()
    herm_c = np.abs(Mcn - Mcn.conj().T).max()
    print("\n  Not generalising from the special case (real n vs complex null direction):")
    print(f"    real n     : |M(A) - M(A)^dag| = {herm_real:.2e}  -> Hermitian, so Jn/2 "
          "IS an orthogonal projection")
    print(f"    complex dir: |Jn^2 - 2Jn| = {idem_c:.2e} (still null/idempotent) but")
    print(f"                 |Jn - Jn^dag| = {herm_c:.2e} -> NOT Hermitian, so Jn/2 is")
    print("                 idempotent ONLY, NOT an orthogonal projection.")
    branch_ok = (herm_real < 1e-12) and (idem_c < 1e-10) and (herm_c > 1e-6)
    print(f"    branch distinction holds as stated: {ok(branch_ok)}")

    passed = det_ok and idem_ok and branch_ok
    print(f"\n  SECTION 5: {ok(passed)}")
    print("  >> The exp maps are free to do whatever they like. H and S never enter the")
    print("     null condition. It is carried entirely by the DIRECTION the amplitude")
    print("     vectors share.")
    return record("5. headline: H and S never enter", passed)


# ===========================================================================
def sec6_crosscheck():
    print("\n" + RULE)
    print("6. CROSS-CHECK AGAINST THE REAL cal.biquaternion (torch.complex64)")
    print(RULE)
    print(f"  cal CDTYPE = {CDTYPE} -> eps ~ 1e-7. Tolerances here are FLOAT32-LIMITED")
    print("  (rtol ~ 1e-5). The complex128 findings above are NOT re-asserted at 1e-15")
    print("  through complex64; that would be dishonest about the arithmetic.\n")

    # Same inputs, both engines. GENERIC (non-cancelling) biquaternions first.
    P = RNG.normal(size=4) + 1j*RNG.normal(size=4)
    Q = RNG.normal(size=4) + 1j*RNG.normal(size=4)
    tP = torch.tensor(P, dtype=CDTYPE)
    tQ = torch.tensor(Q, dtype=CDTYPE)

    # CC1: quat_mul
    mine = qmul(P, Q)
    theirs = quat_mul(tP, tQ).numpy().astype(np.complex128)
    cc1 = np.allclose(mine, theirs, rtol=1e-5, atol=1e-5)
    print(f"  CC1  quat_mul   mirror vs cal:  max|diff| = {np.abs(mine-theirs).max():.2e}"
          f"   {ok(cc1)}   (rtol 1e-5, float32-limited)")

    # CC2: biquat_to_matrix
    mineM = bqmat(P)
    theirsM = biquat_to_matrix(tP).numpy().astype(np.complex128)
    cc2 = np.allclose(mineM, theirsM, rtol=1e-5, atol=1e-5)
    print(f"  CC2  biquat_to_matrix:          max|diff| = "
          f"{np.abs(mineM-theirsM).max():.2e}   {ok(cc2)}")

    # CC3: quat_norm_sq on a GENERIC (non-null) input -- no cancellation, so this
    # comparison is meaningful at rtol 1e-5.
    mineN = Nq(P)
    theirsN = complex(quat_norm_sq(tP).item())
    cc3 = np.isclose(mineN, theirsN, rtol=1e-5, atol=1e-5)
    print(f"  CC3  quat_norm_sq (generic P):  mirror = {mineN:.6f}")
    print(f"                                  cal    = {theirsN:.6f}   {ok(cc3)}")

    # CC4: quat_norm_sq on the NULL A. This is the cancelling case. complex128 gives
    # ~1e-16; complex64 CANNOT. We assert only float32-honest zero, and say so.
    Ap, _Bp = shared_dir_pair(RNG)
    tA = torch.tensor(Ap, dtype=CDTYPE)
    null128 = abs(Nq(Ap))
    null64 = abs(complex(quat_norm_sq(tA).item()))
    cc4 = null64 < 1e-5
    print(f"\n  CC4  quat_norm_sq on the NULL A = (1, i n)   [the cancelling case]")
    print(f"         complex128 mirror: |det| = {null128:.2e}   (true zero to ~1e-16)")
    print(f"         cal complex64:     |det| = {null64:.2e}   {ok(cc4)}")
    print("         Verdict: consistent with zero AT FLOAT32 PRECISION. This is NOT")
    print("         evidence of a 1e-15 identity and is not reported as one.")

    # CC5: the dagger survives the embedding, on the same input. M(q^dag) == M(q)^dag.
    dag = hermitian_conj(tP)
    lhs = biquat_to_matrix(dag).numpy().astype(np.complex128)
    rhs = biquat_to_matrix(tP).numpy().astype(np.complex128).conj().T
    cc5 = np.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)
    print(f"\n  CC5  M(q^dag) == M(q)^dag:      max|diff| = {np.abs(lhs-rhs).max():.2e}"
          f"   {ok(cc5)}")

    passed = cc1 and cc2 and cc3 and cc4 and cc5
    print(f"\n  SECTION 6: {ok(passed)}  (mirror reproduces cal within float32 limits)")
    return record("6. cross-check vs cal.biquaternion", passed)


# ===========================================================================
def sec7_reading():
    print("\n" + RULE)
    print("7. THE READING, PLAINLY")
    print(RULE)
    ans = [
        ("a scalar current is never null", "so J^2 = 2J has only {0, 2}, as in C"),
        ("two complex components suffice", "J_1 = +- i J_0 is the whole null cone at n=2"),
        ("four give a 3-dim null family", "and a DIRECTION for the channels to share"),
        ("det J is quadratic in (g,s)", "N(A) g^2 + 2<A,B> g s + N(B) s^2"),
        ("null A, B, same direction", "-> det J = 0 identically, for all g, s"),
        ("that is closure", "'the two coincide' = one shared direction"),
        ("H and S never enter", "the exp maps are not what makes it null"),
    ]
    print(f"  {'what the parameterisation buys':<32}  {'why':<42}")
    print(f"  {'-'*32}  {'-'*42}")
    for a_, b_ in ans:
        print(f"  {a_:<32}  {b_:<42}")
    print("\n  >> THE EXP MAPS CARRY THE DYNAMICS; THE QUATERNION DIRECTION CARRIES THE")
    print("     NULLITY. Separating those is what the parameterisation is for, and it is")
    print("     why a scalar W cannot do the job: it has nowhere to put the direction.")
    print("\n  >> Closure (def:J-closed, 'the two coincide') IS the shared-direction")
    print("     condition, hence IS the null cone. They are the same statement.")
    return True


# ===========================================================================
def main():
    print(RULE)
    print("exp_null_cone_needs_channels.py -- why the dual exp maps are parameterised")
    print("into quaternion channels. Findings in numpy complex128; cal cross-check at")
    print("float32-limited tolerance. Seed = 31.")
    print(RULE)

    sec0_handedness()
    sec1_channel_count()
    sec2_minimal_null()
    sec3_sympy_quadratic()
    sec4_identical_vanishing()
    sec5_headline()
    sec6_crosscheck()
    sec7_reading()

    print("\n" + RULE)
    print("SUMMARY")
    print(RULE)
    for name, passed in RESULTS:
        print(f"  {name:<48} {ok(passed)}")
    n_fail = sum(1 for _n, p in RESULTS if not p)
    print(f"\n  {len(RESULTS) - n_fail}/{len(RESULTS)} sections PASS, {n_fail} FAIL")
    if n_fail:
        print("  NEGATIVE RESULT: at least one section FAILED. Kept as a negative.")
    else:
        print("  All sections PASS. The finding stands as stated.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
