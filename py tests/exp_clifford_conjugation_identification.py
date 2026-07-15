"""
Are the paper's two hand-defined conjugations actually the canonical Clifford involutions?

Under the identification C (x) H == Cl(3,0), with the central imaginary h = sqrt(-1) mapped to
the pseudoscalar e1e2e3, the claim is:
    the paper's BAR    (cal.quat_conj)      IS Clifford conjugation
    the paper's DAGGER (cal.hermitian_conj) IS REVERSION
If so, <W^dag W>_0 is the standard geometric-algebra reversion norm, not a bespoke composite
of two unrelated involutions invented for the paper.

CLAIMS VERIFIED (against cal.biquaternion, the package under test):
  A1  Grade sign patterns on Cl(3,0), grades k = 0,1,2,3:
        reversion        (-1)^(k(k-1)/2) = (+,+,-,-)
        Clifford conj    (-1)^(k(k+1)/2) = (+,-,-,+)
        grade involution (-1)^k          = (+,-,+,-)
      and reversion composed with grade involution == Clifford conjugation.
  A2  The algebra map is the claimed one: the central imaginary h = (h,0,0,0) embeds as the
      pseudoscalar e1e2e3 = h*1, which is central and squares to -1; the quaternion units
      i, j, k embed as grade-2 bivectors.
  A3  quat_conj (the paper's BAR) == Clifford conjugation, on random biquaternions.
  A4  hermitian_conj (the paper's DAGGER) == REVERSION, on random biquaternions.
      Discriminating controls: bar is NOT reversion, dagger is NOT Clifford conjugation.
      The identification is tight, not an accident of sign conventions.
  A5  dagger == matrix conjugate transpose under the M2(C) embedding biquat_to_matrix.
  A6  <q^dag q>_0 == sum_n |q_n|^2 == the grade-0 part of reversion(M) M, computed entirely
      inside the Clifford picture with no reference to the paper's dagger.
  A7  reversion is an anti-automorphism: reversion(A B) == reversion(B) reversion(A),
      matching (P Q)^dag = Q^dag P^dag.

NEGATIVE FINDING (reported as a negative, kept as a negative):
  N1  The pasted-AI claim "reversion sends t + x_vec to t - x_vec" is FALSE. Reversion FIXES
      grade-1 vectors (sign +1). It is CLIFFORD CONJUGATION that flips them.
  N2  So the pasted-AI boxed equation X X~ = t^2 - x^2 - y^2 - z^2 is FALSE:
        <X reversion(X)>_0     = t^2 + |x|^2  (Euclidean; and X reversion(X) is not even a
                                               pure scalar -- a grade-1 part survives)
        <X clifford_conj(X)>_0 = t^2 - |x|^2  (Minkowski; pure scalar; equals det X)
      The Minkowski form is the BAR product (the reduced norm), not the dagger product.
  N3  Same refutation at package level: cal's hermitian_conj leaves a paravector EXACTLY
      unchanged, while cal's quat_conj is what produces t - x_vec.

Not covered here (see exp_two_closures.py, C1-C7): multiplicativity of the reduced norm, the
null cone, and the bar-vs-dagger interference contrast. This script adds only the
identification of the two conjugations with the canonical Clifford involutions.
"""
import os
import sys

import numpy as np
import torch

# cal/ lives at the repo root, one level above "py tests"; this script is run from inside
# "py tests" as: python exp_clifford_conjugation_identification.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cal.biquaternion import (quat_mul, quat_conj, hermitian_conj, quat_norm_sq,
                              biquat_to_matrix, CDTYPE)

torch.manual_seed(0)
np.set_printoptions(precision=6, suppress=True)

TOL = 1e-4        # cal is complex64 (float32 precision)
TOL_EXACT = 1e-10  # pure-numpy float64 sections

# ---------------------------------------------------------------------------
# Cl(3,0) blade basis, in the Pauli matrix representation e1=s1, e2=s2, e3=s3.
# The 8 blades with REAL coefficients span M2(C) as a real 8-dimensional space and are
# orthonormal under <A,B> = (1/2) Re tr(A B^H), which is what makes the grade projection
# below well defined.
# ---------------------------------------------------------------------------
s0 = np.eye(2, dtype=np.complex128)
s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

BLADES = {
    0: [("1", s0)],
    1: [("e1", s1), ("e2", s2), ("e3", s3)],
    2: [("e12", s1 @ s2), ("e23", s2 @ s3), ("e31", s3 @ s1)],
    3: [("e123", s1 @ s2 @ s3)],
}

REVERSION_SIGN = lambda k: (-1) ** (k * (k - 1) // 2)
CLIFFORD_SIGN = lambda k: (-1) ** (k * (k + 1) // 2)
GRADEINV_SIGN = lambda k: (-1) ** k


def blade_coeff(M, B):
    """Real coefficient of orthonormal blade B in M. M shape (...,2,2) -> (...,)."""
    Bh = B.conj().T
    return 0.5 * np.real(np.einsum("...ij,ji->...", M, Bh))


def apply_grade_signs(M, sign_fn):
    """Rebuild M from its blade decomposition with a per-grade sign."""
    out = np.zeros_like(M)
    for k, blades in BLADES.items():
        s = sign_fn(k)
        for _, B in blades:
            c = blade_coeff(M, B)
            out = out + s * c[..., None, None] * B
    return out


def reversion(M):
    return apply_grade_signs(M, REVERSION_SIGN)


def clifford_conj(M):
    return apply_grade_signs(M, CLIFFORD_SIGN)


def grade_involution(M):
    return apply_grade_signs(M, GRADEINV_SIGN)


def grade_norms(M):
    """Per-grade coefficient norm, for a single 2x2 matrix."""
    return {k: float(np.linalg.norm([blade_coeff(M, B) for _, B in bl]))
            for k, bl in BLADES.items()}


def to_np(t):
    return t.detach().numpy().astype(np.complex128)


def mat(q):
    """cal biquaternion (...,4) -> numpy (...,2,2) complex128."""
    return to_np(biquat_to_matrix(q))


def rand_biquat(n):
    return (torch.randn(n, 4) + 1j * torch.randn(n, 4)).to(CDTYPE)


def ok(b):
    return "PASS" if b else "**FAIL**"


def sgnstr(fn):
    return "(" + ",".join(f"{fn(k):+d}"[0] for k in range(4)) + ")"


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def main():
    q = rand_biquat(200)
    r = rand_biquat(200)
    Mq = mat(q)
    Mr = mat(r)

    # -----------------------------------------------------------------------
    banner("A1. Grade sign patterns of the three canonical involutions on Cl(3,0)")
    print(f"  {'grade k':<10}{'reversion':>14}{'Clifford conj':>16}{'grade inv':>13}")
    print(f"  {'':<10}{'(-1)^k(k-1)/2':>14}{'(-1)^k(k+1)/2':>16}{'(-1)^k':>13}")
    for k in range(4):
        print(f"  {k:<10}{REVERSION_SIGN(k):>+14d}{CLIFFORD_SIGN(k):>+16d}"
              f"{GRADEINV_SIGN(k):>+13d}")
    pat_rev = sgnstr(REVERSION_SIGN)
    pat_cc = sgnstr(CLIFFORD_SIGN)
    pat_gi = sgnstr(GRADEINV_SIGN)
    print(f"\n  reversion        pattern {pat_rev}   expect (+,+,-,-):  {ok(pat_rev == '(+,+,-,-)')}")
    print(f"  Clifford conj    pattern {pat_cc}   expect (+,-,-,+):  {ok(pat_cc == '(+,-,-,+)')}")
    print(f"  grade involution pattern {pat_gi}   expect (+,-,+,-):  {ok(pat_gi == '(+,-,+,-)')}")
    comp = all(REVERSION_SIGN(k) * GRADEINV_SIGN(k) == CLIFFORD_SIGN(k) for k in range(4))
    print(f"  reversion . grade involution == Clifford conjugation:   {ok(comp)}")

    # -----------------------------------------------------------------------
    banner("A2. The algebra map: h = sqrt(-1) IS the pseudoscalar e1e2e3")
    I3 = s1 @ s2 @ s3
    print(f"  e1e2e3 == h*1 (h = 1j):                                {ok(np.allclose(I3, 1j * s0, atol=TOL_EXACT))}")
    print(f"  (e1e2e3)^2 == -1:                                      {ok(np.allclose(I3 @ I3, -s0, atol=TOL_EXACT))}")
    central = all(np.allclose(I3 @ X - X @ I3, 0, atol=TOL_EXACT) for X in (s1, s2, s3, Mq[0]))
    print(f"  e1e2e3 is central (commutes with e1,e2,e3 and a random"
          f"\n    biquaternion):                                       {ok(central)}")

    h = torch.tensor([[1j, 0, 0, 0]], dtype=CDTYPE)
    Mh = mat(h)[0]
    print(f"  cal's central imaginary h = (h,0,0,0) embeds as e1e2e3: {ok(np.allclose(Mh, I3, atol=TOL))}")

    print("\n  Grade content of the cal basis elements under biquat_to_matrix:")
    names = ["1", "i", "j", "k"]
    for n, name in enumerate(names):
        e = torch.zeros(1, 4, dtype=CDTYPE)
        e[0, n] = 1.0
        gn = grade_norms(mat(e)[0])
        live = [k for k in range(4) if gn[k] > 1e-6]
        print(f"    {name:<4} -> grades {live}  (norms { {k: round(v, 3) for k, v in gn.items()} })")
    units_are_bivectors = True
    for n in (1, 2, 3):
        e = torch.zeros(1, 4, dtype=CDTYPE)
        e[0, n] = 1.0
        gn = grade_norms(mat(e)[0])
        units_are_bivectors &= (gn[2] > 0.99 and max(gn[0], gn[1], gn[3]) < 1e-6)
    print(f"  quaternion units i,j,k are pure grade-2 bivectors:      {ok(units_are_bivectors)}")
    print("  >> So a biquaternion q_n = a_n + h b_n spans all four grades: Re(q0) -> grade 0,")
    print("     Im(q0) -> grade 3, Re(q_1..3) -> grade 2, Im(q_1..3) -> grade 1.")

    # -----------------------------------------------------------------------
    banner("A3/A4. Identifying the paper's conjugations with the Clifford involutions")
    M_bar = mat(quat_conj(q))
    M_dag = mat(hermitian_conj(q))
    cc = clifford_conj(Mq)
    rev = reversion(Mq)
    gi = grade_involution(Mq)

    bar_is_cc = np.allclose(M_bar, cc, atol=TOL)
    dag_is_rev = np.allclose(M_dag, rev, atol=TOL)
    print(f"  A3  paper's BAR    (quat_conj)      == Clifford conjugation:  {ok(bar_is_cc)}"
          f"   max|diff|={np.abs(M_bar - cc).max():.2e}")
    print(f"  A4  paper's DAGGER (hermitian_conj) == REVERSION:             {ok(dag_is_rev)}"
          f"   max|diff|={np.abs(M_dag - rev).max():.2e}")

    print("\n  Discriminating controls (these MUST fail, or the identification is vacuous):")
    bar_not_rev = not np.allclose(M_bar, rev, atol=TOL)
    dag_not_cc = not np.allclose(M_dag, cc, atol=TOL)
    bar_not_gi = not np.allclose(M_bar, gi, atol=TOL)
    dag_not_gi = not np.allclose(M_dag, gi, atol=TOL)
    print(f"    bar    is NOT reversion:                {ok(bar_not_rev)}"
          f"   max|diff|={np.abs(M_bar - rev).max():.3f}")
    print(f"    dagger is NOT Clifford conjugation:     {ok(dag_not_cc)}"
          f"   max|diff|={np.abs(M_dag - cc).max():.3f}")
    print(f"    bar    is NOT grade involution:         {ok(bar_not_gi)}"
          f"   max|diff|={np.abs(M_bar - gi).max():.3f}")
    print(f"    dagger is NOT grade involution:         {ok(dag_not_gi)}"
          f"   max|diff|={np.abs(M_dag - gi).max():.3f}")
    print("  >> The dagger is ONE canonical involution (reversion), not a composite of two.")

    # -----------------------------------------------------------------------
    banner("A5. dagger == matrix conjugate transpose under the M2(C) embedding")
    Mq_H = np.conjugate(np.swapaxes(Mq, -1, -2))
    dag_is_H = np.allclose(M_dag, Mq_H, atol=TOL)
    print(f"  M(q^dag) == M(q)^H:                                    {ok(dag_is_H)}"
          f"   max|diff|={np.abs(M_dag - Mq_H).max():.2e}")
    bar_H = np.allclose(M_bar, Mq_H, atol=TOL)
    print(f"  M(q_bar) == M(q)^H (control, must fail):               {ok(not bar_H)}")
    print("  >> Reversion in Cl(3,0) == Hermitian adjoint in M2(C). The paper's dagger is")
    print("     simultaneously the GA reversion and the matrix adjoint; nothing bespoke.")

    # -----------------------------------------------------------------------
    banner("A6. <q^dag q>_0 IS the canonical GA reversion norm")
    dagq = quat_mul(hermitian_conj(q), q)          # cal's dagger closure
    born_cal = to_np(dagq)[..., 0]
    sumsq = (q.real ** 2 + q.imag ** 2).sum(-1).numpy().astype(np.float64)

    # Same quantity computed with NO reference to cal's dagger: pure Clifford picture.
    rev_norm = blade_coeff(np.einsum("...ij,...jk->...ik", rev, Mq), s0)

    real_ok = np.abs(born_cal.imag).max() < TOL
    eq_sumsq = np.allclose(born_cal.real, sumsq, atol=TOL, rtol=1e-4)
    eq_rev = np.allclose(rev_norm, sumsq, atol=TOL, rtol=1e-4)
    print(f"  <q^dag q>_0 is real:                                   {ok(real_ok)}"
          f"   max|imag|={np.abs(born_cal.imag).max():.2e}")
    print(f"  <q^dag q>_0 == sum_n |q_n|^2:                          {ok(eq_sumsq)}"
          f"   max|diff|={np.abs(born_cal.real - sumsq).max():.2e}")
    print(f"  <reversion(M) M>_0 == sum_n |q_n|^2  (Clifford side):  {ok(eq_rev)}"
          f"   max|diff|={np.abs(rev_norm - sumsq).max():.2e}")
    print(f"\n  sample: sum|q_n|^2 = {sumsq[0]:.6f}   <q^dag q>_0 = {born_cal[0].real:.6f}"
          f"   <rev(M) M>_0 = {rev_norm[0]:.6f}")
    print("  >> The Born weight is the textbook reversion norm <X~ X>_0 of Cl(3,0).")

    # -----------------------------------------------------------------------
    banner("A7. Reversion is an anti-automorphism (matches (P Q)^dag = Q^dag P^dag)")
    Mqr = np.einsum("...ij,...jk->...ik", Mq, Mr)
    lhs = reversion(Mqr)
    rhs = np.einsum("...ij,...jk->...ik", reversion(Mr), reversion(Mq))
    anti = np.allclose(lhs, rhs, atol=TOL)
    print(f"  reversion(A B) == reversion(B) reversion(A):           {ok(anti)}"
          f"   max|diff|={np.abs(lhs - rhs).max():.2e}")
    fwd = np.allclose(lhs, np.einsum("...ij,...jk->...ik", reversion(Mq), reversion(Mr)),
                      atol=TOL)
    print(f"  reversion(A B) == reversion(A) reversion(B) (control,"
          f"\n    must fail; reversion reverses order):                 {ok(not fwd)}")

    # -----------------------------------------------------------------------
    banner("N1. NEGATIVE: 'reversion sends t + x_vec to t - x_vec' is FALSE")
    print("  Pasted-AI block claimed: applying the paper's dagger (= reversion) to a")
    print("  paravector X = t + x_vec flips the spatial part, giving X~ = t - x_vec.")
    print(f"\n  Reversion sign on grade 1 is {REVERSION_SIGN(1):+d}. Reversion FIXES vectors.")
    print(f"  Clifford conjugation sign on grade 1 is {CLIFFORD_SIGN(1):+d}. It flips them.\n")

    t, x, y, z = 2.0, 0.3, -0.7, 0.5
    X = t * s0 + x * s1 + y * s2 + z * s3
    Xflip = t * s0 - x * s1 - y * s2 - z * s3

    def show(M, name):
        print(f"    {name:<26} t={blade_coeff(M, s0):+.3f}  x={blade_coeff(M, s1):+.3f}"
              f"  y={blade_coeff(M, s2):+.3f}  z={blade_coeff(M, s3):+.3f}")

    show(X, "X = t + x_vec")
    show(reversion(X), "reversion(X)")
    show(clifford_conj(X), "clifford_conj(X)")
    rev_flips = np.allclose(reversion(X), Xflip, atol=TOL_EXACT)
    cc_flips = np.allclose(clifford_conj(X), Xflip, atol=TOL_EXACT)
    rev_fixes = np.allclose(reversion(X), X, atol=TOL_EXACT)
    print(f"\n  Does reversion give t - x_vec, as the block claims?     {rev_flips}"
          f"   -> claim is {'TRUE' if rev_flips else 'FALSE'}")
    print(f"  Does reversion FIX X entirely (grades 0 and 1 only)?   {rev_fixes}")
    print(f"  Is it Clifford conjugation that gives t - x_vec?       {cc_flips}")
    print(f"  N1 negative finding reproduced (claim is false):       "
          f"{ok((not rev_flips) and rev_fixes and cc_flips)}")

    # -----------------------------------------------------------------------
    banner("N2. NEGATIVE: the boxed equation X X~ = t^2 - x^2 - y^2 - z^2 is FALSE")
    mink = t ** 2 - (x ** 2 + y ** 2 + z ** 2)
    eucl = t ** 2 + (x ** 2 + y ** 2 + z ** 2)
    print(f"  Minkowski t^2 - |x|^2 = {mink:+.4f}      Euclidean t^2 + |x|^2 = {eucl:+.4f}\n")

    XXrev = X @ reversion(X)
    XXcc = X @ clifford_conj(X)
    g0_rev = blade_coeff(XXrev, s0)
    g0_cc = blade_coeff(XXcc, s0)
    vec_rev = np.array([blade_coeff(XXrev, b) for b in (s1, s2, s3)])
    vec_cc = np.array([blade_coeff(XXcc, b) for b in (s1, s2, s3)])
    detX = np.real(np.linalg.det(X))

    show(XXrev, "X * reversion(X)")
    print(f"      grade-0 = {g0_rev:+.4f}   == Euclidean? {np.isclose(g0_rev, eucl)}"
          f"   == Minkowski? {np.isclose(g0_rev, mink)}")
    print(f"      pure scalar? {np.allclose(vec_rev, 0, atol=TOL_EXACT)}"
          f"   surviving grade-1 norm = {np.linalg.norm(vec_rev):.4f}")
    show(XXcc, "X * clifford_conj(X)")
    print(f"      grade-0 = {g0_cc:+.4f}   == Minkowski? {np.isclose(g0_cc, mink)}")
    print(f"      pure scalar? {np.allclose(vec_cc, 0, atol=TOL_EXACT)}"
          f"   == det X = {detX:+.4f}? {np.isclose(g0_cc, detX)}")
    n2 = (np.isclose(g0_rev, eucl) and not np.isclose(g0_rev, mink)
          and not np.allclose(vec_rev, 0, atol=TOL_EXACT)
          and np.isclose(g0_cc, mink) and np.allclose(vec_cc, 0, atol=TOL_EXACT)
          and np.isclose(g0_cc, detX))
    print(f"\n  N2 negative finding reproduced (boxed equation false):  {ok(n2)}")
    print("  >> t^2 - |x|^2 is the CLIFFORD CONJUGATION product = the paper's BAR = det.")
    print("     The reversion product gives t^2 + |x|^2 and is not even a pure scalar.")
    print("     The block merged the paper's two closures into one and lost Lemma 4.1.")

    # -----------------------------------------------------------------------
    banner("N3. Same refutation at package level, using cal's own conjugations")
    # Paravector in cal coordinates. From A2's embedding: e1 = (0,0,-h,0), e2 = (0,-h,0,0),
    # e3 = (0,0,0,-h). So X = t + x e1 + y e2 + z e3 has cal coeffs below.
    Xq = torch.tensor([[t + 0j, -1j * y, -1j * x, -1j * z]], dtype=CDTYPE)
    print(f"  paravector as cal biquaternion: {to_np(Xq)[0]}")
    print(f"  round-trips to the same matrix as the Pauli build:      "
          f"{ok(np.allclose(mat(Xq)[0], X, atol=TOL))}")

    Xq_dag = hermitian_conj(Xq)
    Xq_bar = quat_conj(Xq)
    dag_fixes = torch.allclose(Xq_dag, Xq, atol=TOL)
    print(f"\n  cal hermitian_conj(X) (paper's DAGGER): {to_np(Xq_dag)[0]}")
    print(f"    leaves the paravector EXACTLY unchanged:              {ok(dag_fixes)}")
    print(f"  cal quat_conj(X)      (paper's BAR):    {to_np(Xq_bar)[0]}")
    show(mat(Xq_bar)[0], "  -> as a paravector")
    print(f"    this is the one that gives t - x_vec:                 "
          f"{ok(np.allclose(mat(Xq_bar)[0], Xflip, atol=TOL))}")

    N_bar = quat_norm_sq(Xq).item()
    born = quat_mul(hermitian_conj(Xq), Xq)[..., 0].item()
    print(f"\n  <X X_bar>_0 = quat_norm_sq(X) = {N_bar.real:+.4f}   == Minkowski {mink:+.4f}? "
          f"{ok(np.isclose(N_bar.real, mink, atol=TOL))}")
    print(f"  <X^dag X>_0                   = {born.real:+.4f}   == Euclidean {eucl:+.4f}? "
          f"{ok(np.isclose(born.real, eucl, atol=TOL))}")
    print("  >> cal's bar carries the indefinite metric; cal's dagger carries the positive")
    print("     event weight. The pasted-AI claim assigns the metric to the dagger, which")
    print("     would collapse the two closures into one. It does not survive the code.")

    # -----------------------------------------------------------------------
    banner("SUMMARY")
    positives = [pat_rev == "(+,+,-,-)", pat_cc == "(+,-,-,+)", pat_gi == "(+,-,+,-)", comp,
                 central, units_are_bivectors, bar_is_cc, dag_is_rev, bar_not_rev,
                 dag_not_cc, dag_is_H, real_ok, eq_sumsq, eq_rev, anti]
    print(f"  POSITIVE (A1-A7): all identification claims hold: {ok(all(positives))}")
    print("    bar == Clifford conjugation, dagger == REVERSION == matrix adjoint,")
    print("    and <W^dag W>_0 is the canonical Cl(3,0) reversion norm <W~ W>_0.")
    print(f"  NEGATIVE (N1-N3): the pasted-AI reversion claim is FALSE and stays false:"
          f" {ok((not rev_flips) and rev_fixes and cc_flips and n2 and dag_fixes)}")
    print("    Reversion FIXES grade-1 vectors. X X~ grade-0 = t^2 + |x|^2 (Euclidean),")
    print("    not t^2 - |x|^2. The Minkowski form is X X_bar = det X, the BAR product.")


if __name__ == "__main__":
    main()
