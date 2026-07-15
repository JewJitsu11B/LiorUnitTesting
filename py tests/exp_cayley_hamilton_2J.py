"""
Where does J^2 = 2J come from? Cayley-Hamilton, and nothing else.

The manuscript leans on J^2 = 2J and on <J^dag J>_0 = 2 as if they were separate inputs.
They are not. C (x) H == M_2(C), so [H:C] = 2, so the characteristic polynomial of EVERY
element is QUADRATIC. That single fact -- the same 2 as "rank <= 2" -- produces the whole
identity, and the two premises it needs (det J = 0, tr J = 2) are the null cone and the
Born normalisation respectively. Nothing is imported.

CLAIMS VERIFIED (cal's M_2(C) rep, M(q) = [[q0+h q3, q1+h q2], [-q1+h q2, q0-h q3]]):
  A1  HANDEDNESS GUARD. cal's quat_mul (Eq. 3) is RIGHT-handed: i j = k, i j k = -1, and
      the rep is a homomorphism: M(i) M(j) = M(k).
  A2  tr M(q) = 2 q0 exactly, for every biquaternion. det M(q) = quat_norm_sq(q).
  A3  Cayley-Hamilton: J^2 - tr(J) J + det(J) I = 0 for every J, residual ~1e-16.
      The polynomial is quadratic because [H:C] = 2 -- the same 2 as rank <= 2.
  A4  tr(a a^dag) = 2 <a^dag a>_0 EXACTLY for every a, null or not. So "tr J = 2" is not
      an extra hypothesis; it IS the Born normalisation <a^dag a>_0 = 1.
  A5  J = a a^dag is Hermitian for free -- the dagger is native to C (x) H, and
      M(q^dag) = M(q)^dag. With a null and normalised, every step is forced:
      det J = 0, rank J = 1, tr J = 2, J^2 = 2J (||J^2 - 2J|| ~ 1e-16), <J^dag J>_0 = 2.
  A6  The rank ladder {0, 1, 2} is COMPLETE, not truncated -- because M : C (x) H -> M_2(C)
      is a BIJECTION (verified both ways by round-trip), so the carrier is exactly M_2(C)
      and there is no larger object available to have rank 3. NOTE: "max rank over random
      2x2 draws is 2" is NOT evidence for this -- matrix_rank on a (2,2) array is bounded
      by 2 by the array's SHAPE, and that assertion cannot fail. The bijection is the claim
      with content; the rank census is drawn through the algebra, J = M(q), to match it.

NEGATIVE FINDINGS (reported as negatives, kept as negatives):
  N1  "J^2 = 2J  <=>  det J = 0 AND tr J = 2" is FALSE as a biconditional. The (<=)
      direction holds unconditionally by Cayley-Hamilton. The (=>) direction FAILS at
      both ends of the rank ladder:
          J = 0   : J^2 = 2J holds, but tr J = 0, not 2.
          J = 2 I : J^2 = 2J holds, but det J = 4, not 0 -- the MAXIMALLY non-null element.
      So J^2 = 2J does NOT by itself put the current on the null cone. The full solution
      set of J^2 = 2J is {2E : E^2 = E} = rank 0, rank 1, rank 2 -- exactly the ladder.
      Repaired: J^2 = 2J AND rank J = 1  <=>  det J = 0 AND tr J = 2.
  N2  "<J^dag J>_0 = 2 * rank" is FALSE in general. It holds only on the Hermitian branch.
      An OBLIQUE rank-1 idempotent E = v w^dag/(w^dag v) with w not parallel to v satisfies
      E^2 = E, tr(2E) = 2 and det(2E) = 0 -- so it passes the repaired N1 test -- yet
      E^dag != E and <(2E)^dag (2E)>_0 > 2. The true statement is
          <J^dag J>_0 = 2 tr(E^dag E) >= 2 rank(E),  equality iff E is ORTHOGONAL.
      Idempotence is not projection. det and trace do not see the difference.

PRECISION. cal is complex64 (eps ~ 1e-7) and the null-cone checks are CANCELLING
differences, so complex64 CANNOT resolve them to 1e-15. Every finding above is computed in
numpy complex128 using cal's rep FORMULA, written out. Section 9 then calls the real
cal.biquaternion functions on the same inputs and confirms agreement at a tolerance that is
HONEST for complex64 (rtol 1e-5, float32-limited). No 1e-15 identity is ever asserted
through complex64.
"""
import os
import sys

import numpy as np
import torch

# cal/ lives at the repo root, one level above "py tests"; this script is run from inside
# "py tests" as: python exp_cayley_hamilton_2J.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cal.biquaternion import (quat_mul, hermitian_conj, quat_norm_sq,
                              biquat_to_matrix, matrix_to_biquat, CDTYPE)

SEED = 20260715
rng = np.random.default_rng(SEED)

TOL_F64 = 1e-12    # pure-numpy complex128 sections: real machine precision
RTOL_C64 = 1e-5    # cal cross-check: complex64, eps ~ 1e-7. FLOAT32-LIMITED, not exact.
ATOL_C64 = 1e-5

RULE = "=" * 78
I2 = np.eye(2, dtype=np.complex128)


def banner(title):
    print("\n" + RULE)
    print(title)
    print(RULE)


def ok(flag):
    return "PASS" if flag else "FAIL"


# ---------------------------------------------------------------------------
# cal's formulas, written out in complex128. These mirror cal.biquaternion exactly;
# section 9 checks the mirror against the package itself.
# ---------------------------------------------------------------------------
def qmul(P, Q):
    """cal.quat_mul (paper Eq. 3), in complex128."""
    p0, p1, p2, p3 = P[0], P[1], P[2], P[3]
    q0, q1, q2, q3 = Q[0], Q[1], Q[2], Q[3]
    return np.array([p0*q0 - p1*q1 - p2*q2 - p3*q3,
                     p0*q1 + p1*q0 + p2*q3 - p3*q2,
                     p0*q2 - p1*q3 + p2*q0 + p3*q1,
                     p0*q3 + p1*q2 - p2*q1 + p3*q0], dtype=np.complex128)


def hdag(q):
    """cal.hermitian_conj (paper Eq. 9), in complex128."""
    c = np.conj(q)
    return np.array([c[0], -c[1], -c[2], -c[3]], dtype=np.complex128)


def M(q):
    """cal.biquat_to_matrix, in complex128.  h = 1j."""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return np.array([[q0 + 1j*q3,  q1 + 1j*q2],
                     [-q1 + 1j*q2, q0 - 1j*q3]], dtype=np.complex128)


def M_inv(A):
    """cal.matrix_to_biquat, in complex128."""
    m00, m01, m10, m11 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    return np.array([(m00 + m11)/2, (m01 - m10)/2,
                     (m01 + m10)/2j, (m00 - m11)/2j], dtype=np.complex128)


def born0(q):
    """<q^dag q>_0 = sum_mu |q_mu|^2 -- the grade-0 part of the dagger product."""
    return float(np.sum(np.abs(q)**2))


def born0_mat(A):
    """<J^dag J>_0 for J = M(q):  tr(J^dag J)/2 = ||J||_F^2 / 2."""
    return float(np.real(np.trace(A.conj().T @ A))/2)


def rank_of(A):
    return int(np.linalg.matrix_rank(A, tol=1e-9))


def rand_bq():
    return (rng.normal(size=4) + 1j*rng.normal(size=4)).astype(np.complex128)


def null_dir():
    """a = 1 + h*(i n) with |n| = 1:  N(a) = 1 + (h)^2 |n|^2 = 0.  On the null cone."""
    n = rng.normal(size=3)
    n = n/np.linalg.norm(n)
    return np.array([1.0, 1j*n[0], 1j*n[1], 1j*n[2]], dtype=np.complex128)


def to_t(q):
    return torch.tensor(q, dtype=CDTYPE)


E0 = np.array([1, 0, 0, 0], dtype=np.complex128)
EI = np.array([0, 1, 0, 0], dtype=np.complex128)
EJ = np.array([0, 0, 1, 0], dtype=np.complex128)
EK = np.array([0, 0, 0, 1], dtype=np.complex128)


def main():
    results = {}

    # =======================================================================
    banner("1. HANDEDNESS GUARD -- fix the basis before any claim is made")
    # =======================================================================
    print("  cal.quat_mul is Eq. 3. Check the convention it actually implements, on the")
    print("  basis this script uses, BEFORE trusting any product below.\n")
    ij = qmul(EI, EJ)
    jk = qmul(EJ, EK)
    ki = qmul(EK, EI)
    ijk = qmul(qmul(EI, EJ), EK)
    ii = qmul(EI, EI)
    jj = qmul(EJ, EJ)
    kk = qmul(EK, EK)
    h_rows = [("i j", ij, EK, "k"), ("j k", jk, EI, "i"), ("k i", ki, EJ, "j"),
              ("i i", ii, -E0, "-1"), ("j j", jj, -E0, "-1"), ("k k", kk, -E0, "-1"),
              ("(i j) k", ijk, -E0, "-1")]
    print(f"  {'product':>10}{'computed':>26}{'expected':>12}{'':>8}")
    h_all = True
    for nm, got, want, wname in h_rows:
        good = np.allclose(got, want, atol=TOL_F64)
        h_all = h_all and good
        print(f"  {nm:>10}{str(np.round(got.real, 6)):>26}{wname:>12}{ok(good):>8}")
    # the rep must be a homomorphism, or the matrix picture below means nothing
    hom = np.allclose(M(EI) @ M(EJ), M(EK), atol=TOL_F64)
    hom2 = all(np.allclose(M(qmul(a, b)), M(a) @ M(b), atol=TOL_F64)
               for a, b in [(rand_bq(), rand_bq()) for _ in range(50)])
    print(f"\n  M(i) M(j) == M(k) (rep is a homomorphism on the basis): {ok(hom)}")
    print(f"  M(a b) == M(a) M(b) over 50 random pairs:                {ok(hom2)}")
    print(f"\n  >> RIGHT-HANDED: i j = k and i j k = -1. Section 1: {ok(h_all and hom and hom2)}")
    results["1 handedness guard"] = h_all and hom and hom2

    # =======================================================================
    banner("2. tr M(q) = 2 q0 EXACTLY, and det M(q) = N(q). Finding 1.")
    # =======================================================================
    print("  M(q) = [[q0 + h q3, q1 + h q2], [-q1 + h q2, q0 - h q3]], h = 1j.")
    print("  The off-diagonal h q3 terms cancel on the diagonal sum, so tr = 2 q0 with no")
    print("  reference to the vector part at all. det = q0^2+q1^2+q2^2+q3^2 = quat_norm_sq.\n")
    print(f"  {'trial':>7}{'q0':>26}{'tr M(q)':>26}{'|tr - 2 q0|':>14}{'|det - N|':>12}")
    tr_res, det_res = [], []
    for t in range(5):
        q = rand_bq()
        A = M(q)
        tr = np.trace(A)
        det = np.linalg.det(A)
        N = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
        tr_res.append(abs(tr - 2*q[0]))
        det_res.append(abs(det - N))
        print(f"  {t:>7}{str(np.round(q[0], 5)):>26}{str(np.round(tr, 5)):>26}"
              f"{tr_res[-1]:>14.2e}{det_res[-1]:>12.2e}")
    s2 = max(tr_res) < TOL_F64 and max(det_res) < TOL_F64
    print(f"\n  max |tr M(q) - 2 q0| = {max(tr_res):.3e}   (machine precision, complex128)")
    print(f"  max |det M(q) - N(q)| = {max(det_res):.3e}")
    print(f"\n  >> tr reads off q0 and det reads off the reduced norm. Section 2: {ok(s2)}")
    results["2 tr = 2 q0"] = s2

    # =======================================================================
    banner("3. CAYLEY-HAMILTON. The polynomial is QUADRATIC because [H:C] = 2. Finding 2.")
    # =======================================================================
    print("  C (x) H == M_2(C). A 2x2 matrix has a degree-2 characteristic polynomial:")
    print("      J^2 - tr(J) J + det(J) I = 0")
    print("  The 2 in 'degree 2' is [H:C] = 2, which is the SAME 2 as 'rank <= 2'. It is")
    print("  not the 2 in 'integrate twice' and it is not the 2 in 'two-point function'.\n")
    print(f"  {'trial':>7}{'||J||_F':>12}{'||J^2 - tr(J)J + det(J)I||':>30}{'relative':>12}")
    ch_res = []
    for t in range(6):
        J = M(rand_bq())
        ch = J @ J - np.trace(J)*J + np.linalg.det(J)*I2
        nf = np.linalg.norm(J)
        rel = np.abs(ch).max()/nf**2
        ch_res.append(rel)
        print(f"  {t:>7}{nf:>12.4f}{np.abs(ch).max():>30.3e}{rel:>12.2e}")
    s3 = max(ch_res) < TOL_F64
    print(f"\n  worst relative residual = {max(ch_res):.3e}   (complex128; ~1e-16 as claimed)")
    print(f"\n  >> Cayley-Hamilton holds identically. Section 3: {ok(s3)}")
    results["3 cayley-hamilton"] = s3

    # =======================================================================
    banner("4. NEGATIVE (N1). 'J^2 = 2J <=> det J = 0 AND tr J = 2' is FALSE. Finding 3.")
    # =======================================================================
    print("  The finding as stated is a BICONDITIONAL. Test both directions separately.\n")
    print("  (<=)  det J = 0 and tr J = 2  =>  J^2 = 2J.")
    print("        CH gives J^2 = tr(J) J - det(J) I = 2J - 0. Unconditional. Check it:\n")
    print(f"  {'source of J':>34}{'tr J':>10}{'det J':>12}{'||J^2 - 2J||':>16}")
    back_all = True
    for t in range(3):
        a = null_dir()
        a = a/np.sqrt(born0(a))
        A = M(a)
        J = A @ A.conj().T                    # same object twice: J = a a^dag, not a b^dag
        r = np.abs(J @ J - 2*J).max()
        back_all = back_all and r < TOL_F64
        print(f"  {'a a^dag, a null+normalised ' + str(t):>34}"
              f"{np.real(np.trace(J)):>10.4f}{abs(np.linalg.det(J)):>12.2e}{r:>16.3e}")
    # an OBLIQUE rank-1 idempotent also has det 0, tr 2 -- and CH does not care
    v = rng.normal(size=2) + 1j*rng.normal(size=2)
    w = rng.normal(size=2) + 1j*rng.normal(size=2)
    E_obl = np.outer(v, w.conj())/(w.conj() @ v)
    J_obl = 2*E_obl
    r_obl = np.abs(J_obl @ J_obl - 2*J_obl).max()
    back_all = back_all and r_obl < TOL_F64
    print(f"  {'2 * OBLIQUE rank-1 idempotent':>34}{np.real(np.trace(J_obl)):>10.4f}"
          f"{abs(np.linalg.det(J_obl)):>12.2e}{r_obl:>16.3e}")
    print(f"\n        (<=) direction: {ok(back_all)}")
    if back_all:
        print("        It holds, and it does NOT require Hermiticity. det and trace cannot")
        print("        see the difference between the orthogonal and the oblique rank-1")
        print("        idempotent. Both give J^2 = 2J.")
    else:
        print("        It FAILED. Cayley-Hamilton should make this unconditional; a failure")
        print("        here means the rep or the arithmetic above is wrong, not the finding.")

    print("\n  (=>)  J^2 = 2J  =>  det J = 0 and tr J = 2.  THIS IS THE ONE THAT FAILS.")
    print("        J^2 = 2J says exactly that J/2 is IDEMPOTENT. In M_2(C) the idempotents")
    print("        come in three ranks, and two of them break the claim:\n")
    print(f"  {'J with J^2 = 2J':>34}{'rank':>6}{'tr J':>8}{'det J':>10}"
          f"{'||J^2-2J||':>13}{'tr=2 & det=0?':>15}")
    v1 = rng.normal(size=2) + 1j*rng.normal(size=2)
    v1 = v1/np.linalg.norm(v1)
    E_orth = np.outer(v1, v1.conj())
    fwd_rows = [("0            (rank 0)", np.zeros((2, 2), dtype=np.complex128)),
                ("2 * E_orth   (rank 1)", 2*E_orth),
                ("2 * E_oblique(rank 1)", 2*E_obl),
                ("2 * I        (rank 2)", 2*I2)]
    counterexamples = []
    for nm, J in fwd_rows:
        idem = np.abs(J @ J - 2*J).max()
        trJ = np.real(np.trace(J))
        dJ = np.linalg.det(J)
        claim = bool(np.isclose(trJ, 2.0, atol=TOL_F64) and abs(dJ) < TOL_F64)
        assert idem < TOL_F64, "row must satisfy J^2 = 2J by construction"
        if not claim:
            counterexamples.append(nm.strip())
        print(f"  {nm:>34}{rank_of(J):>6}{trJ:>8.4f}{abs(dJ):>10.2e}"
              f"{idem:>13.3e}{str(claim):>15}")
    fwd_holds = len(counterexamples) == 0
    print(f"\n        (=>) direction: {ok(fwd_holds)}")
    print(f"        Counterexamples found: {len(counterexamples)} -> {counterexamples}")
    print("        J = 0   satisfies J^2 = 2J with tr J = 0, NOT 2.")
    print("        J = 2 I satisfies J^2 = 2J with det J = 4, NOT 0. It is the MAXIMALLY")
    print("        non-null element -- as far from the null cone as M_2(C) goes -- and it")
    print("        still obeys J^2 = 2J. So J^2 = 2J does NOT put the current on the cone.")

    print("\n  REPAIR. Add the rank, which is the thing the two counterexamples differ in:")
    print("      J^2 = 2J AND rank J = 1   <=>   det J = 0 AND tr J = 2")
    print("  (det J = 0 forces rank <= 1; tr J = 2 != 0 forces J != 0, so rank = 1.)\n")
    rep_ok = True
    for nm, J in fwd_rows:
        lhs = bool(np.abs(J @ J - 2*J).max() < TOL_F64 and rank_of(J) == 1)
        rhs = bool(np.isclose(np.real(np.trace(J)), 2.0, atol=TOL_F64)
                   and abs(np.linalg.det(J)) < TOL_F64)
        rep_ok = rep_ok and (lhs == rhs)
        print(f"  {nm:>34}   LHS={str(lhs):>5}   RHS={str(rhs):>5}   agree: {ok(lhs == rhs)}")
    # Random controls. A control that leaves BOTH sides False is a double-negative and
    # tests nothing, so the classes below are chosen to drive lhs and rhs TRUE as well.
    ctl_lhs_true = ctl_rhs_true = 0
    for t in range(200):
        kind = t % 5
        if kind == 0:                          # generic: both sides False
            Jr = M(rand_bq())
        elif kind == 1:                        # a a^dag, null + normalised: both True
            aa = null_dir()
            aa = aa/np.sqrt(born0(aa))
            Ar = M(aa)
            Jr = Ar @ Ar.conj().T
        elif kind == 2:                        # general rank-1 idempotent (S diag(1,0) S^-1)
            S = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
            if abs(np.linalg.det(S)) < 1e-6:
                continue
            Jr = 2*(S @ np.diag([1.0, 0.0]).astype(np.complex128) @ np.linalg.inv(S))
        elif kind == 3:                        # a a^dag, null but NOT normalised: tr != 2
            aa = null_dir()
            Ar = M(aa)
            Jr = Ar @ Ar.conj().T
        else:                                  # the two rank-ladder ends
            Jr = 2*I2 if t % 2 else np.zeros((2, 2), dtype=np.complex128)
        lhs = bool(np.abs(Jr @ Jr - 2*Jr).max() < 1e-9 and rank_of(Jr) == 1)
        rhs = bool(np.isclose(np.real(np.trace(Jr)), 2.0, atol=1e-9)
                   and abs(np.linalg.det(Jr)) < 1e-9)
        ctl_lhs_true += lhs
        ctl_rhs_true += rhs
        rep_ok = rep_ok and (lhs == rhs)
    print(f"\n  random controls: 200 draws over 5 classes (generic, a a^dag normalised,")
    print(f"  general rank-1 idempotent, a a^dag unnormalised, and the ladder ends).")
    print(f"    controls with LHS true: {ctl_lhs_true:>3} / 200   (not a double-negative)")
    print(f"    controls with RHS true: {ctl_rhs_true:>3} / 200")
    print(f"\n  repaired biconditional over the 4 rows + 200 random controls: {ok(rep_ok)}")
    print("\n  >> FINDING 3 IS WRONG AS STATED. The (<=) half is a theorem; the (=>) half")
    print("     is false at rank 0 and rank 2. Section 4 records the finding as FAILED and")
    print(f"     the repair as {ok(rep_ok)}.")
    print(f"     Finding 3 as stated ('<=>'):        {ok(fwd_holds and back_all)}")
    print(f"     Finding 3 repaired (+ rank J = 1):  {ok(rep_ok and back_all)}")
    results["4 finding 3 as stated (<=>)"] = fwd_holds and back_all
    results["4 finding 3 repaired (+rank=1)"] = rep_ok and back_all

    # =======================================================================
    banner("5. tr(a a^dag) = 2 <a^dag a>_0 EXACTLY. So 'tr J = 2' IS Born. Finding 4.")
    # =======================================================================
    print("  ||M(q)||_F^2 = |q0+h q3|^2 + |q0-h q3|^2 + |q1+h q2|^2 + |-q1+h q2|^2")
    print("               = 2(|q0|^2+|q3|^2) + 2(|q1|^2+|q2|^2) = 2 sum_mu |q_mu|^2.")
    print("  and tr(a a^dag) = ||M(a)||_F^2. So tr J = 2 <=> <a^dag a>_0 = 1: it is the")
    print("  NORMALISATION of the state, not a second hypothesis smuggled in beside it.\n")
    print(f"  {'a':>26}{'<a^dag a>_0':>14}{'tr(a a^dag)':>14}{'ratio':>10}{'':>8}")
    s5 = True
    for t in range(4):
        a = rand_bq()                          # generic: NOT null, no normalisation
        A = M(a)
        tr = np.real(np.trace(A @ A.conj().T))
        b = born0(a)
        good = abs(tr - 2*b) < TOL_F64 * max(1.0, abs(tr))
        s5 = s5 and good
        print(f"  {'generic ' + str(t):>26}{b:>14.6f}{tr:>14.6f}{tr/b:>10.6f}{ok(good):>8}")
    for t in range(2):
        a = null_dir()
        A = M(a)
        tr = np.real(np.trace(A @ A.conj().T))
        b = born0(a)
        good = abs(tr - 2*b) < TOL_F64 * max(1.0, abs(tr))
        s5 = s5 and good
        print(f"  {'null ' + str(t):>26}{b:>14.6f}{tr:>14.6f}{tr/b:>10.6f}{ok(good):>8}")
    # the grade-0 part of the dagger product, computed in the ALGEBRA, agrees
    a = rand_bq()
    g0_dag = qmul(hdag(a), a)[0]
    g0_rev = qmul(a, hdag(a))[0]
    alg_ok = (abs(g0_dag - born0(a)) < TOL_F64 and abs(g0_rev - born0(a)) < TOL_F64)
    s5 = s5 and alg_ok
    print(f"\n  <a^dag a>_0 = {g0_dag.real:.6f}, <a a^dag>_0 = {g0_rev.real:.6f}, "
          f"sum|a_mu|^2 = {born0(a):.6f}: {ok(alg_ok)}")
    print(f"\n  >> ratio is 2.000000 for every a, null or not. Section 5: {ok(s5)}")
    results["5 tr(a a^dag) = 2 <a^dag a>_0"] = s5

    # =======================================================================
    banner("6. THE FORCED CHAIN. a null + normalised => everything, no free choices. Finding 5.")
    # =======================================================================
    print("  M(q^dag) = M(q)^dag: the dagger is NATIVE to C (x) H, it survives the embedding.")
    print("  So J = a a^dag is Hermitian for free -- there is no Hermiticity axiom to add.\n")
    dag_native = all(np.allclose(M(hdag(q)), M(q).conj().T, atol=TOL_F64)
                     for q in [rand_bq() for _ in range(50)])
    print(f"  M(q^dag) == M(q)^dag over 50 random biquaternions: {ok(dag_native)}")

    print("\n  Scaling first: only ONE normalisation gives J^2 = 2J. It is <a^dag a>_0 = 1.")
    print("  ONE null direction, drawn ONCE, scaled three ways. The rows differ only by")
    print("  the scale -- redrawing per row would be testing three different vectors.\n")
    a_fixed = null_dir()                       # drawn ONCE, outside the loop, on purpose
    print(f"  {'a = null / scale':>22}{'<a^dag a>_0':>14}{'tr J':>10}{'||J^2 - 2J||':>16}")
    scale_res = []
    for scale in (1.0, np.sqrt(2.0), 2.0):
        a = a_fixed/scale                      # same a_fixed every row
        A = M(a)
        J = A @ A.conj().T
        res = np.abs(J @ J - 2*J).max()
        scale_res.append((scale, res))
        print(f"  {'scale = ' + f'{scale:.4f}':>22}{born0(a):>14.6f}"
              f"{np.real(np.trace(J)):>10.6f}{res:>16.3e}")
    # the claim is EXCLUSIVE: sqrt(2) lands on 2J and the other two must NOT
    s6_scale = (scale_res[1][1] < TOL_F64
                and scale_res[0][1] > 1e-3 and scale_res[2][1] > 1e-3)
    print(f"\n  only the sqrt(2) row is on 2J, the other two are decisively off: {ok(s6_scale)}")
    print("  >> Only scale = sqrt(2), i.e. <a^dag a>_0 = 1, lands on 2J. The other two")
    print("     land on tr(J) J for their own tr. CH is fine with all of them; only the")
    print("     NORMALISED one reads 2.")

    print("\n  Now normalise properly (a -> a/sqrt(<a^dag a>_0)) and read every consequence:\n")
    print(f"  {'trial':>6}{'herm?':>7}{'det J':>10}{'rank':>6}{'tr J':>9}"
          f"{'||J^2 - 2J||':>15}{'<J^dag J>_0':>13}")
    s6 = dag_native and s6_scale
    for t in range(5):
        a = null_dir()
        a = a/np.sqrt(born0(a))
        A = M(a)
        J = A @ A.conj().T                     # same A twice. J = a a^dag.
        herm = np.allclose(J, J.conj().T, atol=TOL_F64)
        det = abs(np.linalg.det(J))
        rk = rank_of(J)
        trJ = np.real(np.trace(J))
        res = np.abs(J @ J - 2*J).max()
        bb = born0_mat(J)
        good = (herm and det < TOL_F64 and rk == 1
                and abs(trJ - 2) < TOL_F64 and res < TOL_F64 and abs(bb - 2) < TOL_F64)
        s6 = s6 and good
        print(f"  {t:>6}{str(herm):>7}{det:>10.2e}{rk:>6}{trJ:>9.6f}"
              f"{res:>15.3e}{bb:>13.6f}")
    print("\n  every step forced, none chosen:")
    print("    J = a a^dag Hermitian   <- the dagger did it (M(q^dag) = M(q)^dag)")
    print("    det J = |det a|^2 = 0   <- a is null")
    print("    rank J = rank a = 1     <- a is a zero divisor, not zero")
    print("    tr J = 2                <- <a^dag a>_0 = 1, the Born normalisation")
    print("    J^2 = 2J                <- Cayley-Hamilton, given the two above")
    print("    <J^dag J>_0 = 2         <- J = 2 E with E an ORTHOGONAL projection of rank 1")
    print(f"\n  >> ||J^2 - 2J|| ~ 1e-16 throughout. Section 6: {ok(s6)}")
    results["6 forced chain (null + normalised)"] = s6

    # =======================================================================
    banner("7. NEGATIVE (N2). '<J^dag J>_0 = 2 rank' is FALSE off the Hermitian branch. Finding 6.")
    # =======================================================================
    print("  Section 6's last line is TRUE for a a^dag but does NOT generalise. J^2 = 2J")
    print("  makes J/2 IDEMPOTENT. An idempotent is a PROJECTION ALONG something; it is an")
    print("  ORTHOGONAL projection only if E^dag = E as well. det and trace do not see the")
    print("  difference. <J^dag J>_0 does.\n")
    w_hat = w/np.linalg.norm(w)
    obliqueness = abs(np.vdot(v/np.linalg.norm(v), w_hat))
    print(f"  the oblique E = v w^dag/(w^dag v):  |<v_hat, w_hat>| = {obliqueness:.6f}")
    print(f"  (1.0 would mean w parallel to v, i.e. E orthogonal after all). E^2 = E: "
          f"{ok(np.abs(E_obl @ E_obl - E_obl).max() < TOL_F64)}\n")
    print(f"  {'J = 2E':>28}{'rank':>6}{'E^dag=E?':>10}{'tr J':>8}{'det J':>10}"
          f"{'<J^dag J>_0':>13}{'2*rank':>8}{'verdict':>10}")
    rows = [("E = 0", np.zeros((2, 2), dtype=np.complex128)),
            ("E = ORTHOGONAL rank-1", E_orth),
            ("E = OBLIQUE rank-1", E_obl),
            ("E = I (rank 2)", I2)]
    eq_flags = []
    for nm, E in rows:
        J = 2*E
        rk = rank_of(J)
        is_h = bool(np.allclose(E, E.conj().T, atol=TOL_F64))
        bb = born0_mat(J)
        eq = bool(abs(bb - 2*rk) < 1e-9)
        eq_flags.append((nm, is_h, eq, bb, rk))
        print(f"  {nm:>28}{rk:>6}{str(is_h):>10}{np.real(np.trace(J)):>8.4f}"
              f"{abs(np.linalg.det(J)):>10.2e}{bb:>13.6f}{2*rk:>8}"
              f"{('EQUAL' if eq else 'EXCEEDS'):>10}")
    obl_row = [r for r in eq_flags if "OBLIQUE" in r[0]][0]
    n2_shown = (not obl_row[2]) and obl_row[3] > 2.0 and (not obl_row[1])
    print(f"\n  >> the OBLIQUE row gives <J^dag J>_0 = {obl_row[3]:.4f}, not 2. It is rank 1,")
    print("     tr 2, det 0, and J^2 = 2J -- it passes every test section 4's repaired")
    print("     biconditional applies -- and the Born read is still wrong. The exact number")
    print("     is draw-dependent (the source note quoted 2.1693 for a different seed);")
    print("     what is invariant is that it EXCEEDS 2.")
    print("\n  The true statement:  <J^dag J>_0 = 2 tr(E^dag E) >= 2 rank(E),")
    print("                       equality iff E is an ORTHOGONAL projection.\n")
    ident_ok = True
    for nm, E in rows:
        J = 2*E
        lhs = born0_mat(J)
        rhs = 2*np.real(np.trace(E.conj().T @ E))
        ident_ok = ident_ok and abs(lhs - rhs) < 1e-9
    print(f"  identity <J^dag J>_0 == 2 tr(E^dag E) on all 4 rows: {ok(ident_ok)}")

    # CLOSED FORM. Independent re-derivation of the oblique number by a route the rest
    # of this script never takes, so it is not the same computation twice:
    #   E = v w^dag/(w^dag v)  =>  E^dag E = w (v^dag v) w^dag / |w^dag v|^2
    #   tr(E^dag E) = ||v||^2 ||w||^2 / |<w,v>|^2 = 1/|<v_hat, w_hat>|^2
    #   so  <J^dag J>_0 = 2/|<v_hat, w_hat>|^2, manifestly >= 2, equality iff w || v.
    cf = 2.0/obliqueness**2
    cf_ok = abs(cf - obl_row[3]) < 1e-6
    print(f"\n  closed form <J^dag J>_0 = 2/|<v_hat, w_hat>|^2 = {cf:.6f}")
    print(f"  numerically computed above                      = {obl_row[3]:.6f}")
    print(f"  the two routes agree: {ok(cf_ok)}")
    print("  >> this is the PROOF, not just a number: 2/|cos|^2 >= 2 with equality iff")
    print("     |cos| = 1 iff w is parallel to v iff E = v v^dag/||v||^2 is HERMITIAN.")
    print("     The Hermitian branch is exactly the equality case, not a lucky sample.")
    # sweep: the inequality, and equality iff Hermitian
    gaps, viol, false_eq = [], 0, 0
    for t in range(2000):
        vv = rng.normal(size=2) + 1j*rng.normal(size=2)
        ww = rng.normal(size=2) + 1j*rng.normal(size=2)
        den = ww.conj() @ vv
        if abs(den) < 1e-6:
            continue
        E = np.outer(vv, ww.conj())/den
        g = 2*np.real(np.trace(E.conj().T @ E)) - 2*1        # rank(E) = 1 by construction
        gaps.append(g)
        if g < -1e-9:
            viol += 1
        if abs(g) < 1e-9 and not np.allclose(E, E.conj().T, atol=1e-7):
            false_eq += 1
    sweep_ok = (viol == 0 and false_eq == 0)
    print(f"  sweep of {len(gaps)} random rank-1 idempotents (generically oblique):")
    print(f"    min  gap (<J^dag J>_0 - 2 rank) = {min(gaps):+.6e}")
    print(f"    mean gap                        = {float(np.mean(gaps)):+.6e}")
    print(f"    violations of the >= bound      = {viol}")
    print(f"    equality reached while E^dag != E = {false_eq}")
    if sweep_ok:
        print("\n  >> the bound holds and is saturated only on the Hermitian branch.")
    else:
        print(f"\n  >> the bound was VIOLATED {viol} times and/or equality was reached off")
        print(f"     the Hermitian branch {false_eq} times. The claimed identity is WRONG.")
    s7 = n2_shown and ident_ok and sweep_ok and cf_ok
    print(f"  >> NEGATIVE N2 demonstrated and the repaired identity verified. Section 7: {ok(s7)}")
    results["7 negative: <J^dag J>_0 = 2 rank is FALSE"] = s7

    # =======================================================================
    banner("8. THE LADDER {0, 1, 2} IS COMPLETE, NOT TRUNCATED. Finding 7.")
    # =======================================================================
    print("  HONESTY FIRST. 'max rank over random 2x2 draws is 2' is NOT evidence for")
    print("  anything: np.linalg.matrix_rank on a (2,2) array is bounded by 2 by the SHAPE")
    print("  of the array. That assertion cannot fail, and a test that cannot fail is not a")
    print("  test. The real content of 'the ladder is COMPLETE, not truncated' is that")
    print("      M : C (x) H --> M_2(C)   is a BIJECTION.")
    print("  Surjective: every 2x2 matrix IS some biquaternion, so nothing of rank 3 is")
    print("  being hidden by the choice of rep. Injective: no biquaternion is lost. THEN")
    print("  'rank <= 2' is a statement about the algebra and not about an array shape.\n")

    # (a) M is a C-linear map C^4 -> M_2(C) == C^4. Build its 4x4 matrix and check rank 4.
    Mmat = np.column_stack([M(e).reshape(4) for e in (E0, EI, EJ, EK)])
    m_rank = int(np.linalg.matrix_rank(Mmat, tol=1e-9))
    bij = (m_rank == 4)
    print(f"  (a) M as a C-linear map C^4 -> C^4 has rank {m_rank} (need 4): {ok(bij)}")

    # (b) surjectivity: round-trip an ARBITRARY 2x2 matrix through matrix_to_biquat.
    surj = True
    for _ in range(2000):
        A = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
        surj = surj and np.allclose(M(M_inv(A)), A, atol=TOL_F64)
    print(f"  (b) M(M_inv(A)) == A for 2000 arbitrary A in M_2(C) (SURJECTIVE): {ok(surj)}")

    # (c) injectivity: round-trip an arbitrary biquaternion the other way.
    inj = True
    for _ in range(2000):
        q = rand_bq()
        inj = inj and np.allclose(M_inv(M(q)), q, atol=TOL_F64)
    print(f"  (c) M_inv(M(q)) == q for 2000 arbitrary biquaternions (INJECTIVE):  {ok(inj)}")
    print("\n  >> M is a bijection. So the carrier IS M_2(C) exactly: no bigger object is")
    print("     available to have rank 3, and no biquaternion is unreachable.")

    # (d) NOW the rank census means something: draw through the ALGEBRA, not raw arrays.
    print("\n  Rank census drawn THROUGH the algebra (J = M(q), q a random biquaternion):\n")
    ranks = [rank_of(M(rand_bq())) for _ in range(2000)]
    seen = sorted(set(ranks))
    mx = max(ranks)
    print(f"  {'rank':>8}{'count over 2000 random biquaternions':>44}")
    for r in (0, 1, 2, 3):
        print(f"  {r:>8}{ranks.count(r):>44}")
    print(f"\n  max rank over 2000 biquaternion draws = {mx}   ranks seen = {seen}")
    print("  (generic draws are rank 2; rank 0 and 1 are measure zero, hence absent here.")
    print("   They are exhibited explicitly in section 7's table instead.)")

    # (e) the solution set of J^2 = 2J really is the ladder: verify each rung is realised
    #     and that no idempotent of any other rank exists.
    rung_ranks = sorted({rank_of(J) for _, J in fwd_rows})
    rungs_ok = (rung_ranks == [0, 1, 2])
    print(f"\n  ranks realised by the solutions of J^2 = 2J in section 4: {rung_ranks}")
    print(f"  they are exactly the rungs {{0, 1, 2}}, each one realised: {ok(rungs_ok)}")

    s8 = bij and surj and inj and (mx == 2) and rungs_ok
    print("\n  >> the ladder is complete because M is a BIJECTION onto M_2(C), not because")
    print("     a 2x2 array cannot report rank 3. The three idempotent classes of section")
    print("     4/7 -- 0, 2E_rank1, 2I -- are exactly the three rungs, each realised. The")
    print(f"     solution set of J^2 = 2J IS the rank ladder. Section 8: {ok(s8)}")
    results["8 rank ladder {0,1,2} complete"] = s8

    # =======================================================================
    banner("9. CROSS-CHECK AGAINST cal.biquaternion -- TOLERANCE IS FLOAT32-LIMITED")
    # =======================================================================
    print(f"  cal.CDTYPE = {CDTYPE}. eps ~ 1e-7. The findings above are complex128 and are")
    print(f"  quoted at ~1e-16; this section only checks that the complex128 mirror above")
    print(f"  IS cal's arithmetic, at rtol = {RTOL_C64:g}, atol = {ATOL_C64:g}. It does NOT")
    print("  and CANNOT re-verify a 1e-15 identity through complex64.\n")
    qs = [rand_bq() for _ in range(40)]
    ps = [rand_bq() for _ in range(40)]
    checks = []

    agree = all(np.allclose(quat_mul(to_t(p), to_t(q)).numpy(), qmul(p, q),
                            rtol=RTOL_C64, atol=ATOL_C64) for p, q in zip(ps, qs))
    checks.append(("cal.quat_mul         == this script's qmul", agree))

    agree = all(np.allclose(hermitian_conj(to_t(q)).numpy(), hdag(q),
                            rtol=RTOL_C64, atol=ATOL_C64) for q in qs)
    checks.append(("cal.hermitian_conj   == this script's hdag", agree))

    agree = all(np.allclose(biquat_to_matrix(to_t(q)).numpy(), M(q),
                            rtol=RTOL_C64, atol=ATOL_C64) for q in qs)
    checks.append(("cal.biquat_to_matrix == this script's M", agree))

    agree = all(np.allclose(quat_norm_sq(to_t(q)).item(),
                            np.linalg.det(M(q)), rtol=RTOL_C64, atol=ATOL_C64) for q in qs)
    checks.append(("cal.quat_norm_sq     == det M(q)", agree))

    # section 8 leans on M_inv, so the mirror of matrix_to_biquat must be checked too
    agree = all(np.allclose(matrix_to_biquat(
        torch.tensor(M(q), dtype=CDTYPE)).numpy(), M_inv(M(q)),
        rtol=RTOL_C64, atol=ATOL_C64) for q in qs)
    checks.append(("cal.matrix_to_biquat == this script's M_inv (sec 8)", agree))

    agree = all(np.allclose(np.trace(biquat_to_matrix(to_t(q)).numpy()),
                            2*q[0], rtol=RTOL_C64, atol=ATOL_C64) for q in qs)
    checks.append(("tr(cal M(q))         == 2 q0           (finding 1)", agree))

    agree = all(np.allclose(biquat_to_matrix(hermitian_conj(to_t(q))).numpy(),
                            biquat_to_matrix(to_t(q)).numpy().conj().T,
                            rtol=RTOL_C64, atol=ATOL_C64) for q in qs)
    checks.append(("cal M(q^dag)         == cal M(q)^dag   (finding 5)", agree))

    agree = all(np.allclose(quat_mul(hermitian_conj(to_t(q)), to_t(q)).numpy()[0].real,
                            born0(q), rtol=RTOL_C64, atol=ATOL_C64) for q in qs)
    checks.append(("cal <q^dag q>_0      == sum |q_mu|^2", agree))

    # handedness, through cal itself
    hand = (np.allclose(quat_mul(to_t(EI), to_t(EJ)).numpy(), EK, atol=ATOL_C64)
            and np.allclose(quat_mul(quat_mul(to_t(EI), to_t(EJ)), to_t(EK)).numpy(),
                            -E0, atol=ATOL_C64))
    checks.append(("cal.quat_mul is RIGHT-handed: ij=k, ijk=-1", hand))

    print(f"  {'check':<52}{'':>8}")
    for nm, good in checks:
        print(f"  {nm:<52}{ok(good):>8}")
    s9a = all(g for _, g in checks)

    print("\n  Now the honest part: what the SAME identities look like when evaluated IN")
    print("  complex64. These are cancelling differences; float32 cannot resolve them.\n")
    print(f"  {'quantity':>40}{'complex128':>14}{'cal complex64':>16}")
    a = null_dir()
    a = a/np.sqrt(born0(a))
    A64 = M(a)
    J64 = A64 @ A64.conj().T
    at = to_t(a)
    Jt = quat_mul(at, hermitian_conj(at))
    Jm = biquat_to_matrix(Jt).numpy()
    print(f"  {'|N(a)| for a null       (want 0)':>40}"
          f"{abs(a[0]**2+a[1]**2+a[2]**2+a[3]**2):>14.2e}"
          f"{abs(quat_norm_sq(at).item()):>16.2e}")
    print(f"  {'|det J|                 (want 0)':>40}{abs(np.linalg.det(J64)):>14.2e}"
          f"{abs(np.linalg.det(Jm)):>16.2e}")
    print(f"  {'|tr J - 2|              (want 0)':>40}"
          f"{abs(np.trace(J64) - 2):>14.2e}{abs(np.trace(Jm) - 2):>16.2e}")
    print(f"  {'||J^2 - 2J||            (want 0)':>40}"
          f"{np.abs(J64 @ J64 - 2*J64).max():>14.2e}"
          f"{np.abs(Jm @ Jm - 2*Jm).max():>16.2e}")
    ch64 = np.abs(Jm @ Jm - np.trace(Jm)*Jm + np.linalg.det(Jm)*I2).max()
    print(f"  {'||CH residual||         (want 0)':>40}"
          f"{np.abs(J64 @ J64 - np.trace(J64)*J64 + np.linalg.det(J64)*I2).max():>14.2e}"
          f"{ch64:>16.2e}")
    agree_J = np.allclose(Jm, J64, rtol=RTOL_C64, atol=ATOL_C64)
    floor_ok = np.abs(Jm @ Jm - 2*Jm).max() < 1e-4
    print(f"\n  cal's J agrees with the complex128 J at rtol {RTOL_C64:g}: {ok(agree_J)}")
    print(f"  cal's ||J^2 - 2J|| sits at the float32 noise floor (< 1e-4): {ok(floor_ok)}")
    print("\n  >> the complex64 column is 1e-7-ish, NOT 1e-16, and that is expected, not a")
    print("     defect: det J = sum J_mu^2 = 0 is a CANCELLATION and float32 has ~7 digits.")
    print("     The findings are the complex128 column. This section certifies only that")
    print("     the complex128 mirror is cal's arithmetic, to the precision cal has.")
    s9 = s9a and agree_J and floor_ok
    print(f"\n  Section 9: {ok(s9)}")
    results["9 cross-check vs cal (float32-limited)"] = s9

    # =======================================================================
    banner("SUMMARY")
    # =======================================================================
    for nm, good in results.items():
        print(f"  {nm:<44}{ok(good):>8}")
    print()
    print("  POSITIVE, and all of it forced by [H:C] = 2:")
    print("    tr M(q) = 2 q0; det M(q) = N(q); Cayley-Hamilton is QUADRATIC because the")
    print("    algebra has degree 2 -- the same 2 as rank <= 2. tr(a a^dag) = 2 <a^dag a>_0")
    print("    exactly, so 'tr J = 2' is the Born normalisation and not an extra input. The")
    print("    dagger is native (M(q^dag) = M(q)^dag), so J = a a^dag is Hermitian for free.")
    print("    a null + normalised => det 0, rank 1, tr 2, J^2 = 2J, <J^dag J>_0 = 2, with")
    print("    no free choices anywhere in the chain.")
    print()
    print("  NEGATIVE N1 -- finding 3 is WRONG AS STATED and is NOT softened here:")
    print("    'J^2 = 2J <=> det J = 0 AND tr J = 2' fails left-to-right at J = 0 (tr 0) and")
    print("    at J = 2I (det 4, the maximally NON-null element). J^2 = 2J only says J/2 is")
    print("    idempotent, and idempotents come in all three ranks. The right-to-left half")
    print("    is a theorem. Repaired: J^2 = 2J AND rank J = 1 <=> det J = 0 AND tr J = 2.")
    print("    Consequence for the manuscript: J^2 = 2J cannot be used to ARGUE the current")
    print("    is null. The null cone is an input to the identity, not an output of it.")
    print()
    print("  NEGATIVE N2 -- '<J^dag J>_0 = 2 rank' is FALSE off the Hermitian branch:")
    print("    an OBLIQUE rank-1 idempotent has tr 2, det 0, J^2 = 2J, rank 1, and still")
    print(f"    reads <J^dag J>_0 = {obl_row[3]:.4f} > 2. Idempotence is not projection.")
    print("    True form: <J^dag J>_0 = 2 tr(E^dag E) >= 2 rank(E), equality iff E^dag = E.")
    print("    a a^dag lands on the Hermitian branch by construction, which is WHY the")
    print("    chain in section 6 closes -- not because 2*rank is a general fact.")
    print()
    print(f"  seed = {SEED}; findings in numpy complex128; cross-check vs cal at complex64,")
    print(f"  rtol = {RTOL_C64:g} (float32-limited, stated as such).")


if __name__ == "__main__":
    main()
