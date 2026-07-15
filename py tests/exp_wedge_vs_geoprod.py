"""
exp_wedge_vs_geoprod.py

"The source current is either a wedge or a geometric product."  That is too loose.
The two constructions land on DIFFERENT branches of the Cayley-Hamilton dichotomy,
and only one of them can carry J^2 = 2J.

  wedge   a^b = (ab - ba)/2   is a COMMUTATOR  ->  tr = 0  ->  W^2 = -det(W) I
                                                   -> nilpotent branch (W^2 = 0)
  geo prod  J = ab            tr J = 2 (ab)_0  ->  can reach the idempotent branch

The 2J condition needs care.  "J^2 = 2J <=> det J = 0 and tr J = 2" is NOT a
biconditional: the (<=) direction is Cayley-Hamilton and holds unconditionally,
but the (=>) direction FAILS at both ends of the rank ladder {0, 1, 2}, and it
fails on honest geometric products (a = b = sqrt(2) gives J = ab = 2I).  The
repaired statement, which sec. 4 verifies rather than asserts, is

      J^2 = 2J  AND  rank J = 1   <=>   tr J = 2 AND det J = 0
                                  <=>   (ab)_0 = 1 AND [a null OR b null].

This matches negative finding N1 of exp_cayley_hamilton_2J.py in this suite.

and the headline NEGATIVE: even on the idempotent branch, J = ab for a GENERIC
partner b is an OBLIQUE idempotent.  J^2 = 2J holds exactly while <J^dag J>_0
ranges over [2, inf).  2 is a FLOOR, not a consequence.  The algebra names the
partner that attains it: b = a^dag.

PRECISION.  cal is torch.complex64 (eps ~ 1e-7).  Every check here is a CANCELLING
difference (det J = sum J_mu^2 = 0 by cancellation; W^2 + det(W) I = 0 identically),
so complex64 cannot resolve them.  The findings are therefore computed in numpy
complex128 using cal's OWN formulas, transcribed literally below.  Section 8 then
calls the real cal.biquaternion functions on the same inputs and confirms agreement
at a tolerance that is honest for complex64 (rtol 1e-5).  No 1e-15 identity is ever
asserted through complex64.

Standalone:  python exp_wedge_vs_geoprod.py     (from the "py tests" directory)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cal.biquaternion import (biquat_to_matrix, hermitian_conj, quat_mul,
                              quat_norm_sq)

RULE = "=" * 78
SEED = 20260715
rng = np.random.default_rng(SEED)
I2 = np.eye(2, dtype=np.complex128)
RESULTS = []


def section(title):
    print("\n" + RULE)
    print(title)
    print(RULE)


def verdict(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"\n  [{'PASS' if ok else 'FAIL'}] {name}")


# -- numpy complex128 mirror of cal.biquaternion (cal's formulas, verbatim) -----

def qmul(P, Q):
    """cal.biquaternion.quat_mul, Eq. 3, in complex128.  Right-handed: i j = k."""
    p0, p1, p2, p3 = P
    q0, q1, q2, q3 = Q
    return np.array([p0*q0 - p1*q1 - p2*q2 - p3*q3,
                     p0*q1 + p1*q0 + p2*q3 - p3*q2,
                     p0*q2 - p1*q3 + p2*q0 + p3*q1,
                     p0*q3 + p1*q2 - p2*q1 + p3*q0], dtype=np.complex128)


def qmat(q):
    """cal.biquaternion.biquat_to_matrix, in complex128.  h = sqrt(-1)."""
    h = 1j
    q0, q1, q2, q3 = q
    return np.array([[q0 + h*q3,  q1 + h*q2],
                     [-q1 + h*q2, q0 - h*q3]], dtype=np.complex128)


def qdag(q):
    """cal.biquaternion.hermitian_conj, in complex128."""
    c = np.conj(q)
    return np.array([c[0], -c[1], -c[2], -c[3]], dtype=np.complex128)


def qnorm_sq(q):
    """cal.biquaternion.quat_norm_sq, in complex128.  = det M(q)."""
    return q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2


# -- derived helpers -----------------------------------------------------------

def born(q):
    """<q^dag q>_0 = sum_mu |q_mu|^2 = Re tr(M^dag M)/2.  The Born closure."""
    return float(np.sum(np.abs(np.asarray(q))**2))


def born_mat(M):
    return float(np.real(np.trace(M.conj().T @ M)) / 2)


def qwedge(a, b):
    """a ^ b = (ab - ba)/2.  SAME a and SAME b in both terms (no sandwich bug)."""
    return (qmul(a, b) - qmul(b, a)) / 2


def rand_bq():
    return (rng.normal(size=4) + 1j*rng.normal(size=4)).astype(np.complex128)


def null_bq():
    """A GENERIC zero divisor: q0^2 + q1^2 + q2^2 + q3^2 = 0, q^dag != q.

    Pick q0 and a direction u in C^3, then scale u so the bilinear u.u kills q0^2.
    (The 1 + h*n family used in the source notes is SELF-ADJOINT, q^dag = q, so it
    sits exactly on the Hermitian point and would hide the obliqueness of sec. 6.)
    """
    q0 = complex(rng.normal(), rng.normal())
    u = rng.normal(size=3) + 1j*rng.normal(size=3)
    s = np.sqrt(-q0**2 / np.sum(u*u))
    v = s*u
    return np.array([q0, v[0], v[1], v[2]], dtype=np.complex128)


def mat_to_q(M):
    """cal.biquaternion.matrix_to_biquat, in complex128. The rep is ONTO M_2(C),
    so EVERY 2x2 complex matrix is some biquaternion -- used in sec. 4 to build a
    geo-product counterexample at rank 0."""
    h = 1j
    m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return np.array([(m00 + m11)/2, (m01 - m10)/2,
                     (m01 + m10)/(2*h), (m00 - m11)/(2*h)], dtype=np.complex128)


def rank_of(M):
    return int(np.linalg.matrix_rank(M, tol=1e-9))


def cfmt(z, nd=3):
    z = complex(z)
    return f"{z.real:.{nd}e}{z.imag:+.{nd}e}j"


print(RULE)
print("exp_wedge_vs_geoprod.py -- wedge (nilpotent branch) vs geo prod (idempotent)")
print(RULE)
print(f"  seed                : numpy default_rng({SEED})")
print( "  findings dtype      : numpy complex128 (cal's formulas, transcribed)")
print(f"  cal dtype           : {torch.complex64} (cross-checked in sec. 8, rtol 1e-5)")


# =============================================================================
section("1. HANDEDNESS GUARD -- i j = k, i j k = -1, on the basis used below")
# =============================================================================
e_i = np.array([0, 1, 0, 0], dtype=np.complex128)
e_j = np.array([0, 0, 1, 0], dtype=np.complex128)
e_k = np.array([0, 0, 0, 1], dtype=np.complex128)
one = np.array([1, 0, 0, 0], dtype=np.complex128)

ij = qmul(e_i, e_j)
ijk = qmul(qmul(e_i, e_j), e_k)
ii = qmul(e_i, e_i)
h_ok = [("i j == k", np.allclose(ij, e_k)),
        ("i i == -1", np.allclose(ii, -one)),
        ("i j k == -1", np.allclose(ijk, -one)),
        ("M(i) M(j) == M(k)  (rep is a homomorphism)",
         np.allclose(qmat(e_i) @ qmat(e_j), qmat(e_k))),
        ("M(i)M(j)M(k) == -I", np.allclose(qmat(e_i) @ qmat(e_j) @ qmat(e_k), -I2))]
for nm, ok in h_ok:
    print(f"  {nm:<44} {str(ok):>6}")

q_spot = rand_bq()
tr_ok = np.isclose(np.trace(qmat(q_spot)), 2*q_spot[0])
det_ok = np.isclose(np.linalg.det(qmat(q_spot)), qnorm_sq(q_spot))
born_ok = np.isclose(born(q_spot), born_mat(qmat(q_spot)))
print(f"\n  {'tr M(q) == 2 q_0':<44} {str(bool(tr_ok)):>6}")
print(f"  {'det M(q) == quat_norm_sq(q)':<44} {str(bool(det_ok)):>6}")
print(f"  {'<q^dag q>_0 == Re tr(M^dag M)/2':<44} {str(bool(born_ok)):>6}")
print(f"  {'M(q^dag) == M(q)^dag':<44} "
      f"{str(bool(np.allclose(qmat(qdag(q_spot)), qmat(q_spot).conj().T))):>6}")
verdict("handedness guard", all(o for _, o in h_ok) and tr_ok and det_ok and born_ok)


# =============================================================================
section("2. FINDING 1 -- the wedge a^b = (ab - ba)/2 is TRACELESS, identically")
# =============================================================================
print("  A commutator has tr[a,b] = tr(ab) - tr(ba) = 0.  The trap here is the")
print("  SANDWICH BUG: (a b1 - b2 a)/2 with two DIFFERENT partners is not a")
print("  commutator at all and is not traceless.  Same a, same b, both terms.\n")
print(f"  {'trial':>6}{'tr(a^b)':>28}{'|tr|':>12}{'traceless?':>12}")
tl_ok = True
for t in range(6):
    a, b = rand_bq(), rand_bq()          # bound once, used twice each
    W = qwedge(a, b)
    Wm = qmat(W)
    tr = np.trace(Wm)
    ok = abs(tr) < 1e-12
    tl_ok = tl_ok and ok
    print(f"  {t:>6}{cfmt(tr):>28}{abs(tr):>12.2e}{str(ok):>12}")

a, b = rand_bq(), rand_bq()
bad = (qmul(a, rand_bq()) - qmul(rand_bq(), a)) / 2   # the bug, shown deliberately
good = qwedge(a, b)
print(f"\n  the SANDWICH BUG, for contrast (two different partners, NOT a commutator):")
print(f"    tr of (a b1 - b2 a)/2  = {cfmt(np.trace(qmat(bad)))}   <- not traceless")
print(f"    tr of (a b  - b  a)/2  = {cfmt(np.trace(qmat(good)))}   <- traceless")
print("\n  >> The wedge is traceless in every trial. It is also PURE VECTOR: the")
print("     commutator of two biquaternions is exactly the cross product a_vec x b_vec.")
xprod = np.array([0,
                  a[2]*b[3] - a[3]*b[2],
                  a[3]*b[1] - a[1]*b[3],
                  a[1]*b[2] - a[2]*b[1]], dtype=np.complex128)
xp_ok = np.allclose(good, xprod, atol=1e-12)
print(f"     ||a^b - (0, a_vec x b_vec)|| = {np.abs(good - xprod).max():.3e}   match: {xp_ok}")
verdict("finding 1: wedge is traceless (tr = 0 identically)", tl_ok and xp_ok)


# =============================================================================
section("3. FINDING 2 -- Cayley-Hamilton: W^2 = -det(W) I. A wedge is NEVER 2W.")
# =============================================================================
print("  CH for 2x2:  W^2 = tr(W) W - det(W) I.  With tr(W) = 0:  W^2 = -det(W) I.")
print("  So W^2 = 2W would force -det(W) I = 2W, i.e. W proportional to I; but")
print("  tr(W) = 0 then forces W = 0.  The ONLY wedge with W^2 = 2W is W = 0.\n")
print(f"  {'trial':>6}{'|det W|':>12}{'||W^2 + det(W) I||':>21}{'||W^2 - 2W||':>15}{'nilpotent?':>12}")
ch_ok = True
ch_worst = 0.0
for t in range(4):
    a, b = rand_bq(), rand_bq()
    Wm = qmat(qwedge(a, b))
    d = np.linalg.det(Wm)
    ch = np.abs(Wm @ Wm + d*I2).max()
    two = np.abs(Wm @ Wm - 2*Wm).max()
    ch_worst = max(ch_worst, ch)
    ok = ch < 1e-10
    ch_ok = ch_ok and ok and (two > 1e-6)
    print(f"  {t:>6}{abs(d):>12.4f}{ch:>21.3e}{two:>15.4f}"
          f"{str(bool(np.abs(Wm @ Wm).max() < 1e-10)):>12}")
print(f"\n  >> ||W^2 + det(W) I|| <= {ch_worst:.1e} (worst of the {4} trials, i.e. rounding")
print("     noise on O(10) entries): the CH identity is exact.")
print("     ||W^2 - 2W|| is O(1) every time: a wedge is never an idempotent double.")

print("\n  --- does a NULL FACTOR make the wedge nilpotent? Testing the loose gloss ---")
print("  det(a^b) = (a_vec x b_vec).(a_vec x b_vec) = (a.a)(b.b) - (a.b)^2  [Lagrange].")
print("  'a is null' means a_0^2 + a.a = 0, which is NOT a.a = 0. So there is no")
print("  reason for det(a^b) to vanish. Check it rather than assume it:\n")
print(f"  {'trial':>6}{'det a (null)':>14}{'|det(a^b)|':>14}{'W^2 = 0?':>11}")
null_factor_nilp = []
for t in range(4):
    a, b = null_bq(), rand_bq()
    Wm = qmat(qwedge(a, b))
    d = np.linalg.det(Wm)
    nilp = bool(np.abs(Wm @ Wm).max() < 1e-8)
    null_factor_nilp.append(nilp)
    print(f"  {t:>6}{abs(qnorm_sq(a)):>14.2e}{abs(d):>14.4f}{str(nilp):>11}")
gloss_false = not any(null_factor_nilp)
print(f"\n  >> A null FACTOR does NOT make the wedge nilpotent: det(a^b) is O(1).")
print("     'the wedge lands on the nilpotent branch' is true only when the WEDGE")
print("     ITSELF is null (det W = 0), which is a codimension-1 condition on the")
print("     pair, not a consequence of one factor being a zero divisor. Reported as")
print("     a sharpening of the finding, not a contradiction of it.")

print("\n  --- the wedge ON the null cone (det W = 0) really is nilpotent ---")
print("  Construct it: a_vec = lam*(1,0,0), b_vec = (b1, h, 1)  =>  a x b = lam*(0,-1,h),")
print("  whose bilinear self-product is lam^2*(1 + h^2) = 0. Scalars a_0, b_0 are free.\n")
print(f"  {'trial':>6}{'|det W|':>12}{'||W^2||':>12}{'tr W':>22}{'W^2 = 0?':>11}")
nilp_ok = True
for t in range(3):
    lam = complex(rng.normal(), rng.normal())
    a = np.array([complex(rng.normal(), rng.normal()), lam, 0, 0], dtype=np.complex128)
    b = np.array([complex(rng.normal(), rng.normal()),
                  complex(rng.normal(), rng.normal()), 1j, 1], dtype=np.complex128)
    W_null = qwedge(a, b)
    Wm = qmat(W_null)
    d = np.linalg.det(Wm)
    nn = np.abs(Wm @ Wm).max()
    ok = (abs(d) < 1e-12) and (nn < 1e-12)
    nilp_ok = nilp_ok and ok
    print(f"  {t:>6}{abs(d):>12.3e}{nn:>12.3e}{cfmt(np.trace(Wm)):>22}{str(ok):>11}")
W_NULL = W_null      # keep the last one for the summary table
print("\n  >> det W = 0 => W^2 = 0 to machine precision. The wedge on the null cone is")
print("     NILPOTENT. The geo prod (sec. 4) is IDEMPOTENT. Different branches.")
verdict("finding 2: W^2 = -det(W) I; never 2W; nilpotent iff det W = 0",
        ch_ok and nilp_ok)
verdict("finding 2 sharpened: a null FACTOR alone does NOT make W nilpotent",
        gloss_false)


# =============================================================================
section("4. FINDING 3 -- the GEOMETRIC PRODUCT can give J^2 = 2J. Exactly when.")
# =============================================================================
print("  J = a b.  det(ab) = det(a) det(b) and tr(ab) = 2 (ab)_0.")
print("  CH gives ONE direction unconditionally:")
print("      tr J = 2 AND det J = 0  ==>  J^2 = tr(J) J - det(J) I = 2J.")
print("  The CONVERSE IS FALSE. Do not write '<=>' here. The counterexamples are")
print("  honest geometric products, not exotica, and they sit at the two ENDS of")
print("  the rank ladder {0, 1, 2} -- the middle rung is the only one that works.\n")
a_null = null_bq()
b_gen = rand_bq()

# -- the (=>) direction, tested rather than assumed. No rng draws here, so the
#    stream (and every downstream number) is unaffected by this block.
ce_rt2 = np.array([np.sqrt(2), 0, 0, 0], dtype=np.complex128)   # NOT null: N = 2
J_r2 = qmul(ce_rt2, ce_rt2)                                     # = 2I, rank 2
# rank-0 end: pick b with image(M(b)) inside ker(M(a)), so M(a)M(b) = 0 exactly.
_, _, _Vh = np.linalg.svd(qmat(a_null))
_y = _Vh[-1].conj()                                             # M(a_null) _y = 0
b_ker = mat_to_q(np.outer(_y, np.array([1, 1j], dtype=np.complex128)))
J_r0 = qmul(a_null, b_ker)                                      # = 0, rank 0
print(f"  {'geo product a b':<30}{'rk':>4}{'||J^2-2J||':>13}{'tr J':>9}"
      f"{'det J':>9}{'RHS holds?':>12}")
print("  " + "-"*77)
ce_rows = [("a = b = sqrt(2)   -> J = 2I", ce_rt2, ce_rt2, J_r2),
           ("a null, b in ker a -> J = 0", a_null, b_ker, J_r0)]
ce_bad = []
for nm, aa, bb, JJ in ce_rows:
    Mc = qmat(JJ)
    idem = np.abs(Mc @ Mc - 2*Mc).max()
    rhs = (abs(qmul(aa, bb)[0] - 1) < 1e-9) and (abs(qnorm_sq(aa)) < 1e-9
                                                 or abs(qnorm_sq(bb)) < 1e-9)
    ce_bad.append(idem < 1e-10 and not rhs)     # J^2 = 2J yet RHS is FALSE
    print(f"  {nm:<30}{rank_of(Mc):>4}{idem:>13.2e}"
          f"{np.trace(Mc).real:>9.2f}{np.linalg.det(Mc).real:>9.2f}{str(rhs):>12}")
conv_false = all(ce_bad)
print("\n  >> BOTH satisfy J^2 = 2J EXACTLY while the right-hand side is FALSE. So")
print("     'J^2 = 2J <=> (ab)_0 = 1 AND [a null OR b null]' is NOT a biconditional;")
print("     the (=>) direction fails at rank 2 (J = 2I: no null factor, (ab)_0 = 2)")
print("     and at rank 0 (J = 0: a IS null, but (ab)_0 = 0). Writing 'iff' here")
print("     overreaches. This is negative finding N1 of exp_cayley_hamilton_2J.py.")
print("\n  REPAIRED, and this IS a biconditional:")
print("      J^2 = 2J  AND  rank J = 1   <=>   tr J = 2 AND det J = 0")
print("                                  <=>   (ab)_0 = 1 AND [a null OR b null].")
print("  rank J = 1 is what excludes the two ends of the ladder. Note (ab)_0 is")
print("  LINEAR in b, so rescaling b by 1/(ab)_0 imposes the trace condition")
print("  without touching the det condition. Build the rank-1 rung:\n")

s = qmul(a_null, b_gen)[0]
b_sc = b_gen / s                       # rescale so (a b)_0 = 1
J_geo = qmul(a_null, b_sc)
Jm = qmat(J_geo)
print(f"  a  = null element, quat_norm_sq(a) = {cfmt(qnorm_sq(a_null))}")
print(f"       det M(a) = {abs(np.linalg.det(qmat(a_null))):.3e}   rank M(a) = {rank_of(qmat(a_null))}")
print(f"  b  = generic, rescaled by 1/(ab)_0 with (ab)_0 = {cfmt(s)}")
print(f"\n  J = a b:")
rows4 = [("(J)_0            ", cfmt(J_geo[0])),
         ("tr J             ", cfmt(np.trace(Jm))),
         ("det J            ", f"{abs(np.linalg.det(Jm)):.3e}"),
         ("rank J           ", str(rank_of(Jm))),
         ("||J^2 - 2J||     ", f"{np.abs(Jm @ Jm - 2*Jm).max():.3e}"),
         ("||J^2||          ", f"{np.abs(Jm @ Jm).max():.3e}"),
         ("<J^dag J>_0      ", f"{born(J_geo):.6f}")]
for nm, v in rows4:
    print(f"    {nm}= {v}")
idem_exact = np.abs(Jm @ Jm - 2*Jm).max()
geo_ok = (idem_exact < 1e-12 and abs(np.trace(Jm) - 2) < 1e-10
          and rank_of(Jm) == 1)
print(f"\n  >> ||J^2 - 2J|| = {idem_exact:.3e} at rank {rank_of(Jm)}. The geometric product")
print("     of a null element with a suitably scaled partner IS an idempotent double.")
print("     The wedge cannot be (tr = 0 != 2); the geo prod can. 'Either wedge or geo")
print("     prod' is too loose: the 2J statement needs the GEO PROD specifically.")
verdict("finding 3: geo prod, null factor, (ab)_0 = 1 gives J^2 = 2J at rank 1",
        geo_ok)
verdict("finding 3 sharpened: the 'iff' is FALSE without rank J = 1 (N1)",
        conv_false)


# =============================================================================
section("5. FINDING 4 -- rank(ab) <= min(rank a, rank b): a ONE-WAY DOOR")
# =============================================================================
print("  A null element has rank 1 (det = 0, nonzero). Once it enters the product")
print("  the rank is capped at 1 and nothing downstream restores it. Contrast the")
print("  generic-generic rows, which stay at rank 2.\n")
print(f"  {'trial':>6}  {'a':>8}{'b':>8}{'rk a':>6}{'rk b':>6}{'rk ab':>7}{'<= min?':>9}"
      f"{'<J^dag J>_0':>14}")
rank_ok = True
cases5 = [("generic", "generic")]*2 + [("NULL", "generic")]*3 + [("NULL", "NULL")]*2
for t, (ka, kb) in enumerate(cases5):
    A = null_bq() if ka == "NULL" else rand_bq()
    B = null_bq() if kb == "NULL" else rand_bq()
    P = qmul(A, B)
    ra, rb, rp = rank_of(qmat(A)), rank_of(qmat(B)), rank_of(qmat(P))
    ok = rp <= min(ra, rb)
    rank_ok = rank_ok and ok
    print(f"  {t:>6}  {ka:>8}{kb:>8}{ra:>6}{rb:>6}{rp:>7}{str(ok):>9}{born(P):>14.4f}")
print("\n  >> rank(ab) <= min(rank a, rank b) holds in every row. With a null factor")
print("     the product is capped at rank 1: 'you have to return back down to rank-1'")
print("     is this inequality, and it is a one-way door. Rank 1 (det = 0) is exactly")
print("     the locus where J^2 = 2J can live.")
verdict("finding 4: rank(ab) <= min(rank a, rank b); null factor caps at rank 1",
        rank_ok)


# =============================================================================
section("6. FINDING 5 (NEGATIVE) -- the generic geo-prod solution is OBLIQUE.")
# =============================================================================
print("  J^2 = 2J means J/2 =: E is IDEMPOTENT. It does NOT mean E is an ORTHOGONAL")
print("  PROJECTION -- that needs E^dag = E as well. For any idempotent E,")
print("      tr(E^dag E) >= rank(E),  with EQUALITY IFF E is orthogonal,")
print("  and <J^dag J>_0 = 2 tr(E^dag E). So the Born weight is 2*rank ONLY on the")
print("  Hermitian branch. Inspect the sec.-4 solution:\n")
E = Jm / 2
obl_idem = np.abs(E @ E - E).max()
obl_gap = np.abs(E.conj().T - E).max()
obl_tr = float(np.real(np.trace(E.conj().T @ E)))
print(f"    ||E^2 - E||       = {obl_idem:.3e}     (idempotent: yes)")
print(f"    ||E^dag - E||     = {obl_gap:.6f}     (0 iff orthogonal)")
print(f"    rank E            = {rank_of(E)}")
print(f"    tr(E^dag E)       = {obl_tr:.6f}     (= rank iff orthogonal)")
print(f"    <J^dag J>_0       = {born(J_geo):.6f}     (= 2 iff orthogonal, rank 1)")
is_oblique = (obl_gap > 1e-3) and (born(J_geo) > 2 + 1e-3) and (obl_idem < 1e-10)
print(f"\n  >> OBLIQUE. J^2 = 2J holds EXACTLY and the Born weight is still"
      f" {born(J_geo):.4f}, not 2.")

print("\n  --- sweep b: J^2 = 2J stays exact while <J^dag J>_0 wanders ---\n")
print(f"  {'trial':>6}{'||J^2 - 2J||':>15}{'||E^dag - E||':>15}{'<J^dag J>_0':>14}{'>= 2?':>8}")
lo, hi = np.inf, 0.0
sweep_ok = True
for t in range(8):
    bb = rand_bq()
    bb = bb / qmul(a_null, bb)[0]
    Jt = qmul(a_null, bb)
    Jtm = qmat(Jt)
    res = np.abs(Jtm @ Jtm - 2*Jtm).max()
    bt = born(Jt)
    lo, hi = min(lo, bt), max(hi, bt)
    ok = (res < 1e-10) and (bt >= 2 - 1e-9)
    sweep_ok = sweep_ok and ok
    print(f"  {t:>6}{res:>15.3e}{np.abs(Jtm.conj().T - Jtm).max():>15.6f}"
          f"{bt:>14.4f}{str(bool(bt >= 2 - 1e-9)):>8}")
print(f"\n  range of <J^dag J>_0 over the sweep: [{lo:.4f}, {hi:.4f}]")
print("  >> J^2 = 2J is a whole FAMILY, and the Born weight ranges over [2, inf) on it.")
print("     2 is the FLOOR, never violated, attained only at the Hermitian point.")
print("     So 'J^2 = 2J' pins the algebra but does NOT pin the Born value to 2.")

print("\n  --- the algebra names the right partner: b = a^dag ---")
print("  M(a a^dag) = M(a) M(a)^dag is Hermitian PSD by construction, and det = 0")
print("  because a is null. Normalise by (a a^dag)_0 = sum|a_mu|^2, which is REAL")
print("  and positive, so the rescaling preserves hermiticity.\n")
a_d = qdag(a_null)
s_h = qmul(a_null, a_d)[0]
J_herm = qmul(a_null, a_d) / s_h
Jhm = qmat(J_herm)
Eh = Jhm / 2
h_rows = [("(a a^dag)_0 (real?)", f"{cfmt(s_h)}"),
          ("(a a^dag)_0 == sum|a_mu|^2", f"{str(bool(np.isclose(s_h, born(a_null))))}"),
          ("a^dag == a ?", f"{str(bool(np.allclose(a_d, a_null)))}  (generic null: no)"),
          ("tr J", f"{cfmt(np.trace(Jhm))}"),
          ("det J", f"{abs(np.linalg.det(Jhm)):.3e}"),
          ("rank J", f"{rank_of(Jhm)}"),
          ("||J^2 - 2J||", f"{np.abs(Jhm @ Jhm - 2*Jhm).max():.3e}"),
          ("||E^dag - E||", f"{np.abs(Eh.conj().T - Eh).max():.3e}   <- HERMITIAN"),
          ("tr(E^dag E)", f"{float(np.real(np.trace(Eh.conj().T @ Eh))):.10f}"),
          ("<J^dag J>_0", f"{born(J_herm):.10f}   <- BORN = 2")]
for nm, v in h_rows:
    print(f"    {nm:<28}= {v}")
herm_ok = (np.abs(Jhm @ Jhm - 2*Jhm).max() < 1e-10
           and np.abs(Eh.conj().T - Eh).max() < 1e-10
           and abs(born(J_herm) - 2.0) < 1e-9)
print(f"\n  >> CONTRAST, same null a, two partners:")
print(f"       b generic (scaled) : J^2 = 2J yes,  E^dag = E NO,   <J^dag J>_0 = {born(J_geo):.4f}")
print(f"       b = a^dag (scaled) : J^2 = 2J yes,  E^dag = E YES,  <J^dag J>_0 = {born(J_herm):.4f}")
print("     Both are geometric products of a null element. Only the dagger partner is")
print("     Hermitian, and only the Hermitian one reads Born = 2. The geo prod does not")
print("     supply that by itself; the DAGGER does. This is the negative, kept as one.")
verdict("finding 5a (NEGATIVE): generic geo-prod solution is OBLIQUE, Born > 2",
        is_oblique)
verdict("finding 5b: Born = 2 is a FLOOR on the J^2 = 2J family, never violated",
        sweep_ok)
verdict("finding 5c: b = a^dag is Hermitian and attains Born = 2 exactly", herm_ok)


# =============================================================================
section("7. FINDING 6 -- SUMMARY TABLE")
# =============================================================================
a_g1, b_g1 = rand_bq(), rand_bq()
a_g2, b_g2 = rand_bq(), rand_bq()
a_n2 = null_bq()
b_n2 = rand_bq()
table = [
    ("wedge a^b, generic a,b",        qwedge(a_g1, b_g1)),
    ("wedge a^b, a NULL",             qwedge(a_n2, b_n2)),
    ("wedge a^b, W itself null",      W_NULL),
    ("geo prod ab, generic a,b",      qmul(a_g2, b_g2)),
    ("geo prod ab, a NULL",           qmul(a_null, b_gen)),
    ("geo prod ab, a NULL, (ab)_0=1", J_geo),
    ("geo prod a a^dag, normalised",  J_herm),
]
print(f"  {'construction':<32}{'tr':>18}{'|det|':>10}{'rk':>4}"
      f"{'J^2=2J?':>9}{'J^2=0?':>8}{'<JdJ>_0':>11}")
print("  " + "-"*76)
idem_rows, born2_rows = [], []
for nm, q in table:
    M = qmat(q)
    tr = np.trace(M)
    trs = f"{tr.real:.2f}{tr.imag:+.2f}j"
    is_idem = bool(np.abs(M@M - 2*M).max() < 1e-8)
    if is_idem:
        idem_rows.append(nm)
    if abs(born(q) - 2.0) < 1e-6:
        born2_rows.append(nm)
    print(f"  {nm:<32}{trs:>18}{abs(np.linalg.det(M)):>10.3f}{rank_of(M):>4}"
          f"{str(is_idem):>9}"
          f"{str(bool(np.abs(M@M).max() < 1e-8)):>8}{born(q):>11.4f}")
# the prose below is CHECKED against the table, not asserted over it
t_two = (len(idem_rows) == 2)
t_geo = all(r.startswith("geo prod") for r in idem_rows)
t_nowedge = not any(r.startswith("wedge") for r in idem_rows)
t_born2 = (born2_rows == ["geo prod a a^dag, normalised"])
print(f"\n  J^2=2J rows: {len(idem_rows)} -> {idem_rows}")
print(f"    exactly two              : {t_two}")
print(f"    both are GEOMETRIC PRODS : {t_geo}")
print(f"    no wedge row is ever True: {t_nowedge}   (tr = 0 forbids it)")
print(f"    only a a^dag reads Born 2: {t_born2}   rows with Born = 2: {born2_rows}")
print("\n  That is the whole result in one table: the idempotent branch is reachable")
print("  only by a geometric product, and only the dagger partner reads Born = 2.")
verdict("finding 6: summary table supports every claim made about it",
        t_two and t_geo and t_nowedge and t_born2)


# =============================================================================
section("8. CROSS-CHECK vs cal.biquaternion (torch.complex64, FLOAT32-LIMITED)")
# =============================================================================
print("  The findings above are complex128. cal is complex64 (eps ~ 1.2e-7). This")
print("  section feeds the SAME inputs to the REAL cal functions and checks agreement")
print("  at rtol 1e-5 / atol 1e-5 -- a tolerance that is HONEST for float32, not the")
print("  1e-15 the findings are asserted at. Nothing here is claimed to 1e-15.\n")


def to_t(q):
    return torch.tensor(np.asarray(q, dtype=np.complex128), dtype=torch.complex64)


def agree(x_t, y_np, tol=1e-5):
    x = x_t.detach().numpy()
    y = np.asarray(y_np)
    return bool(np.allclose(x, y, rtol=tol, atol=tol)), float(np.abs(x - y).max())


checks = []
# handedness through the REAL cal.quat_mul. The diff column is MEASURED, not
# hardcoded -- a printed measurement must come from the arithmetic.
checks.append(("cal.quat_mul: i j == k",
               *agree(quat_mul(to_t(e_i), to_t(e_j)), e_k)))
checks.append(("cal.quat_mul: i j k == -1",
               *agree(quat_mul(quat_mul(to_t(e_i), to_t(e_j)), to_t(e_k)), -one)))

aX, bX = rand_bq(), rand_bq()
checks.append(("cal.quat_mul(a,b) vs c128 qmul", *agree(quat_mul(to_t(aX), to_t(bX)), qmul(aX, bX))))
checks.append(("cal wedge (ab-ba)/2 vs c128 qwedge",
               *agree((quat_mul(to_t(aX), to_t(bX)) - quat_mul(to_t(bX), to_t(aX)))/2,
                      qwedge(aX, bX))))
checks.append(("cal.biquat_to_matrix vs c128 qmat", *agree(biquat_to_matrix(to_t(aX)), qmat(aX))))
checks.append(("cal.hermitian_conj vs c128 qdag", *agree(hermitian_conj(to_t(aX)), qdag(aX))))
checks.append(("cal.quat_norm_sq vs c128 qnorm_sq",
               *agree(quat_norm_sq(to_t(aX)).reshape(1), np.array([qnorm_sq(aX)]))))
checks.append(("cal: M(q^dag) == M(q)^dag",
               *agree(biquat_to_matrix(hermitian_conj(to_t(aX))),
                      biquat_to_matrix(to_t(aX)).detach().numpy().conj().T)))
# the sec-4 and sec-6 objects, through cal
checks.append(("cal J_geo = quat_mul(a_null, b_sc)", *agree(quat_mul(to_t(a_null), to_t(b_sc)), J_geo)))
checks.append(("cal J_herm = quat_mul(a, a^dag)/s",
               *agree(quat_mul(to_t(a_null), hermitian_conj(to_t(a_null)))
                      / torch.tensor(complex(s_h), dtype=torch.complex64), J_herm)))
print(f"  {'check':<40}{'agree (rtol 1e-5)':>19}{'max abs diff':>15}")
print("  " + "-"*74)
xc_ok = True
for nm, ok, d in checks:
    xc_ok = xc_ok and ok
    print(f"  {nm:<40}{str(ok):>19}{d:>15.3e}")

print("\n  --- WHY the findings are not computed in cal's dtype ---")
print("  The same cancelling quantities, evaluated in complex64 vs complex128:\n")
print(f"  {'quantity':<40}{'complex64':>16}{'complex128':>16}")
print("  " + "-"*72)
nsq64 = float(abs(complex(quat_norm_sq(to_t(a_null)))))
nsq128 = float(abs(qnorm_sq(a_null)))
print(f"  {'|quat_norm_sq(a_null)|  (exact 0)':<40}{nsq64:>16.3e}{nsq128:>16.3e}")
Jm64 = biquat_to_matrix(quat_mul(to_t(a_null), to_t(b_sc))).detach().numpy().astype(np.complex128)
r64 = float(np.abs(Jm64 @ Jm64 - 2*Jm64).max())
r128 = float(np.abs(Jm @ Jm - 2*Jm).max())
print(f"  {'||J^2 - 2J||  (exact 0)':<40}{r64:>16.3e}{r128:>16.3e}")
Wm64 = biquat_to_matrix(to_t(W_NULL)).detach().numpy().astype(np.complex128)
w64 = float(np.abs(Wm64 @ Wm64).max())
w128 = float(np.abs(qmat(W_NULL) @ qmat(W_NULL)).max())
print(f"  {'||W^2||, W null  (exact 0)':<40}{w64:>16.3e}{w128:>16.3e}")
print("\n  >> complex64 floors these at ~1e-7, ~1e-6. Asserting 'det = 0 to 1e-15'")
print("     through cal's dtype would be a lie about the arithmetic. The findings are")
print("     complex128; cal agrees with them to float32 precision, which is all the")
print("     cross-check can honestly claim.")
verdict("cross-check: cal.biquaternion agrees at float32-limited tolerance", xc_ok)


# =============================================================================
section("9. VERDICT")
# =============================================================================
for nm, ok in RESULTS:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {nm}")
n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"\n  {len(RESULTS) - n_fail}/{len(RESULTS)} sections PASS, {n_fail} FAIL")
print("\n  SETTLED:")
print("    - The wedge is a commutator: tr = 0 identically, so W^2 = -det(W) I and")
print("      W^2 = 2W is impossible unless W = 0. On the null cone W^2 = 0: NILPOTENT.")
print("    - The geo prod reaches J^2 = 2J at rank 1 iff one factor is null and")
print("      (ab)_0 = 1: IDEMPOTENT. So 'either wedge or geo prod' is too loose --")
print("      only the geo prod reaches the idempotent branch.")
print("    - rank(ab) <= min(rank a, rank b): a null factor is a one-way door to rank 1.")
print("  SHARPENED:")
print("    - A null FACTOR does not make the wedge nilpotent; the wedge must itself be")
print("      null (det W = 0). Lagrange: det(a^b) = (a.a)(b.b) - (a.b)^2, and 'a null'")
print("      says a.a = -a_0^2, not a.a = 0.")
print("    - 'J^2 = 2J <=> tr J = 2 AND det J = 0' is NOT a biconditional. (<=) is CH")
print("      and holds always; (=>) fails at BOTH ends of the rank ladder, on honest")
print("      geo products: a = b = sqrt(2) gives J = 2I (rank 2, no null factor), and")
print("      a null with b in ker a gives J = 0 (rank 0). The 'iff' needs rank J = 1.")
print("  NEGATIVE (kept):")
print("    - J^2 = 2J does NOT deliver Born = 2. The generic geo-prod solution is an")
print("      OBLIQUE idempotent with <J^dag J>_0 > 2. The value ranges over [2, inf);")
print("      2 is a floor attained only at the Hermitian point b = a^dag.")
print("      The Born value comes from the DAGGER, not from the geometric product.")
print(RULE)

sys.exit(0 if n_fail == 0 else 1)
