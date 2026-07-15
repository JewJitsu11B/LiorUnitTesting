"""
The Z_4 double cover: exactly what the channel periods 2 and 4 prove, and what they do not.

Repeated integration acts on the two channels by two different scalars:
    real channel    by  1/(-u)  -> the factor is  -1     (order 2)
    complex channel by  1/(i w) -> the factor is 1/i = -i (order 4)
One group element g = "integrate once", two representations, orders 2 and 4, ratio 2.
2:1 is the double-cover ratio. The question this script settles is whether that is the
SAME 2 as the SU(2) -> SO(3) cover, or a numerical coincidence of two unrelated 2s.

CLAIMS VERIFIED (the confirmed half):
  Z1: <g> ~ Z_4. rho_real(g^n) = (-1)^n and rho_complex(g^n) = (-i)^n are both genuine
      homomorphisms of the SAME group, with orders 2 and 4. Ratio 2.
  Z2: ker(rho_real) = {1, g^2} ~ Z_2, so rho_real factors through Z_4/{+-1} ~ Z_2, while
      rho_complex is faithful on Z_4. The decisive line: rho_complex(g^2) = -1 while
      rho_real(g^2) = +1. The kernel element is the one the real rep cannot see.
      One rep faithful, one factoring through the quotient by {1, g^2}: that IS the
      covering relation, not an analogy to it. ({1, g^2} is NOT the centre of Z_4 --
      Z_4 is abelian, so its own centre is everything. It is the image in <g> of the
      centre {+-I} of the ambient SU(2). See section 2.)
  Z3: it is literally SU(2) -> SO(3) restricted to a cyclic subgroup. U = -i sigma_x =
      exp(-i pi sigma_x / 2) has order 4 in SU(2); its SO(3) image R_ab = (1/2)
      tr(sa U sb U^dag) has order 2; and U^2 = -I, the kernel element, maps to the
      identity downstairs. Identical structure to the channels: 4 upstairs, 2 downstairs,
      the invisible element is -1. CARE: cal's own i does NOT embed to U -- it embeds to
      M(i) = i sigma_y, a DIFFERENT order-4 element. <U> and <M(i)> meet only in {+-I}.
      They are CONJUGATE in SU(2) (checked in section 3), which is why the covering
      reading survives; and section C5 redoes the whole finding on cal's actual i.

CLAIMS REFUTED / BOUNDED (the negative, and it is the load-bearing half):
  Z4: this is the double cover's FINGERPRINT ON A CYCLIC SUBGROUP, not tensor/spinor
      windings. Three independent reasons, each checked rather than asserted:
        (a) no path. n is an INTEGER and Z_4 is FINITE (exactly 4 elements, verified).
            A winding number counts turns of a CONTINUOUS path; there is no path here,
            so there is nothing to wind. A winding needs a continuous parameter, which
            needs the fractional integral back.
        (b) no rep index. The channels are SCALAR amplitudes. Both reps are 1x1.
            Nothing rotates them.
        (c) the hard obstruction. SU(2) has NO nontrivial 1-dimensional representation:
            -I is a commutator [A,B] with A = i sigma_x, B = i sigma_y (verified), and
            any scalar rep chi kills every commutator, so chi(-I) = 1 forced. But
            rho_complex(g^2) = -1 != 1. Therefore rho_complex does NOT extend to SU(2).
            A scalar channel cannot be a spinor. Z3's subgroup inclusion is real; the
            extension to the whole group does not exist.
  Z5: the rate reading is real but it is NOT independent evidence. The complex channel
      advances |arg(-i)| = pi/2 per integration and the real channel |arg(-1)| = pi, so
      the complex is at HALF rate, the spinor-vs-vector relation in the right direction
      (the faithful/order-4 rep is the slow one, exactly as spin-1/2 is). But
      order * |step| = 2pi holds for BOTH channels (verified), so the rate ratio is the
      order ratio restated and carries no information beyond Z1. A rate ratio is not a
      rep index. The SIGN of the real channel's turn is a branch choice, not data.

PRECISION. The findings are computed in numpy complex128 using cal's own rep formulas,
written out inline. Section C then calls the real cal.biquaternion functions on the same
inputs. cal is torch.complex64 (eps ~ 1e-7), so the cross-check tolerance is rtol 1e-5
and is float32-limited by construction; no 1e-15 identity is ever asserted through
complex64. This experiment has no cancelling differences (the Z_4 values are 0 and +-1,
exactly representable), so complex64 is not stressed here -- the honest tolerance is
stated anyway rather than tightened after the fact.

Seeded and deterministic: np.random.default_rng(20260715).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from cal.biquaternion import (quat_mul, hermitian_conj, quat_norm_sq,
                              biquat_to_matrix)

RNG = np.random.default_rng(20260715)
np.set_printoptions(precision=6, suppress=True)

RULE = "=" * 78
TOL = 1e-12          # honest for numpy complex128
TOL_C64 = 1e-5       # honest for torch.complex64 (eps ~ 1e-7); float32-limited

RESULTS = {}


def verdict(key, ok):
    """Record and render a section verdict. Derived from data, never asserted."""
    RESULTS[key] = bool(ok)
    return "PASS" if ok else "FAIL"


def block(M, indent="     "):
    """Indent every line of an array repr, first line included."""
    return "\n".join(indent + ln for ln in np.array2string(M).splitlines())


# ---- cal's rep formulas, written out in numpy complex128 --------------------
# quat_mul (cal/biquaternion.py, paper Eq. 3), right-handed Hamilton rules.
def quat_mul_np(P, Q):
    p0, p1, p2, p3 = P
    q0, q1, q2, q3 = Q
    return np.array([p0*q0 - p1*q1 - p2*q2 - p3*q3,
                     p0*q1 + p1*q0 + p2*q3 - p3*q2,
                     p0*q2 - p1*q3 + p2*q0 + p3*q1,
                     p0*q3 + p1*q2 - p2*q1 + p3*q0], dtype=np.complex128)


# biquat_to_matrix (cal/biquaternion.py): M(q) with h = 1j.
def biquat_to_matrix_np(q):
    q0, q1, q2, q3 = q
    h = 1j
    return np.array([[q0 + h*q3, q1 + h*q2],
                     [-q1 + h*q2, q0 - h*q3]], dtype=np.complex128)


def quat_pow_np(q, n):
    r = np.array([1, 0, 0, 0], dtype=np.complex128)
    for _ in range(n):
        r = quat_mul_np(r, q)
    return r


ONE = np.array([1, 0, 0, 0], dtype=np.complex128)
QI = np.array([0, 1, 0, 0], dtype=np.complex128)
QJ = np.array([0, 0, 1, 0], dtype=np.complex128)
QK = np.array([0, 0, 0, 1], dtype=np.complex128)

I2 = np.eye(2, dtype=np.complex128)
I3 = np.eye(3, dtype=np.float64)
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI = (SX, SY, SZ)


def so3_image(V):
    """R_ab = (1/2) tr(sa V sb V^dag): the SO(3) image of V in SU(2)."""
    R = np.zeros((3, 3))
    for a, sa in enumerate(PAULI):
        for b, sb in enumerate(PAULI):
            R[a, b] = np.real(0.5*np.trace(sa @ V @ sb @ V.conj().T))
    return R


def order_of(step_fn, ident, atol, nmax=8):
    """Least n >= 1 with step_fn(n) == ident, or None."""
    for n in range(1, nmax + 1):
        if np.allclose(step_fn(n), ident, atol=atol):
            return n
    return None


# ============================================================================
print(RULE)
print("0. HANDEDNESS GUARD -- the basis this script actually uses")
print(RULE)
ij = quat_mul_np(QI, QJ)
ijk = quat_mul_np(ij, QK)
ij_is_k = np.allclose(ij, QK, atol=TOL)
ijk_is_m1 = np.allclose(ijk, -ONE, atol=TOL)
print("  numpy complex128, cal's Eq. 3 product written out inline:")
print(f"    i*j     = {np.real_if_close(ij)}   == k  ? {ij_is_k}")
print(f"    (i*j)*k = {np.real_if_close(ijk)}   == -1 ? {ijk_is_m1}")
i_sq = quat_mul_np(QI, QI)
i_sq_is_m1 = np.allclose(i_sq, -ONE, atol=TOL)
print(f"    i*i     = {np.real_if_close(i_sq)}   == -1 ? {i_sq_is_m1}")
print("  The convention is RIGHT-HANDED (i*j = k, ijk = -1), matching cal.")
print(f"\n  SECTION 0: {verdict('S0', ij_is_k and ijk_is_m1 and i_sq_is_m1)}")

# ============================================================================
print("\n" + RULE)
print("1. ONE GROUP ELEMENT g = 'INTEGRATE ONCE'. TWO REPS OF IT. ORDERS 2 AND 4.")
print(RULE)
print("  One integration multiplies:")
print("     real channel    by  1/(-u)  -> the factor is  -1")
print("     complex channel by  1/(i w) -> the factor is 1/i = -i")
print("  So g acts on the two channels by two different scalars. Their powers:\n")
print(f"  {'n':>4}{'rho_real(g^n)':>18}{'rho_complex(g^n)':>22}")
for n in range(5):
    print(f"  {n:>4}{str((-1)**n):>18}{str(np.round((-1j)**n, 6)):>22}")


def rho_real(n):
    return (-1.0 + 0j)**n


def rho_cplx(n):
    return (-1j)**n


ord_real = order_of(lambda n: rho_real(n), 1.0, TOL)
ord_cplx = order_of(lambda n: rho_cplx(n), 1.0, TOL)
# Both must be homomorphisms of the same Z_4, checked on the full multiplication table.
hom_r = all(np.isclose(rho_real((m + n) % 4), rho_real(m)*rho_real(n), atol=TOL)
            for m in range(4) for n in range(4))
hom_c = all(np.isclose(rho_cplx((m + n) % 4), rho_cplx(m)*rho_cplx(n), atol=TOL)
            for m in range(4) for n in range(4))
print(f"\n  order of g on the real channel    = {ord_real}")
print(f"  order of g on the complex channel = {ord_cplx}")
print(f"  ratio = {ord_cplx // ord_real}")
print(f"\n  rho_real  is a homomorphism on Z_4 (all 16 products)? {hom_r}")
print(f"  rho_cplx  is a homomorphism on Z_4 (all 16 products)? {hom_c}")
print("\n  >> ONE group element, TWO representations, orders 2 and 4. Not a coincidence")
print("     of periods: two reps of the same <g> = Z_4.")
ok1 = (ord_real == 2) and (ord_cplx == 4) and hom_r and hom_c
print(f"\n  SECTION 1: {verdict('S1', ok1)}   (expected orders 2 and 4, ratio 2)")

# ============================================================================
print("\n" + RULE)
print("2. THE REAL REP FACTORS THROUGH Z_4/{+-1}. THAT IS THE COVERING RELATION.")
print(RULE)
print(f"  {'n':>4}{'rho_complex(g^n)':>20}{'rho_real(g^n)':>18}{'in ker rho_real?':>20}")
ker_r, ker_c = [], []
for n in range(4):
    ink = bool(np.isclose(rho_real(n), 1.0, atol=TOL))
    if ink:
        ker_r.append(n)
    if np.isclose(rho_cplx(n), 1.0, atol=TOL):
        ker_c.append(n)
    print(f"  {n:>4}{str(np.round(rho_cplx(n), 4)):>20}"
          f"{str(int(np.real(rho_real(n)))):>18}{str(ink):>20}")
print(f"\n  ker(rho_real)    = {{g^n : n in {ker_r}}}, order {len(ker_r)}")
print(f"  ker(rho_complex) = {{g^n : n in {ker_c}}}, order {len(ker_c)}")
g2_c = rho_cplx(2)
g2_r = rho_real(2)
sees = np.isclose(g2_c, -1.0, atol=TOL)
blind = np.isclose(g2_r, 1.0, atol=TOL)
print(f"\n  THE DECISIVE LINE -- the kernel element g^2:")
print(f"    rho_complex(g^2) = {np.round(g2_c, 6)}   -> is -1 ? {bool(sees)}   (the complex rep SEES it)")
print(f"    rho_real(g^2)    = {np.round(g2_r, 6)}   -> is +1 ? {bool(blind)}   (the real rep is BLIND to it)")
print("\n  >> ker(rho_real) = {1, g^2} ~ Z_2, so rho_real factors through Z_4/Z_2 ~ Z_2.")
print("     rho_complex is faithful on all of Z_4. ONE rep faithful, ONE factoring")
print("     through the quotient by {1, g^2}. THAT IS the double-cover relation.")
print("     Wording care: {1, g^2} is NOT the centre of Z_4. Z_4 is ABELIAN, so its own")
print("     centre is all of Z_4 and the quotient by it is trivial. {1, g^2} is the")
print("     image in <g> of the centre {+-I} of the AMBIENT SU(2) (section 3). The")
print("     quotient that carries the cover is by that Z_2, not by Z_4's own centre.")
ok2 = (ker_r == [0, 2]) and (ker_c == [0]) and bool(sees) and bool(blind)
print(f"\n  SECTION 2: {verdict('S2', ok2)}   (ker(rho_real) = {{1, g^2}}, rho_complex faithful)")

# ============================================================================
print("\n" + RULE)
print("3. IT IS LITERALLY SU(2) -> SO(3) RESTRICTED TO A Z_4. NOT AN ANALOGY.")
print(RULE)
U = -1j*SX                              # = exp(-i pi sigma_x / 2)
print("  U = -i sigma_x = exp(-i pi sigma_x / 2), the SU(2) element for a pi rotation about x:")
print(block(U))
det_ok = np.isclose(np.linalg.det(U), 1.0, atol=TOL)
unit_ok = np.allclose(U @ U.conj().T, I2, atol=TOL)
print(f"\n  det U = {np.round(np.linalg.det(U), 6)}  -> in SU(2)? {bool(det_ok and unit_ok)}")
print(f"\n  {'n':>4}{'U^n':>14}{'U^n = I?':>12}{'SO(3) image = I?':>20}")
for n in range(1, 5):
    Vn = np.linalg.matrix_power(U, n)
    isI = np.allclose(Vn, I2, atol=TOL)
    isMI = np.allclose(Vn, -I2, atol=TOL)
    RisI = np.allclose(so3_image(Vn), I3, atol=TOL)
    tag = 'I' if isI else ('-I' if isMI else '(neither)')
    print(f"  {n:>4}{tag:>14}{str(isI):>12}{str(RisI):>20}")
ord_U = order_of(lambda n: np.linalg.matrix_power(U, n), I2, TOL)
ord_R = order_of(lambda n: so3_image(np.linalg.matrix_power(U, n)), I3, TOL)
U2 = U @ U
U2_is_mI = np.allclose(U2, -I2, atol=TOL)
R_of_U2 = so3_image(U2)
R_of_U2_is_I = np.allclose(R_of_U2, I3, atol=TOL)
R1 = so3_image(U)
R1_orth = np.allclose(R1 @ R1.T, I3, atol=TOL) and np.isclose(np.linalg.det(R1), 1.0, atol=TOL)
print(f"\n  order of U in SU(2)  = {ord_U}")
print(f"  order of R in SO(3)  = {ord_R}")
print(f"  R(U) is a proper rotation (R R^T = I, det = +1)? {bool(R1_orth)}")
print(f"  R(U) =\n{block(R1)}")
print(f"\n  U^2 = -I ? {bool(U2_is_mI)}     and its SO(3) image is I ? {bool(R_of_U2_is_I)}")

# U is NOT cal's i. Check the relation rather than asserting an identification.
Mi_np_i = biquat_to_matrix_np(QI)


def in_grp(X, elems):
    return any(np.allclose(X, G, atol=TOL) for G in elems)


grp_U = [np.linalg.matrix_power(U, n) for n in range(4)]
grp_M = [np.linalg.matrix_power(Mi_np_i, n) for n in range(4)]
same_elt = bool(np.allclose(U, Mi_np_i, atol=TOL))
same_sub = all(in_grp(X, grp_M) for X in grp_U)
overlap = sum(1 for X in grp_U if in_grp(X, grp_M))
# V = exp(-i pi sigma_z / 4): the SU(2) rotation by pi/2 about z, x-axis -> y-axis.
V = np.cos(np.pi/4)*I2 - 1j*np.sin(np.pi/4)*SZ
V_in_su2 = np.allclose(V @ V.conj().T, I2, atol=TOL) and np.isclose(np.linalg.det(V), 1.0, atol=TOL)
conj_ok = bool(V_in_su2 and all(in_grp(V @ X @ V.conj().T, grp_M) for X in grp_U))
print(f"\n  BUT U IS NOT cal's i, and that is checked here rather than waved through:")
print(f"     cal's i = (0,1,0,0) embeds to M(i) = i sigma_y, NOT to U = -i sigma_x:")
print(block(Mi_np_i))
print(f"     U == M(i) ?                        {same_elt}")
print(f"     <U> == <M(i)> as subgroups ?       {bool(same_sub)}   (they share only {overlap} of 4: {{+-I}})")
print(f"     V <U> V^dag == <M(i)>, V in SU(2)? {conj_ok}    (V = exp(-i pi sigma_z / 4))")
print(f"  So U and cal's i generate DISTINCT but CONJUGATE Z_4 subgroups of SU(2).")
print(f"  Conjugate subgroups have identical covering structure, so the reading below")
print(f"  stands -- but '<i> IS <U>' would be FALSE, and section C5 redoes the finding")
print(f"  on cal's actual i rather than on the hand-built U.")

print("\n  >> U has order 4 upstairs, its rotation has order 2 downstairs, and U^2 = -I")
print("     is the kernel element SO(3) cannot see. IDENTICAL structure to the channels:")
print("     4 upstairs, 2 downstairs, the invisible element is -1. <i> ~ Z_4 in the unit")
print("     quaternions is a subgroup of exactly this kind (conjugate to <U>, checked")
print("     above), so the Z_4/Z_2 of sections 1-2 is the spinor double cover restricted")
print("     to a cyclic subgroup. Real, and it matches sections 1-2 exactly.")
ok3 = ((ord_U == 4) and (ord_R == 2) and U2_is_mI and R_of_U2_is_I and R1_orth
       and det_ok and conj_ok)
print(f"\n  SECTION 3: {verdict('S3', ok3)}   (order 4 upstairs, order 2 downstairs, U^2 = -I invisible)")

# ============================================================================
print("\n" + RULE)
print("4. NEGATIVE -- THIS IS A FINGERPRINT ON A CYCLIC SUBGROUP, NOT A SPINOR WINDING")
print(RULE)
print("  Kept as a negative. Three independent reasons. (a) and (c) are CHECKED against")
print("  data and can fail. (b) is a RESTATEMENT of how the two reps are defined (both")
print("  are 1x1 by construction), so it is reported as a definition, not sold as a test.\n")

# (a) no path: Z_4 is finite.
grp = {complex(np.round(rho_cplx(n), 12)) for n in range(64)}
finite_ok = (len(grp) == 4)
print(f"  (a) NO PATH. The parameter is n = number of integrations, an INTEGER.")
print(f"      {{rho_complex(g^n) : n = 0..63}} has exactly {len(grp)} distinct elements -> Z_4 is FINITE.")
print(f"      A winding number counts turns of a CONTINUOUS path. Finite group, no path,")
print(f"      nothing to wind.                                    finite? {finite_ok}")

# (b) no rep index: both reps are 1x1.
dim_r = np.atleast_2d(rho_real(1)).shape
dim_c = np.atleast_2d(rho_cplx(1)).shape
scalar_ok = (dim_r == (1, 1)) and (dim_c == (1, 1))
print(f"\n  (b) NO REP INDEX. rho_real acts on a {dim_r[0]}x{dim_r[1]} space, rho_complex on {dim_c[0]}x{dim_c[1]}.")
print(f"      The channels are SCALAR amplitudes. Nothing rotates them.  scalar? {scalar_ok}")
print(f"      Honest label: this line CANNOT fail -- the reps are 1x1 by construction.")
print(f"      It restates the setup. The force of the negative is in (c), which can.")

# (c) the hard obstruction: SU(2) has no nontrivial 1-dim rep.
A = 1j*SX
B = 1j*SY
comm = A @ B @ np.linalg.inv(A) @ np.linalg.inv(B)
comm_is_mI = np.allclose(comm, -I2, atol=TOL)
# any scalar rep chi has chi([A,B]) = chi(A)chi(B)chi(A)^-1 chi(B)^-1 = 1 (scalars commute)
chi_forced = 1.0
extends = np.isclose(rho_cplx(2), chi_forced, atol=TOL)
obstruction_ok = comm_is_mI and (not extends)
print(f"\n  (c) THE HARD OBSTRUCTION. SU(2) has NO nontrivial 1-dimensional rep.")
print(f"      With A = i sigma_x, B = i sigma_y:  A B A^-1 B^-1 = -I ?  {bool(comm_is_mI)}")
print(f"      So -I is a COMMUTATOR. Any scalar rep chi obeys")
print(f"        chi(-I) = chi(A) chi(B) chi(A)^-1 chi(B)^-1 = 1   (scalars commute), forced.")
print(f"      But rho_complex(g^2) = {np.round(rho_cplx(2), 6)} != 1.")
print(f"      Therefore rho_complex does NOT extend to a rep of SU(2).  no extension? {bool(not extends)}")
print(f"      A scalar channel CANNOT be a spinor. Section 3's subgroup inclusion is real;")
print(f"      the extension to the whole group does not exist.")

print(f"\n  {'what you have':<36}{'what a spinor winding needs':>40}")
print(f"  {'-'*36}{'-'*40:>40}")
rows = [("n integrations, n in Z", "theta in [0, 4pi), continuous"),
        ("Z_4, finite cyclic", "SU(2), a Lie group"),
        ("two reps, orders 4 and 2", "spin-1/2 and spin-1 reps"),
        ("g acts by a scalar", "g acts on a 2-dim / 3-dim space"),
        ("no rotation anywhere", "an actual SO(3) rotation acting"),
        ("channels are scalars", "channels carry the rep index"),
        ("no continuous parameter", "I^alpha, alpha real, back in play")]
for a, b in rows:
    print(f"  {a:<36}{b:>40}")
print("\n  >> The channels are SCALAR amplitudes with no rep index and nothing rotates")
print("     them. n is an INTEGER so Z_4 is finite and there is no path, hence no winding")
print("     number. A winding needs a continuous parameter, which needs the fractional")
print("     integral back. What you have is the IMAGE of the double cover under a Z_4")
print("     subgroup -- the shadow it casts on one cyclic subgroup. That shadow is exact")
print("     and worth having. It is NOT the cover itself, and it is NOT a winding.")
ok4 = finite_ok and scalar_ok and obstruction_ok
print(f"\n  SECTION 4: {verdict('S4', ok4)}   (the NEGATIVE reproduces: no path, no index, no extension)")

# ============================================================================
print("\n" + RULE)
print("5. THE RATE READING -- REAL, RIGHT DIRECTION, AND NOT INDEPENDENT EVIDENCE")
print(RULE)
step_c = np.angle(-1j)          # -pi/2, principal value
step_r = np.angle(-1.0 + 0j)    # +pi, principal value (branch choice)
print(f"  phase advanced per integration (principal value of arg):")
print(f"    complex channel: arg(-i) = {step_c:+.6f} rad = {step_c/np.pi:+.3f} pi")
print(f"    real channel:    arg(-1) = {step_r:+.6f} rad = {step_r/np.pi:+.3f} pi")
ratio = abs(step_r) / abs(step_c)
half_rate = np.isclose(ratio, 2.0, atol=TOL)
print(f"\n  |real| / |complex| = {ratio:.6f}  -> complex is at HALF rate? {bool(half_rate)}")
print("  The complex channel advances pi/2 per integration, the real channel pi.")

# The consistency triple: faithful <-> order 4 <-> slow. Same as spin-1/2.
slow_is_faithful = (ord_cplx == 4) and (abs(step_c) < abs(step_r)) and (ker_c == [0])
print(f"\n  Consistency triple (the spinor-vs-vector direction):")
print(f"    complex: faithful={ker_c == [0]}, order={ord_cplx}, |step|={abs(step_c):.6f}  <- the SLOW one")
print(f"    real:    faithful={ker_r == [0]}, order={ord_real}, |step|={abs(step_r):.6f}  <- the FAST one")
print(f"    the faithful/order-4 rep is the SLOW one, exactly as spin-1/2 is? {bool(slow_is_faithful)}")

# SU(2) half-angle, for comparison. theta in [0, pi] keeps arccos on one branch.
print(f"\n  The SU(2) half-angle it is being compared to (theta in [0, pi], no branch issue):")
print(f"  {'theta/pi':>10}{'spinor eigenphase':>20}{'SO(3) angle':>14}{'ratio':>10}")
su2_ok = True
for t in (0.25, 0.5, 0.75, 1.0):
    th = t*np.pi
    Ut = np.cos(th/2)*I2 - 1j*np.sin(th/2)*SX
    eig = np.angle(np.linalg.eigvals(Ut))
    spin_phase = np.max(np.abs(eig))
    Rt = so3_image(Ut)
    ang = np.arccos(np.clip((np.trace(Rt) - 1.0)/2.0, -1.0, 1.0))
    r = ang/spin_phase if spin_phase > 1e-12 else np.nan
    su2_ok = su2_ok and np.isclose(ang, th, atol=1e-9) and np.isclose(r, 2.0, atol=1e-9)
    print(f"  {t:>10.2f}{spin_phase:>20.6f}{ang:>14.6f}{r:>10.6f}")
print(f"    the spinor turns at half the vector's rate at every theta? {bool(su2_ok)}")

# The caveat, and it is arithmetic, not opinion.
prod_c = ord_cplx*abs(step_c)
prod_r = ord_real*abs(step_r)
same_fact = np.isclose(prod_c, 2*np.pi, atol=TOL) and np.isclose(prod_r, 2*np.pi, atol=TOL)
print(f"\n  BUT -- the rate is the ORDER restated, not a second fact:")
print(f"    order * |step|, complex channel = {ord_cplx} * {abs(step_c):.6f} = {prod_c:.6f}  (2pi = {2*np.pi:.6f})")
print(f"    order * |step|, real channel    = {ord_real} * {abs(step_r):.6f} = {prod_r:.6f}  (2pi = {2*np.pi:.6f})")
print(f"    order * |step| = 2pi for BOTH channels? {bool(same_fact)}")
# Scope it: this is a property of THESE two scalars, not a theorem about roots of unity.
z_ce = np.exp(2j*np.pi*3/8)
ord_ce = order_of(lambda n: z_ce**n, 1.0, 1e-9)
prod_ce = ord_ce*abs(np.angle(z_ce))
general = bool(np.isclose(prod_ce, 2*np.pi, atol=1e-9))
print(f"\n    SCOPE -- is order * |arg| = 2pi a GENERAL identity? NO, and here is a")
print(f"    counterexample rather than a claim: z = exp(2 pi i * 3/8) has order {ord_ce},")
print(f"    |arg| = {abs(np.angle(z_ce)):.6f}, product = {prod_ce:.6f} = {prod_ce/np.pi:.0f} pi != 2pi.  general? {general}")
print(f"    It holds for -1 and -i because each happens to have |arg| = 2pi/order. So the")
print(f"    line below is scoped to these two scalars; it is not leaning on a theorem.")
print(f"\n  So 'half rate' and 'order 4 vs order 2' are the SAME fact divided by 2pi. The")
print(f"  rate ratio adds nothing beyond section 1, and a rate ratio is NOT a rep index.")
print(f"  Note also: the principal values disagree in SIGN (real {step_r/np.pi:+.1f} pi, complex")
print(f"  {step_c/np.pi:+.1f} pi). arg(-1) sits on the branch cut, so the DIRECTION of the real")
print(f"  channel's turn is a branch choice, not data. Only the magnitude ratio 2 is")
print(f"  branch-independent, and fixing a continuous lift needs I^alpha -- the very thing")
print(f"  section 4 says is missing. The rate reading is downstream of the same gap.")
ok5 = half_rate and slow_is_faithful and su2_ok and same_fact and (not general)
print(f"\n  SECTION 5: {verdict('S5', ok5)}   (half rate confirmed; and shown to be Z1 restated)")

# ============================================================================
print("\n" + RULE)
print("C. CROSS-CHECK VS THE REAL cal.biquaternion (torch.complex64)")
print(RULE)
print(f"  Tolerance rtol = {TOL_C64:.0e}, and it is FLOAT32-LIMITED: cal is torch.complex64")
print(f"  (eps ~ 1e-7). No 1e-15 identity is asserted through complex64 anywhere here.")
print(f"  This experiment has no cancelling differences (all values are 0 and +-1, exactly")
print(f"  representable), so complex64 is not stressed -- the honest tolerance is stated")
print(f"  anyway rather than tightened after the fact.\n")


def t_bq(a):
    return torch.tensor(np.asarray(a, dtype=np.complex64), dtype=torch.complex64)


def close(t, a):
    return bool(np.allclose(t.detach().numpy(), np.asarray(a, dtype=np.complex64),
                            rtol=TOL_C64, atol=TOL_C64))


tI, tJ, tK, tONE = t_bq(QI), t_bq(QJ), t_bq(QK), t_bq(ONE)

# C1: handedness in cal itself.
c1a = close(quat_mul(tI, tJ), QK)
c1b = close(quat_mul(quat_mul(tI, tJ), tK), -ONE)
print(f"  C1 handedness in cal:  quat_mul(i, j) == k          -> {c1a}")
print(f"                         quat_mul(quat_mul(i,j),k) == -1 -> {c1b}")

# C2: the Z_4 cycle of the biquaternion i, cal vs the complex128 formula.
print(f"\n  C2 the Z_4 cycle of q = i under cal.quat_mul vs the complex128 rep formula:")
print(f"  {'n':>4}{'cal quat_mul i^n':>28}{'numpy c128 i^n':>28}{'agree':>8}")
c2 = True
tpow = t_bq(ONE)
for n in range(1, 5):
    tpow = quat_mul(tpow, tI)
    npow = quat_pow_np(QI, n)
    agree = close(tpow, npow)
    c2 = c2 and agree
    tv = np.real_if_close(tpow.detach().numpy(), tol=1e3)
    print(f"  {n:>4}{str(np.round(tv, 4)):>28}{str(np.round(np.real_if_close(npow), 4)):>28}{str(agree):>8}")
cyc_ok = c2 and close(quat_mul(quat_mul(tI, tI), quat_mul(tI, tI)), ONE)
print(f"     i^4 = 1 in cal -> the cycle closes at 4? {cyc_ok}")

# C3: the matrix embedding is a homomorphism and carries the same orders.
Mi_t = biquat_to_matrix(tI)
Mi = Mi_t.detach().numpy().astype(np.complex128)
Mi_np = biquat_to_matrix_np(QI)
c3a = bool(np.allclose(Mi, Mi_np, rtol=TOL_C64, atol=TOL_C64))
c3b = True
for n in range(1, 5):
    lhs = np.linalg.matrix_power(Mi, n)
    rhs = biquat_to_matrix_np(quat_pow_np(QI, n))
    c3b = c3b and bool(np.allclose(lhs, rhs, rtol=TOL_C64, atol=TOL_C64))
print(f"\n  C3 cal's biquat_to_matrix(i) =\n{block(Mi, indent='      ')}")
print(f"     matches the inline complex128 formula? {c3a}")
print(f"     M(i)^n == M(i^n) for n = 1..4 (homomorphism)? {c3b}")

# C4: <i> sits inside the UNITARY part -- the dagger is the inverse there.
c4a = close(quat_mul(tI, hermitian_conj(tI)), ONE)
c4b = close(hermitian_conj(tI), quat_pow_np(QI, 3))
nrm = quat_norm_sq(tI).detach().numpy()
c4c = bool(np.allclose(nrm, 1.0, rtol=TOL_C64, atol=TOL_C64))
print(f"\n  C4 <i> lies in the unit (SU(2)) part of cal's algebra:")
print(f"     quat_mul(i, hermitian_conj(i)) == 1 ?  {c4a}")
print(f"     hermitian_conj(i) == i^3 == i^-1 ?     {c4b}")
print(f"     quat_norm_sq(i) = {float(np.real(nrm)):.6f} == 1 ?           {c4c}")

# C5: THE FINDING ITSELF, reproduced through cal's own embedding.
det_Mi = np.linalg.det(Mi)
c5a = bool(np.allclose(det_Mi, 1.0, rtol=TOL_C64, atol=TOL_C64))
c5b = bool(np.allclose(Mi @ Mi.conj().T, I2, rtol=TOL_C64, atol=TOL_C64))
ord_Mi = order_of(lambda n: np.linalg.matrix_power(Mi, n), I2, TOL_C64)
ord_RMi = order_of(lambda n: so3_image(np.linalg.matrix_power(Mi, n)), I3, TOL_C64)
Mi2_is_mI = bool(np.allclose(Mi @ Mi, -I2, rtol=TOL_C64, atol=TOL_C64))
print(f"\n  C5 THE FINDING, through cal's own embedding rather than a hand-built U:")
print(f"     det M(i) = {np.round(det_Mi, 6)}, unitary? {c5b}  -> M(i) is in SU(2)? {bool(c5a and c5b)}")
print(f"     order of M(i) in SU(2)        = {ord_Mi}")
print(f"     order of its SO(3) image      = {ord_RMi}")
print(f"     M(i)^2 = -I (the invisible kernel element)? {Mi2_is_mI}")
print(f"     -> 4 upstairs, 2 downstairs, kernel -1: cal's i reproduces sections 1-3.")
c5 = (ord_Mi == 4) and (ord_RMi == 2) and Mi2_is_mI and c5a and c5b

okC = (c1a and c1b and c2 and cyc_ok and c3a and c3b and c4a and c4b and c4c and c5)
print(f"\n  SECTION C: {verdict('SC', okC)}   (cal agrees at rtol {TOL_C64:.0e}, float32-limited)")

# ============================================================================
print("\n" + RULE)
print("V. VERDICT")
print(RULE)
rows = [("two reps of one Z_4, orders 2 and 4",
         ok1, "sec 1, both homomorphisms"),
        ("real rep factors through Z_4/{+-1}",
         (ker_r == [0, 2]) and (ker_c == [0]), "sec 2, ker = {1, g^2}"),
        ("rho_complex(g^2) = -1, rho_real(g^2) = +1",
         bool(sees and blind), "sec 2, the kernel element"),
        ("a genuine 2:1 cover of finite groups",
         ok1 and ok2, "index 2, verified"),
        ("it IS a Z_4 < SU(2) over its SO(3) image",
         ok3, "sec 3, orders 4 vs 2"),
        ("<U> and cal's <M(i)> are the SAME subgroup",
         bool(same_sub), "NO: conjugate, meet in +-I"),
        ("the covering map is squaring",
         bool(i_sq_is_m1 and U2_is_mI), "i^2 = -1, U^2 = -I"),
        ("cal's own i reproduces all of it",
         c5, "sec C5, via biquat_to_matrix"),
        ("the channels are spinor / tensor",
         not ok4, "NO: scalars, no rep index"),
        ("rho_complex extends to SU(2)",
         bool(extends), "NO: -I is a commutator"),
        ("there is a winding number",
         not finite_ok, "NO: n integer, finite group"),
        ("the rate ratio is extra evidence",
         not same_fact, "NO: order * |step| = 2pi"),
        ("4pi vs 2pi return",
         False, "NOT YET: alpha continuous")]
print(f"  {'claim':<42}{'holds':>6}{'basis':>29}")
print(f"  {'-'*42}{'-'*6:>6}{'-'*29:>29}")
for claim, held, basis in rows:
    print(f"  {claim:<42}{('YES' if held else 'NO'):>6}{basis:>29}")

print("\n  >> You found the double cover's fingerprint on a cyclic subgroup, and the")
print("     fingerprint is i^2 = -1: the complex channel needs four steps to come home")
print("     because the real one needs two, and squaring is the map between them. That")
print("     part is exact and it survives every check here, including through cal's own")
print("     embedding. Calling the channels spinor and tensor claims MORE than the")
print("     algebra shows: a scalar rep of SU(2) is trivial, so a scalar channel cannot")
print("     be a spinor, and with n an integer there is no path to wind. The rate reading")
print("     points the right way and is worth one sentence, but it is the order fact")
print("     restated. The negative stands.")

print("\n" + RULE)
print("SECTION SUMMARY")
print(RULE)
for k in ("S0", "S1", "S2", "S3", "S4", "S5", "SC"):
    print(f"  {k:<6}{'PASS' if RESULTS[k] else 'FAIL'}")
all_ok = all(RESULTS.values())
print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}  ({sum(RESULTS.values())}/{len(RESULTS)} sections)")
print("  Note: sections 4 and 5 PASS by reproducing NEGATIVES. A PASS there means the")
print("  limitation is confirmed, not that the spinor/winding reading is supported.")
print(RULE)

sys.exit(0 if all_ok else 1)
