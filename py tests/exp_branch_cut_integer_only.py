"""
Fractional powers need a branch of log; integer powers do not. What that costs.

"Also integrate, not differentiate. I use exp maps rather than logs for a reason."

A power is defined through a logarithm:  z^a = exp(a log z),  and log is MULTIVALUED,
    log z = ln|z| + i(arg z + 2 pi k),   k in Z,
so z^a carries a branch factor e^{2 pi i a k}. That factor equals 1 for EVERY k if and
only if a is an INTEGER. Integer powers are n multiplications: single-valued, no log, no
choice. Fractional powers are a choice, and the choice has to be declared.

Every fractional-integral symbol of the form (i w)^{-alpha} or (-u)^{-alpha} silently
picks a branch of log. This script prices that silence.

CLAIMS UNDER TEST (all six reproduce; none is refuted):
  B1: the branch factor e^{2 pi i a k} has modulus 1 for every a and every k, so a table
      of MODULI looks uniformly fine and hides the whole problem. The disagreement lives
      in the PHASE. Tabulated as arg at k=0 versus arg at k=1: identical at integer a,
      different at fractional a, including at alpha = 1.9965071.
  B2: the geometric channel's rate lambda = -u (u > 0) sits ON the principal branch cut,
      the negative real axis. Not near it: on it. arg(-u + i eps) -> +pi and
      arg(-u - i eps) -> -pi, and the gap stays 2 pi as eps -> 0. There is no limit to
      take; arg(-u) is +pi or -pi and the algebra does not say which. Controlled against
      the identical procedure at +u, OFF the cut, where the gap vanishes -- so the 2 pi
      is a fact about where lambda sits, not an artifact of the eps -> 0 method.
  B3: therefore (-u)^{-alpha} is two different numbers. The arg = +pi and arg = -pi
      branches AGREE exactly at integer alpha and DISAGREE at fractional alpha, at
      alpha = 1.9965071 included (a real 2.2% gap, not a rounding artifact -- the gap is
      printed, and swept over random u to show it is not special to u = 2). The
      fractional integral of the COST channel is a branch choice, not a number.
  B4: 1/(-u)^n by repeated multiplication equals exp(-n log(-u)) with arg = +pi, for
      every integer n. At integer order the log route happens to agree, because
      e^{2 pi i n k} = 1: the log is REMOVABLE. At fractional order it is not removable,
      and there is no repeated-multiplication route to fall back on. That asymmetry is
      the reason to stay integer.
  B5: 1/i^n agrees across EVERY branch k for integer n, so the Z_4 period is branch-free
      and is a THEOREM, assuming nothing. At fractional alpha the branches disagree and
      alpha = 1.9965071 is not well-defined until k is declared.
  B6: integrate versus differentiate. d/ds e^{Ls} MULTIPLIES by L: gain w, unbounded as
      w -> inf. INT e^{Ls} ds MULTIPLIES by 1/L: gain 1/w -> 0, bounded. The first-order
      gains are MEASURED (central difference applied to the mode; the antiderivative
      verified against quadrature before its amplitude is used), not asserted by writing
      gain = w -- an asserted table restates the claim and cannot fail. Measured at
      w = 0.1, 1, 10, 100. Both routes carry the same -1 at
      order 2 (i^2 = -1 does not care which way you go); only integration is bounded.
      An accumulation law built on differentiation would amplify every fine detail of
      the past. Built on integration it forgets them. Only one of those is a memory.

PRECISION. The findings are computed in numpy complex128 (eps ~ 2.2e-16) using cal's
own representation formulas, written out. cal itself is torch complex64 (eps ~ 1e-7), so
the CROSS-CHECK section (section 7) calls the real cal.biquaternion functions on the same
inputs and confirms agreement at rtol = 1e-5, a tolerance that is honest for float32.
No 1e-15 identity is ever asserted through complex64.

Seeded and deterministic: np.random.default_rng(20260715). ASCII only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from cal.biquaternion import (
    CDTYPE,
    biquat_to_matrix,
    hermitian_conj,
    quat_mul,
    quat_norm_sq,
)

RNG = np.random.default_rng(20260715)

RULE = "=" * 78
U = 2.0                       # the geometric channel's rate is lambda = -U
ALPHA = 1.9965071             # the disputed fractional order
INT_TOL = 1e-12               # complex128 tolerance for exact-branch agreement
C64_RTOL = 1e-5               # float32-honest tolerance for the cal cross-check

RESULTS = []


def record(name, ok):
    RESULTS.append((name, ok))
    print(f"\n  [{'PASS' if ok else 'FAIL'}] {name}")


# ---- cal's representation formulas, written out in numpy complex128 --------
# quat_mul, cal/biquaternion.py lines 23-42, verbatim structure.

def quat_mul_np(P, Q):
    """Biquaternion product PQ in complex128. Mirrors cal.biquaternion.quat_mul."""
    p0, p1, p2, p3 = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    q0, q1, q2, q3 = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]
    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0
    return np.stack([r0, r1, r2, r3], axis=-1)


def biquat_to_matrix_np(q):
    """M(q) = [[q0 + h q3, q1 + h q2], [-q1 + h q2, q0 - h q3]], h = 1j. complex128."""
    h = 1j
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.array([[q0 + h*q3, q1 + h*q2],
                     [-q1 + h*q2, q0 - h*q3]], dtype=np.complex128)


def hermitian_conj_np(q):
    """q_dagger = (conj(q0), -conj(q1), -conj(q2), -conj(q3)). complex128."""
    qb = np.conj(q)
    return np.stack([qb[..., 0], -qb[..., 1], -qb[..., 2], -qb[..., 3]], axis=-1)


def bq(q0=0.0, q1=0.0, q2=0.0, q3=0.0):
    return np.array([q0, q1, q2, q3], dtype=np.complex128)


ONE = bq(1.0)
EI = bq(0.0, 1.0)     # the i basis element
EJ = bq(0.0, 0.0, 1.0)
EK = bq(0.0, 0.0, 0.0, 1.0)


# ============================================================================
print(RULE)
print("0. HANDEDNESS GUARD. The basis must be right-handed before anything else.")
print(RULE)
print("  The whole Z_4 argument is a statement about powers of i, so the sign")
print("  convention of the basis is load-bearing. Assert it, do not assume it.\n")

ij = quat_mul_np(EI, EJ)
ijk = quat_mul_np(quat_mul_np(EI, EJ), EK)
ii = quat_mul_np(EI, EI)

print(f"  {'product':>14}{'result (q0, q1, q2, q3)':>34}{'expected':>16}")
for label, got, want, wname in (
    ("i * j", ij, EK, "k"),
    ("i * j * k", ijk, -ONE, "-1"),
    ("i * i", ii, -ONE, "-1"),
):
    got_s = "(" + ", ".join(f"{v.real:+.0f}" for v in got) + ")"
    print(f"  {label:>14}{got_s:>34}{wname:>16}")

guard_np = (np.allclose(ij, EK, atol=INT_TOL)
            and np.allclose(ijk, -ONE, atol=INT_TOL)
            and np.allclose(ii, -ONE, atol=INT_TOL))

t_i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=CDTYPE)
t_j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=CDTYPE)
t_k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=CDTYPE)
t_one = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=CDTYPE)

cal_ij = quat_mul(t_i, t_j)
cal_ijk = quat_mul(quat_mul(t_i, t_j), t_k)
guard_cal = (torch.allclose(cal_ij, t_k, rtol=C64_RTOL, atol=1e-6)
             and torch.allclose(cal_ijk, -t_one, rtol=C64_RTOL, atol=1e-6))

print("\n  Same guard through the real cal.biquaternion.quat_mul (torch complex64):")
print(f"    i * j     -> {[f'{v.real:+.0f}' for v in cal_ij.tolist()]}   (want k)")
print(f"    i * j * k -> {[f'{v.real:+.0f}' for v in cal_ijk.tolist()]}   (want -1)")
print("\n  >> RIGHT-HANDED: i*j = k and ijk = -1, in numpy and in cal alike.")
record("B0 handedness guard: i*j = k, ijk = -1, right-handed in both reps",
       guard_np and guard_cal)


# ============================================================================
print("\n" + RULE)
print("1. log IS MULTIVALUED, AND THE MODULUS TABLE HIDES IT.")
print(RULE)
print("  log z = ln|z| + i(arg z + 2 pi k), k in Z.  So")
print("    z^a = exp(a log z) = |z|^a e^{i a arg z} * e^{2 pi i a k}")
print("                                              ^^^^^^^^^^^^^^")
print("                                              the branch factor")
print("  e^{2 pi i a k} = 1 for EVERY k  <=>  a is an INTEGER.\n")
print("  First the trap. The MODULUS of the branch factor is 1 always, for every a:\n")
print(f"  {'alpha':>11}{'|k=0|':>9}{'|k=1|':>9}{'|k=2|':>9}{'|k=-1|':>9}"
      f"{'moduli agree?':>16}")
mod_all_one = True
for a in (1, 2, 3, 4, 0.5, 1.5, ALPHA):
    vals = [np.exp(2j*np.pi*a*k) for k in (0, 1, 2, -1)]
    mods = [abs(v) for v in vals]
    same_mod = all(abs(m - 1.0) < INT_TOL for m in mods)
    mod_all_one = mod_all_one and same_mod
    print(f"  {a:>11}{mods[0]:>9.3f}{mods[1]:>9.3f}{mods[2]:>9.3f}{mods[3]:>9.3f}"
          f"{str(same_mod):>16}")
print("\n  >> Every modulus is 1, at every alpha, on every branch. A modulus table says")
print("     'no problem here' and it is lying by omission. The branch factor is a PHASE.")
print("\n  Now the phase, k = 0 against k = 1:\n")
print(f"  {'alpha':>11}{'arg at k=0':>14}{'arg at k=1':>14}{'|phase gap|':>14}"
      f"{'integer?':>10}{'same value?':>13}")
b1_ok = True
for a in (1, 2, 3, 4, 0.5, 1.5, ALPHA):
    v0, v1 = np.exp(2j*np.pi*a*0), np.exp(2j*np.pi*a*1)
    gap = abs(np.angle(v1) - np.angle(v0))
    is_int = float(a).is_integer()
    same = abs(v1 - v0) < INT_TOL
    # the claim: single-valued iff integer
    b1_ok = b1_ok and (same == is_int)
    print(f"  {a:>11}{np.angle(v0):>14.6f}{np.angle(v1):>14.6f}{gap:>14.6f}"
          f"{str(is_int):>10}{str(same):>13}")
print("\n  >> INTEGER alpha: every branch agrees. z^n is one number.")
print("     FRACTIONAL alpha: the branches disagree. z^alpha is a SET, and something")
print("     outside the algebra has to choose. An integer-order law never has to.")
record("B1 branch factor is unimodular for all alpha; single-valued IFF alpha integer",
       mod_all_one and b1_ok)


# ============================================================================
print("\n" + RULE)
print("2. THE GEOMETRIC CHANNEL SITS EXACTLY ON THE PRINCIPAL BRANCH CUT.")
print(RULE)
print("  The standard cut of log is the NEGATIVE REAL AXIS. The geometric channel's")
print(f"  rate is lambda = -u, u > 0. That is ON the cut, not near it.\n")
print(f"  lambda = -u = {-U}. Approach the cut from above and from below:\n")
print(f"  {'eps':>10}{'arg(-u + i eps)':>19}{'arg(-u - i eps)':>19}{'gap':>12}"
      f"{'gap - 2pi':>14}")
b2_ok = True
gaps = []
for eps in (1e-1, 1e-3, 1e-6, 1e-12):
    up = np.angle(-U + 1j*eps)
    dn = np.angle(-U - 1j*eps)
    gap = abs(up - dn)
    gaps.append(gap)
    print(f"  {eps:>10.0e}{up:>19.9f}{dn:>19.9f}{gap:>12.6f}{gap - 2*np.pi:>14.3e}")

# CONTROL. Run the IDENTICAL procedure at +u, which is OFF the cut. If the 2 pi
# gap were an artifact of straddling the real axis with eps -> 0, the control
# would show it too. It must not, or section 2 is measuring its own method.
print("\n  CONTROL, same procedure at lambda = +u (OFF the cut). If the 2 pi gap were")
print("  an artifact of straddling the real axis, this column would show it too:\n")
print(f"  {'eps':>10}{'arg(+u + i eps)':>19}{'arg(+u - i eps)':>19}{'gap':>12}"
      f"{'gap - 2pi':>14}")
ctrl_gaps = []
for eps in (1e-1, 1e-3, 1e-6, 1e-12):
    up_c = np.angle(+U + 1j*eps)
    dn_c = np.angle(+U - 1j*eps)
    gap_c = abs(up_c - dn_c)
    ctrl_gaps.append(gap_c)
    print(f"  {eps:>10.0e}{up_c:>19.9f}{dn_c:>19.9f}{gap_c:>12.6f}"
          f"{gap_c - 2*np.pi:>14.3e}")

# the claim: ON the cut the gap converges to 2 pi and does not shrink;
# OFF the cut the same procedure gives a gap that vanishes. Both are required.
b2_ok = (abs(gaps[-1] - 2*np.pi) < 1e-9)          # on the cut: gap -> 2 pi
b2_ok = b2_ok and (gaps[-1] > gaps[0] - 1e-9)     # and it grows, does not shrink
b2_ok = b2_ok and (ctrl_gaps[-1] < 1e-9)          # off the cut: gap -> 0
b2_ok = b2_ok and (ctrl_gaps[0] < gaps[0])        # control is nowhere near it
print(f"\n  on  the cut, gap at eps = 1e-1 : {gaps[0]:.9f}")
print(f"  on  the cut, gap at eps = 1e-12: {gaps[-1]:.9f}   (2 pi = {2*np.pi:.9f})")
print(f"  OFF the cut, gap at eps = 1e-12: {ctrl_gaps[-1]:.9f}   (vanishes)")
print("\n  >> The control VANISHES and the cut does not. So the 2 pi is a fact about")
print("     where lambda = -u sits, not about the eps -> 0 procedure. The test")
print("     discriminates; it is not measuring its own method.")
print("\n  >> The argument JUMPS by 2 pi across the cut and the jump does NOT shrink;")
print("     it GROWS to exactly 2 pi. There is no limit. arg(-u) is +pi or -pi and the")
print("     algebra does not say which. This is not a removable singularity of the")
print("     bookkeeping. It is the cut, and the cost channel is standing on it.")
record("B2 rate lambda = -u lies on the cut; the 2 pi gap does not vanish as eps -> 0",
       b2_ok)


# ============================================================================
print("\n" + RULE)
print("3. SO (-u)^{-alpha} IS TWO NUMBERS, NOT ONE.")
print(RULE)
print("  Take the two branches the previous section left undecided and evaluate the")
print("  geometric channel's fractional symbol on each:\n")
print(f"  {'alpha':>11}{'(-u)^-a via arg=+pi':>26}{'via arg=-pi':>26}"
      f"{'|gap|':>11}{'agree?':>8}")
b3_ok = True
for a in (1, 2, 3, 0.5, 1.5, ALPHA):
    vp = U**(-a)*np.exp(-1j*a*np.pi)
    vm = U**(-a)*np.exp(+1j*a*np.pi)
    gap = abs(vp - vm)
    agree = gap < INT_TOL
    is_int = float(a).is_integer()
    b3_ok = b3_ok and (agree == is_int)
    print(f"  {a:>11}{str(np.round(vp, 6)):>26}{str(np.round(vm, 6)):>26}"
          f"{gap:>11.2e}{str(agree):>8}")

vp_a = U**(-ALPHA)*np.exp(-1j*ALPHA*np.pi)
vm_a = U**(-ALPHA)*np.exp(+1j*ALPHA*np.pi)
rel = abs(vp_a - vm_a) / abs(vp_a)
print(f"\n  At the disputed order alpha = {ALPHA}:")
print(f"    |branch gap|      = {abs(vp_a - vm_a):.8f}")
print(f"    relative gap      = {rel:.6%}  of |(-u)^-alpha| = {abs(vp_a):.8f}")
print(f"    complex128 eps    = {np.finfo(np.complex128).eps:.2e}")
print("  >> The gap is ~2% of the value and ~1e14 times machine epsilon. This is a real")
print("     disagreement about which number the symbol denotes, not a rounding artifact.")
print("     alpha = 1.9965071 is NEAR 2, so the gap is small; near is not integer.")

print("\n  And it is not special to u = 2. Random rates, seeded (default_rng(20260715)):\n")
print(f"  {'u':>11}{'|gap| at alpha=2 (int)':>25}{'|gap| at alpha=1.9965071':>27}"
      f"{'int exact?':>12}")
b3_rand_ok = True
for u in RNG.uniform(0.25, 8.0, size=5):
    gi = abs(u**(-2)*np.exp(-2j*np.pi) - u**(-2)*np.exp(+2j*np.pi))
    gf = abs(u**(-ALPHA)*np.exp(-1j*ALPHA*np.pi) - u**(-ALPHA)*np.exp(+1j*ALPHA*np.pi))
    # RELATIVE test. gi carries the u^-2 prefactor, so an ABSOLUTE tolerance on it
    # is a statement about how big u is, not about the branch. Divide the scale out.
    int_exact = (gi / u**(-2)) < INT_TOL
    ok = int_exact and (gf > 1e-6)
    b3_rand_ok = b3_rand_ok and ok
    print(f"  {u:>11.6f}{gi:>25.2e}{gf:>27.8f}{str(int_exact):>12}")
print("\n  >> At INTEGER alpha the two branches agree exactly, at every rate. At")
print("     fractional alpha they never do. The fractional integral of the COST channel")
print("     is not a number -- it is a branch choice. Any claim that 'geo rotates by")
print("     -alpha*pi' has silently picked arg(-u) = +pi and not said so.")
record("B3 (-u)^-alpha: branches agree IFF alpha integer; disagree at alpha = 1.9965071",
       b3_ok and b3_rand_ok)


# ============================================================================
print("\n" + RULE)
print("4. INTEGRATION NEEDS NO LOG. THAT IS THE POINT.")
print(RULE)
print("  INT e^{L s} ds = e^{L s} / L.  Integrate n times: divide by L^n.")
print("  L^n for integer n is n MULTIPLICATIONS. No log, no branch, no choice.\n")
print(f"  {'n':>4}{'1/(-u)^n by repeated mult':>28}{'via exp(-n log(-u)), arg=+pi':>32}"
      f"{'|gap|':>11}{'agree?':>8}")
b4_ok = True
for n in range(5):
    direct = 1.0 + 0.0j
    for _ in range(n):
        direct = direct / (-U)
    vialog = np.exp(-n*(np.log(U) + 1j*np.pi))
    gap = abs(direct - vialog)
    agree = gap < 1e-14
    b4_ok = b4_ok and agree
    print(f"  {n:>4}{str(np.round(direct, 8)):>28}{str(np.round(vialog, 8)):>32}"
          f"{gap:>11.2e}{str(agree):>8}")
print("\n  >> Identical. For integer n you may compute it either way and the log route")
print("     happens to agree, because e^{2 pi i n k} = 1. The log is REMOVABLE at")
print("     integer order: it appears in the derivation and cancels out of the answer.")
print("     For fractional alpha it is NOT removable, and there is no repeated-")
print("     multiplication route to fall back on -- 'multiply 1.9965071 times' is not")
print("     an operation. That asymmetry is the whole reason to stay integer.")
record("B4 1/(-u)^n: repeated multiplication == exp(-n log(-u)); the log is removable",
       b4_ok)


# ============================================================================
print("\n" + RULE)
print("5. HENCE Z_4 IS BRANCH-FREE AND IS A THEOREM.")
print(RULE)
print("  The claim: integrate 4 times, 1/i^4 = 1, back to the start.")
print("  Compute it every way and see which route needs a convention:\n")
print(f"  {'n':>4}{'1/i^n, repeated mult':>24}{'exp(-n log i), k=0':>22}"
      f"{'k=1':>22}{'k=-1':>22}{'all agree?':>12}")
b5_int_ok = True
for n in range(5):
    direct = 1.0 + 0.0j
    for _ in range(n):
        direct = direct / 1j
    k0 = np.exp(-n*(1j*np.pi/2))
    k1 = np.exp(-n*(1j*np.pi/2 + 2j*np.pi))
    km = np.exp(-n*(1j*np.pi/2 - 2j*np.pi))
    ok = (abs(direct - k0) < 1e-12 and abs(direct - k1) < 1e-12
          and abs(direct - km) < 1e-12)
    b5_int_ok = b5_int_ok and ok
    print(f"  {n:>4}{str(np.round(direct, 6)):>24}{str(np.round(k0, 6)):>22}"
          f"{str(np.round(k1, 6)):>22}{str(np.round(km, 6)):>22}{str(ok):>12}")
print("\n  now the fractional version, exactly the same test:\n")
print(f"  {'alpha':>11}{'exp(-a log i), k=0':>24}{'k=1':>24}{'|gap|':>11}{'agree?':>8}")
b5_frac_ok = True
for a in (0.5, 1.5, 2.0, ALPHA):
    k0 = np.exp(-a*(1j*np.pi/2))
    k1 = np.exp(-a*(1j*np.pi/2 + 2j*np.pi))
    gap = abs(k0 - k1)
    agree = gap < INT_TOL
    is_int = float(a).is_integer()
    b5_frac_ok = b5_frac_ok and (agree == is_int)
    print(f"  {a:>11}{str(np.round(k0, 6)):>24}{str(np.round(k1, 6)):>24}"
          f"{gap:>11.2e}{str(agree):>8}")
print("\n  >> Integer n: every branch agrees, no convention is needed, and the period-4")
print("     structure is a THEOREM -- it survives the integer route with nothing assumed.")
print("     Fractional alpha: the branches disagree, and alpha = 1.9965071 is not even")
print("     WELL-DEFINED until k is declared. Note alpha = 2.0 passes: it is an integer")
print("     written as a float, and the test keys on the value, not the spelling.")
record("B5 1/i^n branch-free for integer n (Z_4 is a theorem); ill-defined at fractional",
       b5_int_ok and b5_frac_ok)


# ============================================================================
print("\n" + RULE)
print("6. INTEGRATE, NOT DIFFERENTIATE. THE OTHER HALF OF THE REASON.")
print(RULE)
print("  d/ds e^{Ls}   = L e^{Ls}     MULTIPLY by L    -> amplifies")
print("  INT e^{Ls} ds = e^{Ls} / L   MULTIPLY by 1/L  -> attenuates\n")
print("  The first-order gains below are MEASURED, not asserted. d/ds is APPLIED to")
print("  f(s) = e^{i w s} as a central difference (step h = 1e-6), and the candidate")
print("  antiderivative is VERIFIED against trapezoid quadrature before its amplitude")
print("  is read off as the integration gain. A table that merely printed w and 1/w")
print("  back at you would restate the claim, not test it: it could not fail.\n")
print(f"  {'w':>7}{'|d/ds| gain':>13}{'exact w':>10}{'|INT| gain':>12}{'exact 1/w':>11}"
      f"{'w^2':>10}{'1/w^2':>11}{'antideriv?':>12}")
b6_ok = True
prev_d, prev_i = None, None
S0, H, L = 0.3, 1e-6, 1e-4
for w in (0.1, 1.0, 10.0, 100.0):
    f = lambda t, w=w: np.exp(1j*w*t)
    G = lambda t, w=w: np.exp(1j*w*t)/(1j*w)   # candidate antiderivative of f
    # MEASURE the differentiation gain: apply the operator, do not assume it.
    gd = abs((f(S0 + H) - f(S0 - H))/(2*H)) / abs(f(S0))
    # VERIFY G' = f by quadrature BEFORE using |G| as the integration gain.
    ts = np.linspace(S0, S0 + L, 4001)
    anti_res = abs(np.trapezoid(f(ts), ts) - (G(S0 + L) - G(S0)))
    anti_ok = anti_res < 1e-11
    gi = abs(G(S0)) / abs(f(S0))
    # the measured gains must MATCH w and 1/w; this is what can fail.
    meas_ok = (anti_ok and abs(gd - w) <= 1e-6*w and abs(gi - 1.0/w) <= 1e-6/w)
    b6_ok = b6_ok and meas_ok
    if prev_d is not None:
        # monotone: differentiation gain grows with w, integration gain shrinks
        b6_ok = b6_ok and (gd > prev_d) and (gi < prev_i)
    prev_d, prev_i = gd, gi
    print(f"  {w:>7.1f}{gd:>13.6f}{w:>10.2f}{gi:>12.6f}{1.0/w:>11.4f}"
          f"{w**2:>10.2f}{1.0/w**2:>11.6f}{str(anti_ok):>12}")
print("\n  >> The measured gains match w and 1/w to 1e-6 relative, and the antiderivative")
print("     check passes at every w, so the two columns are a measurement agreeing with")
print("     the algebra rather than the algebra printed twice.")
print("     Differentiation gain is UNBOUNDED as w -> inf: high frequencies blow up.")
print("     Integration gain is bounded by 1/w -> 0: high frequencies are suppressed.")
print("     An accumulation law built on differentiation would amplify every fine")
print("     detail of the past. Built on integration it forgets them. Only one of those")
print("     is a memory. THIS is why an accumulation law integrates.")
print("\n  And both routes carry the SAME minus at order 2:\n")
print(f"  {'w':>8}{'d^2 factor (i w)^2':>24}{'INT^2 factor 1/(i w)^2':>26}"
      f"{'both real < 0?':>16}")
b6_sign_ok = True
for w in (0.1, 1.0, 10.0, 100.0):
    d2 = (1j*w)**2
    i2 = 1.0/(1j*w)**2
    ok = (abs(d2.imag) < 1e-12 and abs(i2.imag) < 1e-12
          and d2.real < 0 and i2.real < 0)
    b6_sign_ok = b6_sign_ok and ok
    print(f"  {w:>8.1f}{str(np.round(d2, 8)):>24}{str(np.round(i2, 8)):>26}"
          f"{str(ok):>16}")
print(f"\n    at w = 1:  d^2/ds^2 e^{{i w s}} = (i w)^2 e^{{iws}}   = "
      f"{((1j*1.0)**2).real:+.0f} e^{{iws}}")
print(f"               INT^2     e^{{i w s}} = e^{{iws}}/(i w)^2 = "
      f"{(1/(1j*1.0)**2).real:+.0f} e^{{iws}}")
print("  >> Same -1, opposite magnitude (w^2 against 1/w^2). i^2 = -1 does not care")
print("     which way you go; BOUNDEDNESS does. That is the reason, and it is not")
print("     aesthetic. Note the -1 is exact at w = 1 and the SIGN is what generalises:")
print("     both factors stay real and negative at every w, with magnitudes that")
print("     diverge and vanish respectively.")
record("B6 MEASURED d/ds gain w unbounded, INT gain 1/w -> 0; both carry i^2 = -1 "
       "at order 2",
       b6_ok and b6_sign_ok)


# ============================================================================
print("\n" + RULE)
print("7. CROSS-CHECK AGAINST THE REAL cal.biquaternion (torch complex64).")
print(RULE)
print("  Everything above is numpy complex128 built from cal's representation formulas.")
print("  Here the SAME claims are run through the actual cal functions. cal is")
print(f"  {CDTYPE}, so float32 eps ~ {np.finfo(np.float32).eps:.2e} and the honest")
print(f"  tolerance is rtol = {C64_RTOL:.0e}. Nothing here is asserted to 1e-15.\n")

# 7a. The Z_4 claim in cal. The real span of {1, e1} is a copy of C, so
#     1/i^n in C corresponds to (e1^{-1})^n = (-e1)^n by repeated quat_mul.
print("  7a. Z_4 through cal.quat_mul. The real span of {1, i} inside the")
print("      biquaternions is a copy of C, so 1/i^n corresponds to (-i)^n built by")
print("      repeated quat_mul on the basis element i = (0,1,0,0):\n")
print(f"  {'n':>4}{'cal (-i)^n as (q0, q1)':>28}{'numpy 1/i^n':>22}{'|gap|':>12}"
      f"{'agree (rtol 1e-5)?':>20}")
neg_i_t = torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=CDTYPE)
acc = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=CDTYPE)
x7a = True
for n in range(5):
    # the iso R[i] -> C sends q0 + q1*i (q0, q1 REAL) to q0 + 1j*q1
    q0c, q1c = complex(acc[0].item()), complex(acc[1].item())
    cal_c = q0c.real + 1j*q1c.real
    # the {1, i} plane only: components 2 and 3 must stay zero, and the
    # coefficients must stay real for the iso to apply
    planar = (abs(complex(acc[2].item())) < 1e-6
              and abs(complex(acc[3].item())) < 1e-6
              and abs(q0c.imag) < 1e-6 and abs(q1c.imag) < 1e-6)
    np_c = 1.0 + 0.0j
    for _ in range(n):
        np_c = np_c / 1j
    gap = abs(cal_c - np_c)
    ok = planar and np.isclose(cal_c, np_c, rtol=C64_RTOL, atol=1e-6)
    x7a = x7a and ok
    print(f"  {n:>4}{f'({acc[0].item().real:+.0f}, {acc[1].item().real:+.0f})':>28}"
          f"{str(np.round(np_c, 6)):>22}{gap:>12.2e}{str(ok):>20}")
    acc = quat_mul(acc, neg_i_t)
print("\n  >> cal's own product reproduces the period-4 cycle 1, -i, -1, +i, 1 exactly,")
print("     and it never leaves the {1, i} plane. The Z_4 needs no branch in cal either.")

# 7b. The scalar rate -u through cal.
print("\n  7b. The rate lambda = -u embedded as the scalar biquaternion (-u,0,0,0),")
print("      with 1/(-u)^n built by repeated quat_mul on its inverse (-1/u,0,0,0):\n")
print(f"  {'n':>4}{'cal 1/(-u)^n (q0)':>22}{'numpy 1/(-u)^n':>20}{'|gap|':>12}"
      f"{'agree (rtol 1e-5)?':>20}")
inv_u_t = torch.tensor([-1.0/U, 0.0, 0.0, 0.0], dtype=CDTYPE)
acc = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=CDTYPE)
x7b = True
for n in range(5):
    cal_v = complex(acc[0].item())
    np_v = 1.0 + 0.0j
    for _ in range(n):
        np_v = np_v / (-U)
    gap = abs(cal_v - np_v)
    ok = np.isclose(cal_v, np_v, rtol=C64_RTOL, atol=1e-7)
    x7b = x7b and ok
    print(f"  {n:>4}{cal_v.real:>22.8f}{np_v.real:>20.8f}{gap:>12.2e}{str(ok):>20}")
    acc = quat_mul(acc, inv_u_t)

# 7c. The embedding invariants on the same scalar.
print("\n  7c. The M2(C) embedding and the reduced norm on the same rate, through the")
print("      real cal.biquat_to_matrix and cal.quat_norm_sq:\n")
lam_t = torch.tensor([-U, 0.0, 0.0, 0.0], dtype=CDTYPE)
lam_np = bq(-U)
M_cal = biquat_to_matrix(lam_t)
M_np = biquat_to_matrix_np(lam_np)
tr_cal = complex((M_cal[0, 0] + M_cal[1, 1]).item())
det_cal = complex((M_cal[0, 0]*M_cal[1, 1] - M_cal[0, 1]*M_cal[1, 0]).item())
nsq_cal = complex(quat_norm_sq(lam_t).item())
tr_np = M_np[0, 0] + M_np[1, 1]
det_np = M_np[0, 0]*M_np[1, 1] - M_np[0, 1]*M_np[1, 0]
print(f"    tr M(lambda)      cal = {tr_cal.real:+.6f}   numpy = {tr_np.real:+.6f}"
      f"   want 2*q0 = {2*(-U):+.6f}")
print(f"    det M(lambda)     cal = {det_cal.real:+.6f}   numpy = {det_np.real:+.6f}"
      f"   want u^2  = {U**2:+.6f}")
print(f"    quat_norm_sq      cal = {nsq_cal.real:+.6f}   (= det M, the reduced norm)")
x7c = (np.isclose(tr_cal, tr_np, rtol=C64_RTOL, atol=1e-6)
       and np.isclose(det_cal, det_np, rtol=C64_RTOL, atol=1e-6)
       and np.isclose(det_cal, nsq_cal, rtol=C64_RTOL, atol=1e-6)
       and np.isclose(tr_cal, 2*(-U), rtol=C64_RTOL, atol=1e-6))
print("\n    >> The rate is a NEGATIVE real scalar in the embedding: tr M = -2u < 0.")
print("       That is precisely the statement that lambda lands on the negative real")
print("       axis, which is where the principal cut of log lives. Section 2 is not an")
print("       artifact of a numpy convention; it is where cal's own embedding puts it.")

# 7d. The dagger survives the embedding, on the same object.
print("\n  7d. The dagger through the embedding, on the same rate (M(q_dag) = M(q)_dag):\n")
qd_cal = hermitian_conj(lam_t)
qd_np = hermitian_conj_np(lam_np)
M_qd_cal = biquat_to_matrix(qd_cal)
M_dag_cal = biquat_to_matrix(lam_t).conj().transpose(-2, -1)
x7d = (np.allclose(np.array(qd_cal.tolist()), qd_np, rtol=C64_RTOL, atol=1e-6)
       and torch.allclose(M_qd_cal, M_dag_cal, rtol=C64_RTOL, atol=1e-6))
print(f"    cal   hermitian_conj(lambda) = {[f'{v.real:+.1f}' for v in qd_cal.tolist()]}")
print(f"    numpy hermitian_conj(lambda) = {[f'{v.real:+.1f}' for v in qd_np]}")
print(f"    M(q_dagger) == M(q)_dagger    : {torch.allclose(M_qd_cal, M_dag_cal, rtol=C64_RTOL, atol=1e-6)}")
print("\n    >> lambda = -u is real, so the dagger fixes it: a negative REAL rate, with")
print("       nothing imaginary to hide the cut behind.")

print(f"\n  All cross-checks at rtol = {C64_RTOL:.0e}. This tolerance is FLOAT32-LIMITED:")
print("  cal is torch.complex64 and cannot resolve these to complex128 precision. The")
print("  findings above are complex128; this section only confirms cal agrees with them")
print("  to the precision cal actually has.")
record("B7 cross-check: cal.biquaternion reproduces sections 4-5 and the embedding "
       "invariants at float32-honest rtol 1e-5",
       x7a and x7b and x7c and x7d)


# ============================================================================
print("\n" + RULE)
print("SUMMARY")
print(RULE)
for name, ok in RESULTS:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"\n  {len(RESULTS) - n_fail} / {len(RESULTS)} sections PASS, {n_fail} FAIL.")
print("\n  THE RESULT. A power is exp of a log, and a log is a choice of branch. The")
print("  choice is invisible at integer order -- every branch gives the same number, so")
print("  the log cancels and never had to be declared. At fractional order the choice is")
print("  the answer: (-u)^{-alpha} is two different numbers, and the cost channel's rate")
print("  lambda = -u sits exactly on the cut where the two meet. alpha = 1.9965071 does")
print("  not denote a number until k is declared, and nothing in the framework declares")
print("  it. The integer-order route needs no declaration, and integration -- unlike")
print("  differentiation -- is the direction that stays bounded. An accumulation law")
print("  integrates, at integer order, for the same reason both times: it is the route")
print("  that does not require a convention nobody wrote down.")
print(RULE)

sys.exit(0 if n_fail == 0 else 1)
