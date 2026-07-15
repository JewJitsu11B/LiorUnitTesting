"""
exp_integrate_twice_period4.py

"Literally integrate the source current n times. Not an operator. The actual
 expression."

J(s) = A e^{-u s} + B e^{i w s}   -- one real-decaying channel, one complex-
oscillating channel, ADDED.

Claim under test: you do not need a fractional-order operator I^alpha, a
channel-angle axiom (theta = alpha pi/2), or an imported Z_4 from h^4 = 1 to
get the sign flip at "integrate twice" or the period-4 structure. Both fall
out of an integral sign applied to a complex exponential:

    INT e^{L s} ds = e^{L s} / L    ==>   n integrations divide by L^n.

    real channel    L = -u   ->  1/(-u)^n    sign has period 2
    complex channel L = i w  ->  1/(i w)^n   phase has period 4   <== the Z_4

FINDINGS ENCODED (each verified independently here, not copied):
  1. INT^1 = -A e^{-us}/u - i B e^{iws}/w
     INT^2 =  A e^{-us}/u^2 -  B e^{iws}/w^2    <- ADDED became SUBTRACTED
     INT^4 =  A e^{-us}/u^4 +  B e^{iws}/w^4    <- back to (+,+)
  2. The Z_4 is 1/i^n, DERIVED from integrating e^{i w s}, not posited.
  3. At closure u = w, stripping the common u^{-n}, the sign/phase structure
     cycles (+,+) -> (-,-i) -> (+,-) -> (-,+i) -> (+,+). Minimal period 4.
  4. At n=2 the channels are antiparallel (+,-). Prop 5.2, with no axiom.
  5. INT^2 inverts d^2/ds^2. "Integrate twice" IS "invert a 2nd-order operator".

PRECISION NOTE.
The findings are computed SYMBOLICALLY (sympy, exact) and in numpy complex128.
The cal cross-check section calls cal.biquaternion, which is torch.complex64
(eps ~ 1e-7); that section is checked at rtol 1e-5 and the tolerance is
float32-limited by the library dtype, NOT by the mathematics. No 1e-15 identity
is ever asserted through complex64.

Standalone:  python exp_integrate_twice_period4.py
Seeded, deterministic, ASCII only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp
import torch

from cal.biquaternion import quat_mul, biquat_to_matrix

RULE = "=" * 78
SEED = 20260715
C64_RTOL = 1e-5          # honest for torch.complex64; see PRECISION NOTE
EXACT_TOL = 1e-13        # numpy complex128, no cancelling differences here

results = {}


def verdict(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


# ---------------------------------------------------------------- symbols
s, u, w = sp.symbols('s u w', positive=True)
A, B = sp.symbols('A B')
J = A * sp.exp(-u * s) + B * sp.exp(sp.I * w * s)


print(RULE)
print("0. HANDEDNESS GUARD -- cal.quat_mul basis convention")
print(RULE)
print("  Asserting the basis is RIGHT-HANDED before any claim is built on it.")
print("  Required: i*j = k  and  i*j*k = -1.\n")

CD = torch.complex64
E = torch.tensor([1, 0, 0, 0], dtype=CD)
I_ = torch.tensor([0, 1, 0, 0], dtype=CD)
Jq = torch.tensor([0, 0, 1, 0], dtype=CD)
K_ = torch.tensor([0, 0, 0, 1], dtype=CD)

ij = quat_mul(I_, Jq)
ijk = quat_mul(ij, K_)
ii = quat_mul(I_, I_)
jj = quat_mul(Jq, Jq)
kk = quat_mul(K_, K_)
ji = quat_mul(Jq, I_)

print(f"  {'product':>8}   {'result (q0,q1,q2,q3)':<30}{'expected':>10}{'ok?':>8}")
rows = [("i*j", ij, K_, "k"),
        ("j*i", ji, -K_, "-k"),
        ("i*j*k", ijk, -E, "-1"),
        ("i*i", ii, -E, "-1"),
        ("j*j", jj, -E, "-1"),
        ("k*k", kk, -E, "-1")]
hand_ok = True
for nm, got, exp, lab in rows:
    match = torch.allclose(got, exp)
    hand_ok &= bool(match)
    vec = "(" + ", ".join(f"{v.real:+.0f}" for v in got.numpy()) + ")"
    print(f"  {nm:>8}   {vec:<30}{lab:>10}{str(bool(match)):>8}")
print()
results['0'] = verdict("basis is right-handed: i*j = k, ijk = -1, j*i = -k", hand_ok)
assert hand_ok, "HANDEDNESS GUARD FAILED -- every downstream claim is void"


print("\n" + RULE)
print("1. LITERALLY INTEGRATE J n TIMES. sympy. No operator, no I^alpha.")
print(RULE)
print(f"  J       = {J}\n")

int_n = [J]
cur = J
for n in range(1, 5):
    cur = sp.simplify(sp.integrate(cur, s))
    int_n.append(cur)
    print(f"  INT^{n} J = {sp.expand(cur)}")

expected = {
    1: -A * sp.exp(-u * s) / u - sp.I * B * sp.exp(sp.I * w * s) / w,
    2:  A * sp.exp(-u * s) / u**2 - B * sp.exp(sp.I * w * s) / w**2,
    3: -A * sp.exp(-u * s) / u**3 + sp.I * B * sp.exp(sp.I * w * s) / w**3,
    4:  A * sp.exp(-u * s) / u**4 + B * sp.exp(sp.I * w * s) / w**4,
}
print("\n  Checking each against the finding, symbolically (exact, not numeric):")
print(f"  {'n':>3}{'finding says':>46}{'agrees?':>12}")
sec1_ok = True
for n in range(1, 5):
    ok = sp.simplify(int_n[n] - expected[n]) == 0
    sec1_ok &= ok
    print(f"  {n:>3}{str(expected[n]):>46}{str(ok):>12}")
print()
print("  n=1: the complex channel picked up 1/i = -i.        ADDED -> quadrature")
print("  n=2: the complex channel picked up 1/i^2 = -1.      ADDED -> SUBTRACTED")
print("  n=4: the complex channel picked up 1/i^4 = +1.      back to (+,+)")
print("  Nothing acted on J. It was integrated. i did the rest.\n")
results['1'] = verdict("INT^1, INT^2, INT^3, INT^4 match the finding exactly", sec1_ok)


print("\n" + RULE)
print("2. WHY: n INTEGRATIONS DIVIDE e^{L s} BY L^n. THAT IS THE WHOLE MECHANISM.")
print(RULE)
L = sp.Symbol('L', nonzero=True)
gen = sp.exp(L * s)
gen_ok = True
for n in range(1, 5):
    got = sp.integrate(gen, s)
    gen = sp.simplify(got)
    tgt = sp.exp(L * s) / L**n
    ok = sp.simplify(gen - tgt) == 0
    gen_ok &= ok
    print(f"  INT^{n} e^(Ls) = {gen}      == e^(Ls)/L^{n} ? {ok}")

print()
verdict("2a: INT^n e^(Ls) == e^(Ls)/L^n for n=1..4 (sympy, exact)", gen_ok)

print("\n  Substitute the two channel rates and read off the coefficient:")
print(f"  {'n':>3}{'real  1/(-u)^n':>20}{'complex  1/(i w)^n':>24}"
      f"{'real sign':>12}{'complex phase':>16}")
# gen_ok MUST gate this section: the verdict below asserts the L^n identity,
# so a failure of that identity has to be able to fail the section.
sec2_ok = bool(gen_ok)
for n in range(9):
    r = sp.simplify(1 / (-u)**n)
    c = sp.simplify(1 / (sp.I * w)**n)
    r_sign = sp.simplify(r * u**n)          # strip magnitude -> +1 / -1
    c_ph = sp.simplify(c * w**n)            # strip magnitude -> power of i
    sec2_ok &= (r_sign in (sp.Integer(1), sp.Integer(-1)))
    sec2_ok &= (sp.simplify(c_ph - sp.simplify(1 / sp.I**n)) == 0)
    print(f"  {n:>3}{str(r):>20}{str(c):>24}{str(r_sign):>12}{str(c_ph):>16}")

print("\n  Period of each channel's stripped factor, measured (not assumed):")
r_seq = [sp.simplify((1 / (-u)**n) * u**n) for n in range(12)]
c_seq = [sp.simplify((1 / (sp.I * w)**n) * w**n) for n in range(12)]


def minimal_period(seq):
    for p in range(1, len(seq)):
        if all(seq[n] == seq[n + p] for n in range(len(seq) - p)):
            return p
    return None


pr = minimal_period(r_seq)
pc = minimal_period(c_seq)
print(f"    real channel    (-1)^n : minimal period = {pr}   expected 2")
print(f"    complex channel i^{{-n}} : minimal period = {pc}   expected 4")
sec2_ok &= (pr == 2 and pc == 4)
print("\n  >> The 4 is 1/i^n coming back to 1 after four integrations. It is DERIVED")
print("     from integrating a complex exponential. It is not h^4 = 1 imported from")
print("     the biquaternion algebra, and it is not a posited Z_4.\n")
results['2'] = verdict("INT^n e^(Ls) = e^(Ls)/L^n; real period 2, complex period 4",
                       sec2_ok)


print("\n" + RULE)
print("3. AT CLOSURE u = w: STRIP THE COMMON u^{-n}. ONLY SIGNS AND i-POWERS LEFT.")
print(RULE)
Jc = J.subs(w, u)
print(f"  J = {Jc}      (closure: the two rates have equal magnitude)\n")
print("  Multiply INT^n J by u^n to remove the shared magnitude decay, so that the")
print("  ONLY thing that survives is the sign/phase pair (real chan, complex chan).\n")
print(f"  {'n':>3}{'u^n INT^n J':>52}{'pair':>14}")

pairs = []
sec3_ok = True
v = Jc
for n in range(0, 9):
    if n > 0:
        v = sp.integrate(v, s)
    stripped = sp.simplify(sp.expand(v * u**n))
    r_c = sp.simplify(sp.Integer(-1)**n)
    c_c = sp.simplify(1 / sp.I**n)
    tgt = A * sp.exp(-u * s) * r_c + B * sp.exp(sp.I * u * s) * c_c
    ok = sp.simplify(stripped - tgt) == 0
    sec3_ok &= ok
    pairs.append((r_c, c_c))

    def tag(z):
        return {sp.Integer(1): '+', sp.Integer(-1): '-',
                sp.I: '+i', -sp.I: '-i'}.get(z, str(z))

    lab = f"({tag(r_c)},{tag(c_c)})"
    if n <= 4:
        print(f"  {n:>3}{str(sp.expand(stripped)):>52}{lab:>14}")
    if not ok:
        print(f"  {n:>3}  MISMATCH vs {tgt}")

print("\n  The cycle, read straight off the table:")
cyc = []
for n in range(5):
    r_c, c_c = pairs[n]
    t = {sp.Integer(1): '+', sp.Integer(-1): '-', sp.I: '+i', -sp.I: '-i'}
    cyc.append(f"({t[r_c]},{t[c_c]})")
print("    n=0 -> n=1 -> n=2 -> n=3 -> n=4")
print("    " + " -> ".join(cyc))
expect_cyc = ['(+,+)', '(-,-i)', '(+,-)', '(-,+i)', '(+,+)']
cyc_ok = (cyc == expect_cyc)
print(f"    finding says: {' -> '.join(expect_cyc)}")
print(f"    agrees? {cyc_ok}")

pmin = minimal_period(pairs)
print(f"\n  Minimal period of the PAIR, measured over n=0..8: {pmin}   expected 4")
print("  Not 2: at n=2 the real channel has returned (+) but the complex channel")
print(f"          has not -- it is {pairs[2][1]}, versus {pairs[0][1]} at n=0.")
per_ok = (pmin == 4) and (pairs[2] != pairs[0]) and (pairs[4] == pairs[0])

print("\n  Where is each channel REAL? (both real == the two 'corners')")
print(f"  {'n':>3}{'real chan real?':>18}{'complex chan real?':>22}{'both real?':>14}")
both_real = []
for n in range(4):
    r_c, c_c = pairs[n]
    rr = sp.im(r_c) == 0
    cr = sp.im(c_c) == 0
    both_real.append(bool(rr and cr))
    print(f"  {n:>3}{str(bool(rr)):>18}{str(bool(cr)):>22}{str(bool(rr and cr)):>14}")
real_ok = (both_real == [True, False, True, False])
print(f"\n  Both channels real at n in {{{', '.join(str(n) for n in range(4) if both_real[n])}}}"
      f" (mod 4). Finding says {{0, 2}}. agrees? {real_ok}")
sec3_ok = sec3_ok and cyc_ok and per_ok and real_ok
print()
results['3'] = verdict("closure cycle (+,+) -> (-,-i) -> (+,-) -> (-,+i) -> (+,+), "
                       "minimal period exactly 4, both real only at n in {0,2}",
                       sec3_ok)


print("\n" + RULE)
print("4. AT n=2 THE CHANNELS ARE ANTIPARALLEL. PROP 5.2, WITH NO AXIOM.")
print(RULE)
J2 = sp.simplify(sp.integrate(sp.integrate(J, s), s))
print(f"  INT^2 J = {sp.expand(J2)}\n")
r2 = sp.simplify(1 / (-u)**2)
c2 = sp.simplify(1 / (sp.I * w)**2)
print(f"  real channel    coefficient 1/(-u)^2  = {r2}    positive? "
      f"{sp.ask(sp.Q.positive(r2))}")
print(f"  complex channel coefficient 1/(i w)^2 = {c2}   negative? "
      f"{sp.ask(sp.Q.negative(c2))}")
anti_ok = bool(sp.ask(sp.Q.positive(r2))) and bool(sp.ask(sp.Q.negative(c2)))

print("\n  The relative sign between the two channels, before and after:")
print(f"  {'n':>3}{'rel. sign  (complex coeff)/(real coeff), stripped':>54}{'':>6}")
rel_ok = True
for n in (0, 2):
    rel = sp.simplify((1 / (sp.I * u)**n) / (1 / (-u)**n))
    print(f"  {n:>3}{str(rel):>54}")
    if n == 0 and rel != 1:
        rel_ok = False
    if n == 2 and rel != -1:
        rel_ok = False
print("\n  n=0: relative sign +1. The channels are ADDED (parallel).")
print("  n=2: relative sign -1. The channels are SUBTRACTED (ANTIPARALLEL).")
print("\n  Inventory of what was used to get the antiparallel configuration:")
print(f"  {'ingredient':<38}{'used?':>10}")
for ing in ["an integral sign", "1/i^2 = -1"]:
    print(f"  {ing:<38}{'YES':>10}")
for ing in ["a channel-angle axiom", "theta = alpha pi/2", "Axiom 3",
            "the operator I^alpha", "h^4 = 1 from the algebra"]:
    print(f"  {ing:<38}{'no':>10}")
print()
results['4'] = verdict("n=2 gives relative sign -1 (antiparallel) from 1/i^2 alone; "
                       "n=0 gives +1", anti_ok and rel_ok)


print("\n" + RULE)
print("5. IS 'INTEGRATE TWICE' THE SAME AS 'INVERT d^2/ds^2'? sympy, several f.")
print(RULE)


def INT2(f):
    return sp.integrate(sp.integrate(f, s), s)


def D2(f):
    return sp.diff(f, s, 2)


funcs = [("A e^{-u s}", A * sp.exp(-u * s)),
         ("B e^{i w s}", B * sp.exp(sp.I * w * s)),
         ("J (both channels)", J),
         ("sin(w s)", sp.sin(w * s)),
         ("s^3", s**3)]

print("  5a. RIGHT inverse: d^2/ds^2 ( INT^2 f ) == f ?\n")
print(f"  {'f':>22}{'d2(INT2 f) - f':>28}{'exact?':>10}")
r_inv_ok = True
for nm, f in funcs:
    d = sp.simplify(D2(INT2(f)) - f)
    ok = (d == 0)
    r_inv_ok &= ok
    print(f"  {nm:>22}{str(d):>28}{str(ok):>10}")
verdict("5a: INT^2 is an EXACT right inverse of d^2/ds^2 for every f tested",
        r_inv_ok)

print("\n  5b. LEFT inverse: INT^2 ( d^2/ds^2 f ) == f ?\n")
print(f"  {'f':>22}{'INT2(d2 f) - f':>28}{'exact?':>10}")
l_inv_ok = True
for nm, f in funcs:
    d = sp.simplify(INT2(D2(f)) - f)
    ok = (d == 0)
    l_inv_ok &= ok
    print(f"  {nm:>22}{str(d):>28}{str(ok):>10}")
verdict("5b: INT^2 is also an exact LEFT inverse on the functions tested above",
        l_inv_ok)

print("\n  5c. BUT: the unrestricted claim is FALSE. d^2/ds^2 has a kernel.")
print("      ker(d^2/ds^2) = span{1, s}. Anything in it is destroyed and cannot")
print("      be recovered by integrating. Explicit counterexample:\n")
f_bad = s**3 + s + 1
d_bad = sp.simplify(INT2(D2(f_bad)) - f_bad)
print(f"  {'f':>22}{'INT2(d2 f) - f':>28}{'exact?':>10}")
print(f"  {'s^3 + s + 1':>22}{str(d_bad):>28}{str(d_bad == 0):>10}")
counterexample_found = (d_bad != 0)
verdict("5c: the UNRESTRICTED claim 'INT^2 == (d^2/ds^2)^{-1} for all f' is FALSE "
        "(counterexample above)", counterexample_found)

print("\n  >> FINDING 5 AS LITERALLY STATED ('INT^2 is exactly the inverse of")
print("     d^2/ds^2, for several f') IS IMPRECISE. The precise, true statement:")
print("       - INT^2 is an EXACT right inverse of d^2/ds^2, always (5a).")
print("       - It is a two-sided inverse exactly on functions with no component")
print("         in ker(d^2/ds^2) = span{1, s}.")
print("     Both channel exponentials e^{-us} and e^{iws} have zero component in")
print("     span{1, s} (u, w > 0 and nonzero), so on the ACTUAL source current J")
print("     the two-sided inverse holds and the finding's intent stands: for J,")
print("     'integrate twice' IS 'invert a second-order operator'. The overreach")
print("     is the words 'for several f' generalising to 'for all f'.\n")
sec5_ok = r_inv_ok and l_inv_ok and counterexample_found
results['5'] = verdict("INT^2 inverts d^2/ds^2 on J and on every kernel-free f tested; "
                       "unrestricted form correctly refuted", sec5_ok)


print("\n" + RULE)
print("6. NUMERIC CONFIRMATION IN numpy complex128 (independent of sympy)")
print(RULE)
rng = np.random.default_rng(SEED)

u_v = float(rng.uniform(0.5, 2.0))
w_v = float(rng.uniform(0.5, 2.0))
A_v = complex(rng.normal(), rng.normal())
B_v = complex(rng.normal(), rng.normal())
s_grid = np.linspace(0.1, 3.0, 7)

print(f"  seed = {SEED}   (numpy default_rng)")
print(f"  u = {u_v:.6f}   w = {w_v:.6f}")
print(f"  A = {A_v:.6f}   B = {B_v:.6f}\n")

f_num = sp.lambdify((s, u, w, A, B), J, 'numpy')
J_num = f_num(s_grid, u_v, w_v, A_v, B_v)

print("  Evaluate the CLOSED FORMS from section 1 and compare to the direct")
print("  coefficient rule  coeff_n = 1/L^n  applied to each channel:\n")
print(f"  {'n':>3}{'max |sympy INT^n - (A/(-u)^n) e^-us - (B/(iw)^n) e^iws|':>60}"
      f"{'ok?':>8}")
sec6_ok = True
for n in range(0, 5):
    g = sp.lambdify((s, u, w, A, B), int_n[n], 'numpy')
    lhs = np.asarray(g(s_grid, u_v, w_v, A_v, B_v), dtype=np.complex128)
    rhs = (A_v / (-u_v)**n) * np.exp(-u_v * s_grid) \
        + (B_v / (1j * w_v)**n) * np.exp(1j * w_v * s_grid)
    err = float(np.abs(lhs - rhs).max())
    ok = err < EXACT_TOL
    sec6_ok &= ok
    print(f"  {n:>3}{err:>60.3e}{str(ok):>8}")

print("\n  The n=2 flip, numerically, on the B channel coefficient:")
c0 = B_v / (1j * w_v)**0
c2 = B_v / (1j * w_v)**2
ratio = c2 / c0
print(f"    n=0 B-coefficient = {c0:.6f}")
print(f"    n=2 B-coefficient = {c2:.6f}")
print(f"    ratio (n=2)/(n=0) = {ratio:.6f}")
print(f"    is the ratio a NEGATIVE REAL? {bool(abs(ratio.imag) < EXACT_TOL and ratio.real < 0)}")
print(f"    equals -1/w^2 = {-1.0 / w_v**2:.6f} ? "
      f"{bool(abs(ratio + 1.0 / w_v**2) < EXACT_TOL)}")
flip_ok = bool(abs(ratio.imag) < EXACT_TOL and ratio.real < 0
               and abs(ratio + 1.0 / w_v**2) < EXACT_TOL)

a0 = A_v / (-u_v)**0
a2 = A_v / (-u_v)**2
ar = a2 / a0
print(f"\n    the A channel, same test: ratio (n=2)/(n=0) = {ar:.6f}")
print(f"    is the ratio a POSITIVE REAL? {bool(abs(ar.imag) < EXACT_TOL and ar.real > 0)}")
noflip_ok = bool(abs(ar.imag) < EXACT_TOL and ar.real > 0)
print("\n  >> The A channel keeps its sign at n=2; the B channel flips. That is the")
print("     antiparallel configuration, arrived at with float arithmetic and no")
print("     symbolic algebra. Tolerance here is complex128 and NOT float32-limited.\n")
sec6_ok = sec6_ok and flip_ok and noflip_ok
results['6'] = verdict("numpy complex128 reproduces INT^n for n=0..4 and the n=2 "
                       "B-channel flip / A-channel no-flip", sec6_ok)


print("\n" + RULE)
print("7. CAL CROSS-CHECK -- BIQUATERNION-VALUED A AND B (torch.complex64)")
print(RULE)
print("  This finding is scalar calculus, so the cal surface it touches is thin.")
print("  What IS worth checking: integration acts componentwise and C-linearly, so")
print("  a biquaternion-valued source J(s) = A e^{-us} + B e^{iws} with A, B in")
print("  H_C must show the SAME n=2 flip in every one of its 4 components, and the")
print("  flip must survive cal's M2(C) embedding (which is C-linear in q).")
print("  No quat_mul is needed or used below -- the arithmetic is componentwise.\n")
print("  TOLERANCE: cal is torch.complex64 (eps ~ 1e-7). The comparisons below are")
print(f"  made at rtol = {C64_RTOL:g}. That bound is FLOAT32-LIMITED BY THE LIBRARY")
print("  DTYPE, not by the mathematics -- sections 1-6 above establish the result")
print("  exactly (sympy) and at 1e-13 (complex128). No 1e-15 identity is asserted")
print("  through complex64 anywhere in this section.\n")

# Draw A and B ONCE each; reuse the same objects everywhere below.
A_bq_np = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(np.complex128)
B_bq_np = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(np.complex128)
A_bq = torch.tensor(A_bq_np, dtype=CD)
B_bq = torch.tensor(B_bq_np, dtype=CD)

print("  A (biquaternion) =", np.array2string(A_bq_np, precision=4))
print("  B (biquaternion) =", np.array2string(B_bq_np, precision=4))
print(f"  u = {u_v:.6f}   w = {w_v:.6f}\n")


def channel_coeffs_np(n):
    """Coefficient biquaternions after n integrations, complex128 reference."""
    return A_bq_np / (-u_v)**n, B_bq_np / (1j * w_v)**n


def channel_coeffs_cal(n):
    """Same, built with torch/cal dtype. Componentwise; no quat_mul."""
    a = A_bq / torch.tensor((-u_v)**n, dtype=CD)
    b = B_bq / torch.tensor((1j * w_v)**n, dtype=CD)
    return a, b


print("  7a. cal (complex64) coefficients vs numpy (complex128) reference:\n")
print(f"  {'n':>3}{'max rel err A-chan':>24}{'max rel err B-chan':>24}"
      f"{'within rtol?':>14}")
cal_ok = True
for n in range(0, 5):
    an, bn = channel_coeffs_np(n)
    ac, bc = channel_coeffs_cal(n)
    ea = float(np.abs(ac.numpy().astype(np.complex128) - an).max()
               / np.abs(an).max())
    eb = float(np.abs(bc.numpy().astype(np.complex128) - bn).max()
               / np.abs(bn).max())
    ok = (ea < C64_RTOL) and (eb < C64_RTOL)
    cal_ok &= ok
    print(f"  {n:>3}{ea:>24.3e}{eb:>24.3e}{str(ok):>14}")

print("\n  7b. The n=2 flip, COMPONENTWISE, in cal's dtype.")
print("      For each mu: coeff_mu(n=2) / coeff_mu(n=0) must be")
print("      +1/u^2 (positive real) on the A channel, -1/w^2 (negative real) on B.\n")
a0c, b0c = channel_coeffs_cal(0)
a2c, b2c = channel_coeffs_cal(2)
ra = (a2c / a0c).numpy().astype(np.complex128)
rb = (b2c / b0c).numpy().astype(np.complex128)
print(f"  {'mu':>4}{'A ratio':>26}{'sign':>10}{'B ratio':>26}{'sign':>10}")
comp_ok = True
for mu in range(4):
    sa = 'pos' if (abs(ra[mu].imag) < C64_RTOL and ra[mu].real > 0) else 'BAD'
    sbn = 'NEG' if (abs(rb[mu].imag) < C64_RTOL and rb[mu].real < 0) else 'BAD'
    ok_mu = (sa == 'pos') and (sbn == 'NEG')
    ok_mu &= bool(abs(ra[mu] - 1.0 / u_v**2) < C64_RTOL * abs(1.0 / u_v**2))
    ok_mu &= bool(abs(rb[mu] + 1.0 / w_v**2) < C64_RTOL * abs(1.0 / w_v**2))
    comp_ok &= ok_mu
    print(f"  {mu:>4}{str(np.round(ra[mu], 6)):>26}{sa:>10}"
          f"{str(np.round(rb[mu], 6)):>26}{sbn:>10}")
print(f"\n    target A ratio = +1/u^2 = {1.0 / u_v**2:+.6f}")
print(f"    target B ratio = -1/w^2 = {-1.0 / w_v**2:+.6f}")
print(f"    all 4 components flip on B and hold on A? {comp_ok}")

print("\n  7c. The flip survives cal's M2(C) embedding (biquat_to_matrix is")
print("      C-linear in q, so M(INT^2 J) must equal the same combination of")
print("      M(A) and M(B)):\n")
s_probe = 0.7
J2_bq = A_bq / torch.tensor(u_v**2, dtype=CD) * np.exp(-u_v * s_probe) \
    - B_bq / torch.tensor(w_v**2, dtype=CD) * np.exp(1j * w_v * s_probe)
M_lhs = biquat_to_matrix(J2_bq).numpy().astype(np.complex128)
M_rhs = biquat_to_matrix(A_bq).numpy().astype(np.complex128) \
    * (np.exp(-u_v * s_probe) / u_v**2) \
    - biquat_to_matrix(B_bq).numpy().astype(np.complex128) \
    * (np.exp(1j * w_v * s_probe) / w_v**2)
emb_err = float(np.abs(M_lhs - M_rhs).max() / np.abs(M_rhs).max())
print(f"    s = {s_probe}")
print(f"    max rel | M(INT^2 J) - [ M(A) e^-us/u^2 - M(B) e^iws/w^2 ] | = {emb_err:.3e}")
print(f"    within rtol {C64_RTOL:g} (float32-limited)? {emb_err < C64_RTOL}")
emb_ok = emb_err < C64_RTOL
print("\n    Note the MINUS in the bracket: the subtraction is carried into M2(C)")
print("    unchanged, because M is linear. The n=2 antiparallel configuration is")
print("    not an artifact of the scalar presentation.\n")
sec7_ok = cal_ok and comp_ok and emb_ok
results['7'] = verdict("cal complex64 agrees with the complex128 reference at "
                       "rtol 1e-5; the n=2 flip is componentwise and survives "
                       "biquat_to_matrix", sec7_ok)


print("\n" + RULE)
print("8. SUMMARY")
print(RULE)
labels = {
    '0': "handedness guard (i*j = k, ijk = -1) via cal.quat_mul",
    '1': "INT^1..INT^4 match the finding exactly (sympy)",
    '2': "n integrations divide by L^n; real period 2, complex period 4",
    '3': "closure cycle (+,+)->(-,-i)->(+,-)->(-,+i)->(+,+), min period 4",
    '4': "n=2 antiparallel from 1/i^2 = -1, no axiom used",
    '5': "INT^2 inverts d^2/ds^2 on J (unrestricted form refuted)",
    '6': "numpy complex128 numeric confirmation, tol 1e-13",
    '7': "cal.biquaternion cross-check, rtol 1e-5 (float32-limited)",
}
for k in sorted(results):
    print(f"  [{'PASS' if results[k] else 'FAIL'}] section {k}: {labels[k]}")

n_fail = sum(1 for v in results.values() if not v)
print()
print(f"  sections: {len(results)}   passed: {len(results) - n_fail}   failed: {n_fail}")
print()
print("  WHAT THIS ESTABLISHES")
print("  --------------------")
print("  The Z_4 is not an axiom and not an import from h^4 = 1. Integrating a")
print("  complex exponential n times multiplies it by 1/(i w)^n, and 1/i^n has")
print("  period 4. The real channel's 1/(-u)^n has period 2. At closure u = w the")
print("  common magnitude strips out and the pair cycles with minimal period 4,")
print("  measured. At n=2 the complex channel's factor is 1/i^2 = -1 while the real")
print("  channel's is +1: the two channels, which were ADDED in J, are SUBTRACTED in")
print("  INT^2 J. That is Prop 5.2, and it required an integral sign and i^2 = -1.")
print("  It did not require a channel-angle axiom, theta = alpha pi/2, Axiom 3, or")
print("  the operator I^alpha.")
print()
print("  WHAT THIS DOES NOT ESTABLISH")
print("  ----------------------------")
print("  Finding 5 as worded overreaches. INT^2 is an exact RIGHT inverse of")
print("  d^2/ds^2 always, but a two-sided inverse only on functions with no")
print("  component in ker(d^2/ds^2) = span{1, s} (see 5c: f = s^3 + s + 1 is not")
print("  recovered). The channel exponentials are kernel-free, so the claim holds")
print("  where the finding actually uses it -- but 'for several f' must not be read")
print("  as 'for all f'.")
print()
print("  Also: the period-4 statement is exact for the SIGN/PHASE structure only.")
print("  It is exact for the full coefficient 1/(i w)^n only when w = 1; otherwise")
print("  the magnitude w^{-n} keeps shrinking and n=4 is n=0 attenuated by w^{-4}.")
print("  The stripping by u^n in section 3 is what makes the period exact, and that")
print("  strip is a deliberate step, not a free one.")
print(RULE)

sys.exit(0 if n_fail == 0 else 1)
