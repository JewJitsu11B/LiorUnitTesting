"""
Is <W^dag W>_0 really "the unique real, non-negative, conjugation-invariant quadratic form
on C (x) H"?  No. And the obvious repair needs one more generator than you would guess.

born_axiomatic.tex Lemma 4.3 (lem:unique) claims uniqueness up to positive scale, and its
proof says "This is a statement about the algebra alone", closing with "No further
independent quadratic combination of W with its conjugates produces a real non-negative
scalar not proportional to the dagger form." The only invariance the proof actually invokes
is invariance under the dagger involution itself:
    <(W^dag)^dag W^dag>_0 = <W W^dag>_0,
which is a Z_2. A Z_2 cannot pin a form up to scale, and it does not.

Throughout, "quadratic form" means REAL-quadratic on the 8-real-dimensional carrier
C (x) H = R^8. That is the only reading under which the dagger form itself qualifies:
sum_n |W_n|^2 is not C-bilinear either. So the counterexamples below are in exactly the
same class as the form the lemma is defending, and are fair game.

CLAIMS VERIFIED (against cal.biquaternion, the package under test):

  NEGATIVE (reported as a negative, kept as a negative):
  U1  The family F_ab(q) = a|q0|^2 + b(|q1|^2+|q2|^2+|q3|^2), a=2, b=1, is real,
      non-negative, and POSITIVE DEFINITE (min over the unit sphere = min(a,b) = 1 > 0).
  U2  F_21 is invariant under the bar involution, under the dagger involution, AND under
      quaternionic rotation q -> u q u^dag with u a unit quaternion -- yet the ratio
      F_21/dagger VARIES across random q, so it is not the dagger form rescaled.
      Every invariance Lemma 4.3 states is satisfied. The uniqueness claim is FALSE.
  W1  Why conjugation cannot see it: q -> u q u^dag FIXES q_0 exactly and acts on
      (q1,q2,q3) by a real SO(3) matrix (verified: R^T R = I, det R = +1). It never moves
      weight between the q_0 slot and the q_k slots, so any form that weights those two
      groups separately survives it. That is the whole reason the counterexample lives.

  REPAIR:
  R1  The missing generator is LEFT multiplication q -> u q (the spinor action). Clean
      witness: q = 1, u = i_q, so u*q = i_q. Dagger form 1 -> 1 (invariant); F_21 2 -> 1
      (NOT invariant). Left multiplication carries q_0 into the q_1 slot; conjugation
      cannot.
  R2  Over 2000 random unit quaternions: dagger form invariant under q -> u q, F_21 not.
      Reason, one line: <(uq)^dag (uq)>_0 = <q^dag u^dag u q>_0 and u^dag u = |u|^2 = 1.
  R3  Scanning a with b=1: only a = b survives left multiplication.

  NEGATIVE ON THE REPAIR (this is the part that is not in the source scripts):
  R4  The repair is INSUFFICIENT if "unit quaternion" means a unit REAL quaternion (SU(2),
      the spinor action). A second family survives it:
          G_ab(q) = a||Re q||^2 + b||Im q||^2,  a=2, b=1
      is real, non-negative, positive definite, bar- and dagger-invariant, invariant under
      q -> u q u^dag, AND invariant under q -> u q for every unit REAL quaternion u --
      because a real u acts on Re q and Im q separately. It is still not proportional to
      the dagger form.
      G_ab is killed by the h-phase u = h = (h,0,0,0), which satisfies u^dag u = 1 but
      N(u) = -1 (h is dagger-unit, not bar-unit). So the group that must be named is the
      DAGGER-UNIT group {u : u^dag u = 1} = U(2), i.e. SU(2) TIMES the global phase
      e^{h phi}, not SU(2) alone. The one-line proof in R2 is unchanged; only the naming
      of the group needs fixing.

  DIMENSION COUNT (the uniqueness claim, made exact):
  D1  Dimension of the space of real quadratic forms on R^8 invariant under successively
      larger groups, by Reynolds/SVD nullspace count (clean gap: ~2e+00 vs ~2e-07):
          dagger involution alone (what Lemma 4.3 states)   dim = 20   <- not 1
          + bar involution                                  dim = 14
          + conjugation q -> u q u^dag                      dim =  4
          + left mult by unit REAL quaternions (SU(2))      dim =  2   <- still not 1
          + left mult by dagger-unit u (u^dag u = 1)        dim =  1   <- uniqueness
      The lemma asserts dim = 1 from a group under which the true answer is 20.
  D2  MINIMAL repair: the lemma's own dagger-involution invariance PLUS left multiplication
      by the dagger-unit group already gives dim = 1, with no bar and no conjugation
      needed. The surviving form is recovered numerically and IS the identity form
      sum_n |q_n|^2 = <q^dag q>_0. Reason the involution earns its keep: the dagger
      conjugates left multiplication into RIGHT multiplication, ((u q)^dag = q^dag u^dag),
      so Z_2 + left-U(2) generates the two-sided unitary action, and bi-unitary invariance
      forces the Frobenius/dagger form. Left multiplication ALONE leaves dim = 4.

THE COST, stated honestly: requiring invariance under the dagger-unit group is requiring
UNITARITY (the spinor action plus global phase invariance). That is a PHYSICAL premise, not
an algebraic fact, so the sentence "This is a statement about the algebra alone" cannot be
kept -- the algebra alone gives dim = 20, not dim = 1. But the premise is one line, it is
far cheaper than Gleason, and unlike the paper's Prop 5.2 route it is actually true. Name
the group and the lemma is repaired.

Not covered here (see exp_two_closures.py, C1-C7): multiplicativity of the reduced norm, the
null cone, the bar-vs-dagger interference contrast. See exp_clifford_conjugation_
identification.py (A1-A7) for bar == Clifford conjugation and dagger == reversion. This
script adds only the uniqueness/invariance-group analysis of the Born form.
"""
import os
import sys

import numpy as np
import torch

# cal/ lives at the repo root, one level above "py tests"; this script is run from inside
# "py tests" as: python exp_born_form_uniqueness.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cal.biquaternion import (quat_mul, quat_conj, hermitian_conj, quat_norm_sq,
                              biquat_to_matrix, CDTYPE)

torch.manual_seed(0)
rng = np.random.default_rng(43)
np.set_printoptions(precision=6, suppress=True)

TOL = 1e-3        # cal is complex64 (float32 precision); forms here are O(1..30)
NSAMP = 2000
A_W, B_W = 2.0, 1.0     # the counterexample weights: a != b


def ok(b):
    return "PASS" if b else "**FAIL**"


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def rand_biquat(n):
    return (torch.randn(n, 4) + 1j * torch.randn(n, 4)).to(CDTYPE)


def rand_real_unit_quat(n):
    """Unit REAL quaternion: |u| = 1, u^dag = u_bar = u^-1. This is SU(2)."""
    v = torch.randn(n, 4)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.to(CDTYPE)


def rand_dagger_unit_quat(n):
    """u with u^dag u = 1, i.e. U(2) = {e^{h phi} * (unit real quaternion)}."""
    v = rand_real_unit_quat(n)
    phi = torch.rand(n, 1) * 2 * np.pi
    return (torch.exp(1j * phi.to(CDTYPE)) * v).to(CDTYPE)


# --- the three forms -------------------------------------------------------
def dagger_form(q):
    """<q^dag q>_0, computed with cal's own product and conjugation."""
    return quat_mul(hermitian_conj(q), q)[..., 0].real


def weighted_form(q, a=A_W, b=B_W):
    """F_ab(q) = a|q0|^2 + b(|q1|^2+|q2|^2+|q3|^2)."""
    m = q.real ** 2 + q.imag ** 2
    return a * m[..., 0] + b * m[..., 1:].sum(-1)


def reim_form(q, a=A_W, b=B_W):
    """G_ab(q) = a||Re q||^2 + b||Im q||^2."""
    return a * (q.real ** 2).sum(-1) + b * (q.imag ** 2).sum(-1)


def inv_under(form, q, qt):
    """Is `form` invariant on the pair (q, transformed q)?  -> (bool, max abs diff)"""
    d = (form(q) - form(qt)).abs().max().item()
    return bool(torch.allclose(form(q), form(qt), atol=TOL, rtol=TOL)), d


# --- real 8-dim coordinates, for the dimension count -----------------------
def to_real(q):
    a = q.detach().numpy()
    return np.stack([a.real, a.imag], -1).reshape(*a.shape[:-1], 8)


def from_real(x):
    y = np.asarray(x).reshape(*np.shape(x)[:-1], 4, 2)
    return torch.tensor(y[..., 0] + 1j * y[..., 1], dtype=CDTYPE)


def real_mat(f):
    """8x8 real matrix of an R-linear map f on C (x) H (built by acting on a basis)."""
    return to_real(f(from_real(np.eye(8)))).T


def main():
    q = rand_biquat(NSAMP)
    u_real = rand_real_unit_quat(NSAMP)

    # =======================================================================
    banner("U1. The counterexample family is real, non-negative, POSITIVE DEFINITE")
    print(f"  Candidate form  F_ab(q) = {A_W}|q0|^2 + {B_W}(|q1|^2+|q2|^2+|q3|^2),  a != b\n")
    Fq = weighted_form(q)
    real_nonneg = bool((Fq >= 0).all())
    # positive definite: minimum over the unit sphere must be min(a,b) > 0
    qn = q / torch.sqrt(dagger_form(q)).unsqueeze(-1)
    fmin = weighted_form(qn).min().item()
    pos_def = fmin > 0.99 * min(A_W, B_W)
    print(f"  real and >= 0 on {NSAMP} random biquaternions:            {ok(real_nonneg)}"
          f"   min F = {Fq.min():.4f}")
    print(f"  positive definite: min F over ||q||_dag = 1 is min(a,b):  {ok(pos_def)}"
          f"   min = {fmin:.4f}  (expect {min(A_W, B_W):.2f})")
    print("  >> So F is exactly the kind of object Lemma 4.3 says cannot exist besides the")
    print("     dagger form: real, non-negative, and vanishing only at q = 0.")

    # =======================================================================
    banner("U2. NEGATIVE: F_21 satisfies EVERY invariance Lemma 4.3 states, and is not")
    print("    proportional to the dagger form. The uniqueness claim is false.\n")
    q_bar = quat_conj(q)
    q_dag = hermitian_conj(q)
    q_conj = quat_mul(quat_mul(u_real, q), hermitian_conj(u_real))   # u q u^dag

    checks = [
        ("bar involution      q -> q_bar", q_bar),
        ("dagger involution   q -> q^dag", q_dag),
        ("rotation            q -> u q u^dag, |u| = 1", q_conj),
    ]
    print(f"  {'transformation':<46}{'dagger form':>14}{'F_21':>14}")
    all_inv = True
    for name, qt in checks:
        di, dd = inv_under(dagger_form, q, qt)
        wi, wd = inv_under(weighted_form, q, qt)
        all_inv &= wi
        print(f"  {name:<46}{ok(di):>14}{ok(wi):>14}")
    print(f"\n  F_21 invariant under all three (the lemma's own list):  {ok(all_inv)}")

    print("\n  Is F_21 just the dagger form rescaled?  Ratio F_21/dagger on random q:")
    qs = rand_biquat(6)
    ratios = (weighted_form(qs) / dagger_form(qs)).numpy()
    print(f"    {np.round(ratios, 4)}")
    const = bool(np.allclose(ratios, ratios[0], atol=1e-3))
    print(f"    constant ratio (would mean 'proportional')?             {const}")
    print(f"    ratio spread: min {ratios.min():.4f}  max {ratios.max():.4f}"
          f"  (bounded by b={B_W:.0f} and a={A_W:.0f}, as it must be)")
    u2 = all_inv and not const
    print(f"\n  U2 negative finding reproduced (uniqueness claim FALSE):  {ok(u2)}")

    # =======================================================================
    banner("W1. WHY conjugation is blind to it: q -> u q u^dag fixes q_0, rotates the rest")
    u1 = rand_real_unit_quat(1)
    qq = rand_biquat(NSAMP)
    qr = quat_mul(quat_mul(u1.expand(NSAMP, 4), qq), hermitian_conj(u1).expand(NSAMP, 4))
    d0 = (qr[..., 0] - qq[..., 0]).abs().max().item()
    scal_fixed = d0 < 1e-4
    vnorm_q = (qq[..., 1:].real ** 2 + qq[..., 1:].imag ** 2).sum(-1)
    vnorm_r = (qr[..., 1:].real ** 2 + qr[..., 1:].imag ** 2).sum(-1)
    vnorm_kept = bool(torch.allclose(vnorm_q, vnorm_r, atol=TOL, rtol=TOL))
    print(f"  scalar slot q_0 EXACTLY unchanged:                       {ok(scal_fixed)}"
          f"   max|dq_0| = {d0:.2e}")
    print(f"  |q1|^2+|q2|^2+|q3|^2 unchanged:                          {ok(vnorm_kept)}"
          f"   max|diff| = {(vnorm_q - vnorm_r).abs().max():.2e}")

    # recover the 3x3 real matrix that conjugation induces on the vector part
    R = np.zeros((3, 3))
    for k in range(3):
        e = torch.zeros(1, 4, dtype=CDTYPE)
        e[0, k + 1] = 1.0
        img = quat_mul(quat_mul(u1, e), hermitian_conj(u1))[0]
        R[:, k] = img[1:].real.numpy()
    orth = np.allclose(R.T @ R, np.eye(3), atol=1e-4)
    detR = float(np.linalg.det(R))
    rot = orth and abs(detR - 1.0) < 1e-3
    nontrivial = not np.allclose(R, np.eye(3), atol=1e-2)
    print(f"\n  induced action on (q1,q2,q3) is a real 3x3 matrix R(u):")
    print("   ", np.array2string(R, prefix="    "))
    print(f"    R^T R = I:          {ok(orth)}")
    print(f"    det R = {detR:+.6f}:  {ok(abs(detR - 1.0) < 1e-3)}   -> R is in SO(3):  {ok(rot)}")
    print(f"    R is not the identity (the test is not vacuous):       {ok(nontrivial)}")
    print("\n  >> Conjugation is (fixed q_0) (x) SO(3) on the vector part. It NEVER moves")
    print("     weight between the q_0 slot and the q_k slots. Any form that weights those")
    print("     two groups separately is automatically conjugation-invariant. That is")
    print("     exactly what F_ab does, and exactly why it survives.")

    # =======================================================================
    banner("R1. THE REPAIR: LEFT multiplication q -> u q. Clean witness q = 1, u = i_q")
    q_one = torch.tensor([[1.0 + 0j, 0, 0, 0]], dtype=CDTYPE)
    i_q = torch.tensor([[0j, 1.0 + 0j, 0, 0]], dtype=CDTYPE)
    uq = quat_mul(i_q, q_one)
    print(f"  q       = {to_real(q_one)[0][::2]}  (real coeffs (q0,q1,q2,q3))")
    print(f"  i_q * q = {to_real(uq)[0][::2]}  (left mult carried q_0 -> q_1 slot)\n")
    print(f"  {'':16}{'dagger form':>14}{'F_21':>10}")
    print(f"  {'q = 1':16}{dagger_form(q_one).item():>14.2f}{weighted_form(q_one).item():>10.2f}")
    print(f"  {'i_q * q = i_q':16}{dagger_form(uq).item():>14.2f}{weighted_form(uq).item():>10.2f}")
    wit_dag = abs(dagger_form(q_one).item() - dagger_form(uq).item()) < 1e-4
    wit_w = abs(weighted_form(q_one).item() - weighted_form(uq).item()) > 0.5
    print(f"\n  dagger form 1 -> 1, invariant:                           {ok(wit_dag)}")
    print(f"  F_21        2 -> 1, NOT invariant:                       {ok(wit_w)}")
    print("  >> One element of the group is enough to kill the counterexample.")

    # =======================================================================
    banner("R2. Systematically: 2000 random unit quaternions, q -> u q")
    q_left = quat_mul(u_real, q)
    di, dd = inv_under(dagger_form, q, q_left)
    wi, wd = inv_under(weighted_form, q, q_left)
    print(f"  dagger form IS invariant under q -> u q:                 {ok(di)}"
          f"   max|diff| = {dd:.2e}")
    print(f"  F_21 is NOT invariant (counterexample dies):             {ok(not wi)}"
          f"   max|diff| = {wd:.3f}")
    print("\n  Reason, one line: <(uq)^dag (uq)>_0 = <q^dag u^dag u q>_0 and u^dag u = 1.")
    uu = quat_mul(hermitian_conj(u_real), u_real)
    uu_one = bool(torch.allclose(uu[..., 0].real, torch.ones(NSAMP), atol=TOL)) and \
        bool(uu[..., 1:].abs().max() < TOL)
    print(f"  u^dag u = 1 verified on all {NSAMP} sampled u:             {ok(uu_one)}"
          f"   max|u^dag u - 1| = {(uu - torch.tensor([1.0+0j,0,0,0],dtype=CDTYPE)).abs().max():.2e}")

    # =======================================================================
    banner("R3. Scanning the family: only a = b survives left multiplication")
    print(f"  {'a (with b=1)':>14}{'max|F(q)-F(uq)|':>20}{'invariant?':>14}")
    surv = []
    for a in (0.5, 0.9, 1.0, 1.1, 2.0, 5.0):
        f = lambda x, _a=a: weighted_form(x, _a, 1.0)
        i, d = inv_under(f, q, q_left)
        surv.append((a, i))
        print(f"  {a:>14.2f}{d:>20.4f}{ok(i) if i else 'no':>14}")
    only_ab = all((i and abs(a - 1.0) < 1e-9) or (not i and abs(a - 1.0) > 1e-9)
                  for a, i in surv)
    print(f"\n  exactly the a = b member survives:                       {ok(only_ab)}")
    print("  >> Within this family, left multiplication forces a = b and lands on the")
    print("     dagger form. But 'within this family' is doing work -- see R4.")

    # =======================================================================
    banner("R4. NEGATIVE ON THE REPAIR: SU(2) alone is NOT enough. A second family lives.")
    print(f"  G_ab(q) = {A_W}||Re q||^2 + {B_W}||Im q||^2   (a != b again)\n")
    Gq = reim_form(q)
    g_nonneg = bool((Gq >= 0).all())
    gmin = reim_form(qn).min().item()
    g_posdef = gmin > 0.99 * min(A_W, B_W)
    print(f"  real and >= 0:                                           {ok(g_nonneg)}")
    print(f"  positive definite (min over unit sphere = min(a,b)):     {ok(g_posdef)}"
          f"   min = {gmin:.4f}")

    print(f"\n  {'transformation':<46}{'dagger':>10}{'F_21':>10}{'G_21':>10}")
    for name, qt in [("bar involution      q -> q_bar", q_bar),
                     ("dagger involution   q -> q^dag", q_dag),
                     ("rotation            q -> u q u^dag", q_conj),
                     ("LEFT MULT (repair)  q -> u q, u real unit", q_left)]:
        r = []
        for f in (dagger_form, weighted_form, reim_form):
            i, _ = inv_under(f, q, qt)
            r.append("yes" if i else "NO")
        print(f"  {name:<46}{r[0]:>10}{r[1]:>10}{r[2]:>10}")

    g_survives, _ = inv_under(reim_form, q, q_left)
    gr = (reim_form(qs) / dagger_form(qs)).numpy()
    g_const = bool(np.allclose(gr, gr[0], atol=1e-3))
    print(f"\n  G_21 survives the repair as stated (u a unit REAL quaternion): {ok(g_survives)}")
    print(f"  G_21/dagger ratio across random q: {np.round(gr, 4)}")
    print(f"  constant (i.e. proportional to the dagger form)?         {g_const}")
    print("  Reason: a REAL u acts on Re q and Im q separately, so any form weighting")
    print("  those two blocks separately is left-SU(2)-invariant. The spinor action does")
    print("  not mix the real and imaginary parts of the complex coefficients.")

    # the h-phase kills it
    h = torch.tensor([[1j, 0, 0, 0]], dtype=CDTYPE)
    hdh = quat_mul(hermitian_conj(h), h)[0]
    Nh = quat_norm_sq(h)[0]
    q_h = quat_mul(h.expand(NSAMP, 4), q)
    hd_i, hd_d = inv_under(dagger_form, q, q_h)
    hg_i, hg_d = inv_under(reim_form, q, q_h)
    hw_i, hw_d = inv_under(weighted_form, q, q_h)
    print(f"\n  The generator that kills G_ab: the h-phase u = h = (h,0,0,0).")
    print(f"    u^dag u = {hdh.numpy()}  -> is u dagger-unit?  YES"
          f"  {ok(abs(hdh[0].item() - 1) < 1e-6)}")
    print(f"    N(u) = q0^2+... = {Nh.item():+.1f}      -> is u bar-unit?     NO, N(h) = -1"
          f"  {ok(abs(Nh.item() + 1) < 1e-6)}")
    print(f"    so h is a unit quaternion for the DAGGER but not for the reduced norm,")
    print(f"    and it is not a real quaternion -- it is the global phase e^{{h pi/2}}.\n")
    print(f"  {'q -> h q':<46}{'dagger':>10}{'F_21':>10}{'G_21':>10}")
    print(f"  {'invariant?':<46}"
          f"{('yes' if hd_i else 'NO'):>10}{('yes' if hw_i else 'NO'):>10}{('yes' if hg_i else 'NO'):>10}")
    print(f"  {'max|diff|':<46}{hd_d:>10.2e}{hw_d:>10.2f}{hg_d:>10.2f}")
    r4 = g_survives and (not g_const) and hd_i and (not hg_i)
    print(f"\n  R4 negative finding reproduced (repair as stated insufficient): {ok(r4)}")
    print("  >> Neither generator alone does the job: F_ab survives the h-phase, G_ab")
    print("     survives SU(2). The group that must be named is the DAGGER-UNIT group")
    print("     {u : u^dag u = 1} = U(2) = SU(2) x global phase. The one-line proof in R2")
    print("     is unchanged (it only ever used u^dag u = 1); only the naming was wrong.")
    ud = rand_dagger_unit_quat(NSAMP)
    uud = quat_mul(hermitian_conj(ud), ud)
    ud_ok = bool((uud[..., 0].real - 1).abs().max() < TOL and uud[..., 1:].abs().max() < TOL)
    q_ud = quat_mul(ud, q)
    fd_i, fd_d = inv_under(dagger_form, q, q_ud)
    fw_i, _ = inv_under(weighted_form, q, q_ud)
    fg_i, _ = inv_under(reim_form, q, q_ud)
    print(f"\n  Sampling the full dagger-unit group u = e^{{h phi}} v, v a unit real quat:")
    print(f"    u^dag u = 1 on all {NSAMP}:                               {ok(ud_ok)}")
    print(f"    dagger form invariant under q -> u q:                  {ok(fd_i)}"
          f"   max|diff| = {fd_d:.2e}")
    print(f"    F_21 killed (no longer invariant):                    {ok(not fw_i)}")
    print(f"    G_21 killed (no longer invariant):                    {ok(not fg_i)}")
    print("    >> BOTH counterexamples die against the full dagger-unit group.")

    # =======================================================================
    banner("D1. The uniqueness claim made exact: dimension of the invariant form space")
    print("  A real quadratic form on C (x) H = R^8 is a symmetric 8x8 matrix S, a 36-")
    print("  dimensional space. Invariance under an R-linear g with real matrix R_g reads")
    print("  R_g^T S R_g = S. Stack (T_g - I) over sampled group elements and count the")
    print("  nullspace by SVD. 'Unique up to positive scale' means dim = 1.\n")

    # symmetric basis of the 36-dim space
    basis = []
    for i in range(8):
        E = np.zeros((8, 8))
        E[i, i] = 1.0
        basis.append(E)
    for i in range(8):
        for j in range(i + 1, 8):
            E = np.zeros((8, 8))
            E[i, j] = E[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(E)
    basis = np.array(basis)

    def vecS(S):
        return np.einsum("ij,kij->k", S, basis)

    def T_of(R):
        return np.array([vecS(R.T @ b @ R) for b in basis]).T

    NG = 12
    us_r = [rand_real_unit_quat(1) for _ in range(NG)]
    us_d = [rand_dagger_unit_quat(1) for _ in range(NG)]
    g_dag = lambda x: hermitian_conj(x)
    g_bar = lambda x: quat_conj(x)
    mk_left = lambda u: (lambda x: quat_mul(u.expand(x.shape[0], 4), x))
    mk_conj = lambda u: (lambda x: quat_mul(quat_mul(u.expand(x.shape[0], 4), x),
                                            hermitian_conj(u).expand(x.shape[0], 4)))

    def dim_inv(maps):
        A = np.vstack([T_of(real_mat(f)) - np.eye(36) for f in maps])
        sv = np.linalg.svd(A, compute_uv=False)
        d = int((sv < 1e-3).sum())
        gap_lo = sv[36 - d - 1] if d < 36 else 0.0
        gap_hi = sv[36 - d] if d > 0 else float("nan")
        return d, gap_lo, gap_hi, A

    CONJ = [mk_conj(u) for u in us_r]
    LEFT_R = [mk_left(u) for u in us_r]
    LEFT_D = [mk_left(u) for u in us_d]

    rows = [
        ("dagger involution alone (what Lemma 4.3 states)", [g_dag], 20),
        ("  + bar involution", [g_dag, g_bar], 14),
        ("  + conjugation q -> u q u^dag", [g_dag, g_bar] + CONJ, 4),
        ("  + left mult, unit REAL quats (repair as stated)", [g_dag, g_bar] + CONJ + LEFT_R, 2),
        ("  + left mult, dagger-unit u (u^dag u = 1)", [g_dag, g_bar] + CONJ + LEFT_D, 1),
    ]
    print(f"  {'invariance group imposed':<50}{'dim':>5}{'last 0 sv':>12}{'first nz sv':>13}")
    dims = []
    for name, maps, expect in rows:
        d, lo, hi, _ = dim_inv(maps)
        dims.append(d)
        flag = "" if d == expect else "  <-- UNEXPECTED"
        print(f"  {name:<50}{d:>5}{hi:>12.1e}{lo:>13.1e}{flag}")
    d1 = dims == [20, 14, 4, 2, 1]
    print(f"\n  D1 dimension ladder 20 -> 14 -> 4 -> 2 -> 1 reproduced:   {ok(d1)}")
    print("  >> Lemma 4.3 asserts dim = 1 while invoking only the dagger involution, under")
    print("     which the true dimension is 20. Even after adding bar and conjugation the")
    print("     answer is 4, and after the SU(2) repair it is 2. Only the dagger-unit")
    print("     group gets to 1.")
    print("\n  The dim-4 space at the conjugation stage is exactly")
    print("      a(Re q0)^2 + b(Im q0)^2 + c||Re q_vec||^2 + d||Im q_vec||^2,")
    print("  which contains F_ab (a=b, c=d) and G_ab (a=c, b=d) as the two 2-parameter")
    print("  slices, meeting at the dagger form a=b=c=d. Check a random member is invariant:")
    cf = rng.uniform(0.5, 3.0, size=4)

    def four_param(x, c=cf):
        return (c[0] * x[..., 0].real ** 2 + c[1] * x[..., 0].imag ** 2
                + c[2] * (x[..., 1:].real ** 2).sum(-1) + c[3] * (x[..., 1:].imag ** 2).sum(-1))

    fp_i, fp_d = inv_under(four_param, q, q_conj)
    print(f"    coeffs (a,b,c,d) = {np.round(cf, 3)}")
    print(f"    invariant under q -> u q u^dag:                        {ok(fp_i)}"
          f"   max|diff| = {fp_d:.2e}")

    # =======================================================================
    banner("D2. The MINIMAL repair, and what the surviving form actually is")
    d_left_only, lo0, hi0, _ = dim_inv(LEFT_D)
    d_min, lo1, hi1, A_min = dim_inv([g_dag] + LEFT_D)
    print(f"  left mult by dagger-unit u ALONE, no involutions:  dim = {d_left_only}"
          f"   (not 1: the four\n    columnwise-weighted forms survive)")
    print(f"  dagger involution + left mult by dagger-unit u:    dim = {d_min}"
          f"   <- uniqueness,\n    with NO bar and NO conjugation needed")
    print("\n  Why the involution earns its keep: (u q)^dag = q^dag u^dag, so conjugating")
    print("  left multiplication by the dagger involution gives RIGHT multiplication. The")
    print("  Z_2 the lemma already states, plus left-U(2), generates the TWO-SIDED unitary")
    print("  action, and bi-unitary invariance forces the Frobenius (= dagger) form.")

    U_, S_, Vt_ = np.linalg.svd(A_min)
    v = Vt_[-1]
    Smat = np.einsum("k,kij->ij", v, basis)
    Smat = Smat / Smat[0, 0]
    is_id = np.allclose(Smat, np.eye(8), atol=1e-3)
    print(f"\n  Recovering the surviving form (nullspace vector, normalised S[0,0] = 1):")
    print("   ", np.array2string(np.round(Smat, 4), prefix="    "))
    print(f"    S == I_8:                                             {ok(is_id)}"
          f"   max|S - I| = {np.abs(Smat - np.eye(8)).max():.2e}")
    print("  >> S = I_8 means the form is the plain sum over all 8 real coordinates,")
    print("     i.e. sum_n (Re q_n)^2 + (Im q_n)^2 = sum_n |q_n|^2 = <q^dag q>_0.")
    # cross-check against cal's own dagger form
    xr = to_real(qs)
    quad = np.einsum("ni,ij,nj->n", xr, Smat, xr)
    match = np.allclose(quad, dagger_form(qs).numpy(), atol=1e-3, rtol=1e-3)
    print(f"    recovered form evaluated on random q == cal's <q^dag q>_0:  {ok(match)}")
    print(f"      recovered {np.round(quad, 4)}")
    print(f"      cal       {np.round(dagger_form(qs).numpy(), 4)}")
    d2 = (d_left_only == 4) and (d_min == 1) and is_id and match

    # =======================================================================
    banner("SUMMARY")
    print(f"  NEGATIVE (U1,U2,W1): Lemma 4.3's uniqueness claim is FALSE as stated: {ok(u2)}")
    print(f"    F_ab(q) = {A_W}|q0|^2 + {B_W}|q_vec|^2 is real, non-negative, positive definite,")
    print("    bar-invariant, dagger-invariant, and invariant under q -> u q u^dag, and is")
    print("    NOT proportional to the dagger form. The lemma's stated invariance is a Z_2")
    print("    (the dagger involution), under which the invariant space has dimension 20.")
    print(f"  REPAIR (R1,R2,R3): left multiplication q -> u q kills F_ab:            {ok(wit_dag and wit_w and di and not wi and only_ab)}")
    print("    <(uq)^dag (uq)>_0 = <q^dag u^dag u q>_0 = <q^dag q>_0 since u^dag u = 1.")
    print(f"  NEGATIVE ON THE REPAIR (R4): SU(2) alone is NOT enough:               {ok(r4)}")
    print(f"    G_ab(q) = {A_W}||Re q||^2 + {B_W}||Im q||^2 survives left mult by every unit REAL")
    print("    quaternion. The group must be the dagger-unit group {u : u^dag u = 1} = U(2)")
    print("    = SU(2) x global phase; the extra generator is the h-phase, h^dag h = +1")
    print("    while N(h) = -1.")
    print(f"  DIMENSION LADDER (D1,D2): 20 -> 14 -> 4 -> 2 -> 1:                     {ok(d1 and d2)}")
    print("    Minimal sufficient hypothesis: the dagger involution (already in the lemma)")
    print("    PLUS left multiplication by the dagger-unit group. dim = 1, and the")
    print("    surviving form is exactly <q^dag q>_0.")
    print("\n  THE COST, honestly: invariance under the dagger-unit group is UNITARITY -- the")
    print("  spinor action plus global phase invariance. That is a physical premise, so the")
    print("  sentence 'This is a statement about the algebra alone' must go: the algebra")
    print("  alone gives 20, not 1. But it is one line, far cheaper than Gleason, and unlike")
    print("  the Prop 5.2 route it is actually true. Name the group and the lemma is fixed.")

    allgood = u2 and r4 and d1 and d2 and only_ab and wit_dag and wit_w
    print(f"\n  ALL CLAIMED FINDINGS REPRODUCED: {ok(allgood)}")


if __name__ == "__main__":
    main()
