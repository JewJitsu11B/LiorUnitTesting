"""
What the dagger closure W^dag W actually outputs: a FORWARD-CONE PARAVECTOR.

Sharpens born_axiomatic.tex Cor 6.2 and Cor 6.4. The claim under test is structural, not
numerical: W^dag W is never an arbitrary biquaternion. It is always Hermitian and positive
semidefinite, hence a PARAVECTOR (grade 0 + grade 1 only, with identically zero bivector and
pseudoscalar content), and it always lies in the CLOSED FORWARD light cone of the (1,3)
Hermitian slice of Cl(3,0), with

    t^2 - |x_vec|^2 = det(W^dag W) = |det W|^2 = |N(W)|^2 >= 0,

null exactly when W is a zero divisor. The Born weight is its TIME component t = <W^dag W>_0.

CLAIMS VERIFIED (POSITIVE), against cal.biquaternion, the package under test:
  P1  W^dag W is Hermitian, three ways that agree: (i) M(W)^H M(W) equals its own conjugate
      transpose; (ii) cal's hermitian_conj FIXES it as a biquaternion; (iii) its biquaternion
      coordinates have q0 real and q1,q2,q3 purely imaginary. Control: a generic W is NOT
      fixed by hermitian_conj, so (ii) is a real constraint and not a tautology.
  P2  Grade content of W^dag W, from the Cl(3,0) blade basis {1; e1,e2,e3; e12,e23,e31; e123}
      with coeff = 0.5 * Re tr(M B^dag): grade-2 and grade-3 are IDENTICALLY ZERO, while
      grade-0 and grade-1 are generically nonzero. So W^dag W is a paravector t + x_vec, and
      (t,x,y,z) are literally its grade-0 and grade-1 blade coefficients.
  P3  Positive semidefinite: eigenvalues of W^dag W are all >= 0, and t >= |x_vec|, i.e. the
      output sits in the CLOSED FORWARD cone (t > 0 for W nonzero).
  P4  Interval identity to precision on generic / random / null-cone / central W:
          t^2 - |x_vec|^2 == det(W^dag W) == |det W|^2 == |N(W)|^2 = |quat_norm_sq(W)|^2,
      and the Born weight is the time component t == <W^dag W>_0 == sum_n |W_n|^2.
  P5  Matrix rank: rank(W^dag W) == rank(W) == 2 generically, and == 1 on the null cone,
      using the paper's own zero divisor q = 1 + h*i from Lemma 4.1(c). The interval is
      exactly 0 there, i.e. the paravector is NULL exactly when W is a zero divisor.
  P6  For CENTRAL W = w*1, W^dag W = |w|^2 * IDENTITY: grade-0 only, grade-1 norm 0. Its
      "rank 2" is just the rank of the identity matrix and carries no structural information.
      Rank 2 is not evidence of biquaternionic structure for the W the theorem actually uses.

NEGATIVE FINDING (reported as a negative, kept as a negative):
  N1  Grade-0 IS the timelike direction of that slice. So the Born weight is a TIME COMPONENT,
      not a Lorentz scalar. Under the SL(2,C) action X -> L X L^dag with det L = 1 (e.g.
      L = diag(e^{r/2}, e^{-r/2})), which is exactly the Lorentz action on the (1,3) slice:
          t CHANGES     (t' = t cosh r + z sinh r)
          t^2 - |x|^2   is INVARIANT (= det, and det L = 1)
      The invariant of the slice is the BAR closure |N(W)|^2, not the dagger closure t.
  N2  The boost is not an outside intrusion: L X L^dag with X = W^dag W equals W'^dag W' for
      W' = W L^dag, an ordinary right multiplication INSIDE the algebra. So the frame
      dependence of the Born weight is reachable by the algebra's own operations.

  This is the ordinary Dirac-current situation: rho = psi^dag psi is likewise a time component
  of a conserved 4-current j^mu = psi^dag gamma^0 gamma^mu psi, not a scalar. There it is
  repaired by current conservation d_mu j^mu = 0 plus an integral over a spacelike slice, which
  makes the TOTAL probability frame independent even though the DENSITY is not. That repair is
  available here too. But that apparatus -- a conserved current, a choice of spacelike
  hypersurface, and an integral over it -- is exactly the "external partition integral" that
  Thm 6.1 claims not to need. The positive result P1-P6 and the negative N1-N2 are the same
  fact seen twice: the closure lands in the forward cone, which is a strong geometric statement,
  and grade-0 is a coordinate on that cone, which is why it is not by itself a probability.

Not covered here (see exp_two_closures.py, C1-C7): multiplicativity of the reduced norm, the
bar-vs-dagger interference contrast, the null cone as a bar/dagger complementarity. Not covered
here (see exp_clifford_conjugation_identification.py, A1-A7): the identification of the paper's
bar and dagger with Clifford conjugation and reversion. This script adds only the paravector /
forward-cone structure of the OUTPUT and the frame dependence of its time component.
"""
import os
import sys

import numpy as np
import torch

# cal/ lives at the repo root, one level above "py tests"; this script is run from inside
# "py tests" as: python exp_born_paravector_forward_cone.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cal.biquaternion import (quat_mul, quat_conj, hermitian_conj, hermitian, anti_hermitian,
                              quat_norm_sq, biquat_to_matrix, CDTYPE)

torch.manual_seed(0)
RNG = np.random.default_rng(11)
np.set_printoptions(precision=6, suppress=True)

TOL = 1e-4         # cal is complex64 (float32 precision)
TOL_EXACT = 1e-10  # pure-numpy float64 steps

# ---------------------------------------------------------------------------
# Cl(3,0) blade basis in the Pauli representation e1=s1, e2=s2, e3=s3. The 8 blades with
# REAL coefficients span M2(C) as a real 8-dimensional space and are orthonormal under
# <A,B> = (1/2) Re tr(A B^dag), which is what makes the grade projection well defined.
# The Hermitian slice is spanned by {1, e1, e2, e3} = grades 0 and 1: the paravectors.
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


def blade_coeff(M, B):
    """Real coefficient of orthonormal blade B in M. M shape (...,2,2) -> (...,)."""
    Bh = B.conj().T
    return 0.5 * np.real(np.einsum("...ij,ji->...", M, Bh))


def grade_norms(M):
    """Per-grade coefficient norm. M shape (...,2,2) -> dict grade -> (...,) array."""
    return {k: np.linalg.norm(np.stack([blade_coeff(M, B) for _, B in bl], axis=-1), axis=-1)
            for k, bl in BLADES.items()}


def paravector(M):
    """Grade-0 and grade-1 blade coefficients (t,x,y,z) of the (1,3) Hermitian slice."""
    return np.stack([blade_coeff(M, s0), blade_coeff(M, s1),
                     blade_coeff(M, s2), blade_coeff(M, s3)], axis=-1)


def interval(p):
    """t^2 - |x_vec|^2 of a paravector row (t,x,y,z)."""
    return p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)


def to_np(t):
    return t.detach().numpy().astype(np.complex128)


def mat(q):
    """cal biquaternion (...,4) -> numpy (...,2,2) complex128."""
    return to_np(biquat_to_matrix(q))


def dagger_closure(q):
    """W^dag W as a cal biquaternion."""
    return quat_mul(hermitian_conj(q), q)


def rank(M, rtol=1e-5):
    """Matrix rank via singular values, relative to the largest."""
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > rtol * s[..., 0]).sum())


def rand_biquat(n):
    return (torch.randn(n, 4) + 1j * torch.randn(n, 4)).to(CDTYPE)


def bq(q0, q1, q2, q3):
    return torch.tensor([[q0, q1, q2, q3]], dtype=CDTYPE)


def ok(b):
    return "PASS" if b else "**FAIL**"


def banner(t):
    print()
    print("=" * 96)
    print(t)
    print("=" * 96)


# ---------------------------------------------------------------------------
# The four named W. Coordinates are cal biquaternion coefficients (q0,q1,q2,q3).
#   generic W = G + S with G = hermitian(W) = 0.9 + 0.4h*i + 0.2h*j  (dagger-even, thermal)
#                     and S = anti_hermitian(W) = 0.6*i + 0.3*k      (dagger-odd, spectral)
#   null cone W = 1 + h*i is the paper's own zero divisor, Lemma 4.1(c): N = 1 + h^2 = 0.
#   central W = w*1 is the W the theorem's literal Axiom 2 actually produces.
# ---------------------------------------------------------------------------
W_GENERIC = bq(0.9, 0.6 + 0.4j, 0.2j, 0.3)
W_NULL = bq(1.0, 1j, 0.0, 0.0)
W_CENTRAL = bq(0.7 + 1.2j, 0.0, 0.0, 0.0)


def main():
    W_random = rand_biquat(1)
    batch = rand_biquat(200)

    cases = [("generic W = G + S", W_GENERIC),
             ("random W", W_random),
             ("null cone 1 + h*i", W_NULL),
             ("central w*1", W_CENTRAL)]

    # -----------------------------------------------------------------------
    banner("P1. W^dag W is HERMITIAN -- three descriptions that agree")

    print("  Setup check on the generic W: the channel split the paper uses.")
    G = hermitian(W_GENERIC)
    S = anti_hermitian(W_GENERIC)
    g_herm = torch.allclose(hermitian_conj(G), G, atol=TOL)
    s_anti = torch.allclose(hermitian_conj(S), -S, atol=TOL)
    print(f"    G = hermitian(W)      = {to_np(G)[0]}   dagger-even:  {ok(g_herm)}")
    print(f"    S = anti_hermitian(W) = {to_np(S)[0]}   dagger-odd:   {ok(s_anti)}")
    print(f"    G + S == W:                                                  "
          f"{ok(torch.allclose(G + S, W_GENERIC, atol=TOL))}")

    print(f"\n  {'case':<22}{'(i) M^H M self-adjoint':>24}{'(ii) dagger fixes it':>22}"
          f"{'(iii) q0 real, vec imag':>26}")
    p1_all = []
    for name, W in cases:
        M = mat(W)[0]
        X = M.conj().T @ M
        herm_mat = np.allclose(X, X.conj().T, atol=TOL_EXACT)

        P = dagger_closure(W)
        herm_cal = torch.allclose(hermitian_conj(P), P, atol=TOL)

        c = to_np(P)[0]
        coord = (abs(c[0].imag) < TOL and max(abs(c[1].real), abs(c[2].real),
                                              abs(c[3].real)) < TOL)
        p1_all += [herm_mat, herm_cal, coord]
        print(f"  {name:<22}{ok(herm_mat):>24}{ok(herm_cal):>22}{ok(coord):>26}")

    Pb = dagger_closure(batch)
    Mb = mat(batch)
    Xb = np.einsum("...ji,...jk->...ik", Mb.conj(), Mb)
    batch_herm = np.allclose(Xb, np.conjugate(np.swapaxes(Xb, -1, -2)), atol=TOL_EXACT)
    batch_cal = torch.allclose(hermitian_conj(Pb), Pb, atol=TOL)
    print(f"\n  batch of 200 random W: (i) {ok(batch_herm)}   (ii) {ok(batch_cal)}"
          f"   max|W^dag W - (W^dag W)^dag| = "
          f"{(hermitian_conj(Pb) - Pb).abs().max().item():.2e}")

    ctrl = not torch.allclose(hermitian_conj(batch), batch, atol=1e-2)
    print(f"  control -- generic W is NOT itself dagger-fixed (must hold, or (ii) is"
          f"\n    vacuous):                                                  {ok(ctrl)}")
    print("  >> M(W)^H M(W) is a Gram matrix, so Hermiticity is automatic and needs no")
    print("     hypothesis on W. The dagger closure CANNOT leave the Hermitian slice.")

    # -----------------------------------------------------------------------
    banner("P2. Grade content: grade-2 and grade-3 are IDENTICALLY ZERO -- a PARAVECTOR")

    print(f"  {'case':<22}{'grade 0':>12}{'grade 1':>12}{'grade 2':>12}{'grade 3':>12}"
          f"{'paravector?':>14}")
    p2_all = []
    for name, W in cases:
        X = mat(dagger_closure(W))[0]
        gn = grade_norms(X)
        is_para = gn[2] < TOL and gn[3] < TOL
        p2_all.append(is_para)
        print(f"  {name:<22}{gn[0]:>12.6f}{gn[1]:>12.6f}{gn[2]:>12.2e}{gn[3]:>12.2e}"
              f"{ok(is_para):>14}")

    Xb_out = mat(Pb)
    gnb = grade_norms(Xb_out)
    batch_para = bool(gnb[2].max() < TOL and gnb[3].max() < TOL)
    g0_live = bool(gnb[0].min() > 1e-3)
    g1_live = bool(gnb[1].min() > 1e-3)
    print(f"\n  batch of 200 random W:")
    print(f"    max grade-2 norm = {gnb[2].max():.3e}   max grade-3 norm = {gnb[3].max():.3e}"
          f"   -> paravector: {ok(batch_para)}")
    print(f"    min grade-0 norm = {gnb[0].min():.6f}   min grade-1 norm = {gnb[1].min():.6f}"
          f"   -> both generically nonzero: {ok(g0_live and g1_live)}")

    # Control: a generic W (not a closure output) DOES carry grade 2 and grade 3. Stated
    # distributionally, not per-draw: grade 3 is Im(q0) alone, so an individual Gaussian draw
    # can land near zero by chance (the min below). What matters is that the input grades are
    # O(1) on average while the OUTPUT grades are at float32 round-off for every single draw.
    gnW = grade_norms(Mb)
    ctrl2 = bool(gnW[2].mean() > 0.1 and gnW[3].mean() > 0.1
                 and gnW[2].mean() > 1e3 * max(gnb[2].max(), 1e-12))
    print(f"    control -- the INPUT W carries grade 2 (mean norm {gnW[2].mean():.4f},"
          f" min {gnW[2].min():.4f}, max {gnW[2].max():.4f})")
    print(f"      and grade 3 (mean norm {gnW[3].mean():.4f}, min {gnW[3].min():.4f},"
          f" max {gnW[3].max():.4f}), so the")
    print(f"      vanishing above is the closure's doing, not an empty basis: {ok(ctrl2)}")
    print(f"      input mean grade-2 norm {gnW[2].mean():.4f} vs output MAX grade-2 norm"
          f" {gnb[2].max():.2e}: a factor of {gnW[2].mean() / max(gnb[2].max(), 1e-12):.1e}")
    print("  >> The bivector and pseudoscalar content of W is annihilated by the closure.")
    print("     Grades 2 and 3 are the anti-Hermitian directions of M2(C); the closure output")
    print("     is Hermitian, so it has none. (t,x,y,z) ARE the grade-0/grade-1 coefficients.")

    # -----------------------------------------------------------------------
    banner("P3. Positive semidefinite: the output is in the CLOSED FORWARD cone")

    print(f"  {'case':<22}{'eig_min':>12}{'eig_max':>12}{'t':>12}{'|x_vec|':>12}"
          f"{'t - |x_vec|':>14}{'PSD & forward':>16}")
    p3_all = []
    for name, W in cases:
        X = mat(dagger_closure(W))[0]
        ev = np.linalg.eigvalsh(X)
        p = paravector(X)
        t, xn = p[0], float(np.linalg.norm(p[1:]))
        good = bool(ev.min() > -TOL and t > 0 and (t - xn) > -TOL)
        p3_all.append(good)
        print(f"  {name:<22}{ev.min():>12.6f}{ev.max():>12.6f}{t:>12.6f}{xn:>12.6f}"
              f"{t - xn:>14.6f}{ok(good):>16}")

    evb = np.linalg.eigvalsh(Xb_out)
    pb = paravector(Xb_out)
    tb = pb[..., 0]
    xnb = np.linalg.norm(pb[..., 1:], axis=-1)
    batch_psd = bool(evb.min() > -TOL)
    batch_fwd = bool(tb.min() > 0 and (tb - xnb).min() > -TOL)
    print(f"\n  batch of 200 random W: min eigenvalue over all = {evb.min():.3e}"
          f"   -> PSD: {ok(batch_psd)}")
    print(f"    min t = {tb.min():.6f}   min (t - |x_vec|) = {(tb - xnb).min():.3e}"
          f"   -> closed forward cone: {ok(batch_fwd)}")

    # -----------------------------------------------------------------------
    banner("P4. Interval identity: t^2 - |x|^2 = det(W^dag W) = |det W|^2 = |N(W)|^2")

    print(f"  {'case':<22}{'t':>11}{'|x_vec|':>11}{'t^2-|x|^2':>13}{'det(WdagW)':>13}"
          f"{'|det W|^2':>13}{'|N(W)|^2':>13}{'sum|W_n|^2':>13}")
    p4_all = []
    for name, W in cases:
        M = mat(W)[0]
        X = mat(dagger_closure(W))[0]
        p = paravector(X)
        iv = float(interval(p))
        det_out = float(np.real(np.linalg.det(X)))
        det_in2 = float(abs(np.linalg.det(M)) ** 2)
        Nw2 = float(abs(quat_norm_sq(W).item()) ** 2)
        born = float((W.real ** 2 + W.imag ** 2).sum(-1).item())
        scale = max(1.0, abs(iv), det_in2)
        agree = (abs(iv - det_out) < TOL * scale and abs(iv - det_in2) < TOL * scale
                 and abs(det_in2 - Nw2) < TOL * scale
                 and abs(float(p[0]) - born) < TOL * max(1.0, born))
        p4_all.append(agree)
        print(f"  {name:<22}{p[0]:>11.5f}{np.linalg.norm(p[1:]):>11.5f}{iv:>13.6f}"
              f"{det_out:>13.6f}{det_in2:>13.6f}{Nw2:>13.6f}{born:>13.6f}")
    print(f"\n  all four rows agree across all five columns to rel. tol {TOL:.0e}:"
          f"    {ok(all(p4_all))}")

    ivb = interval(pb)
    det_out_b = np.real(np.linalg.det(Xb_out))
    det_in2_b = np.abs(np.linalg.det(Mb)) ** 2
    Nw2_b = np.abs(to_np(quat_norm_sq(batch))) ** 2
    born_b = (batch.real ** 2 + batch.imag ** 2).sum(-1).numpy().astype(np.float64)
    scale_b = np.maximum(1.0, np.abs(ivb))
    e1 = float((np.abs(ivb - det_out_b) / scale_b).max())
    e2 = float((np.abs(ivb - det_in2_b) / scale_b).max())
    e3 = float((np.abs(det_in2_b - Nw2_b) / np.maximum(1.0, det_in2_b)).max())
    e4 = float((np.abs(tb - born_b) / np.maximum(1.0, born_b)).max())
    batch_iv = bool(max(e1, e2, e3, e4) < 1e-3)
    print(f"\n  batch of 200 random W, max RELATIVE error:")
    print(f"    |(t^2-|x|^2) - det(W^dag W)| : {e1:.3e}")
    print(f"    |(t^2-|x|^2) - |det W|^2|    : {e2:.3e}")
    print(f"    ||det W|^2 - |N(W)|^2|       : {e3:.3e}")
    print(f"    |t - sum_n |W_n|^2|          : {e4:.3e}")
    print(f"    all below 1e-3 (float32 package; t^2 - |x|^2 is a cancelling difference,")
    print(f"    so it loses digits by construction):                        {ok(batch_iv)}")
    nonneg = bool(ivb.min() > -TOL)
    print(f"    min interval over the batch = {ivb.min():.3e}   -> always >= 0: {ok(nonneg)}")
    print("  >> The interval of the output is the SQUARED MODULUS of the reduced norm of the")
    print("     input. It is >= 0 for free, which is why the cone is never left.")

    # -----------------------------------------------------------------------
    banner("P5. Rank: 2 generically, 1 on the null cone (the paper's own zero divisor)")

    print(f"  {'case':<22}{'rank(W)':>10}{'rank(WdagW)':>14}{'det W':>26}"
          f"{'interval':>12}{'null?':>8}")
    p5_all = []
    for name, W in cases:
        M = mat(W)[0]
        X = mat(dagger_closure(W))[0]
        rW, rX = rank(M), rank(X)
        detW = np.linalg.det(M)
        iv = float(interval(paravector(X)))
        is_null = abs(iv) < TOL
        expect = 1 if abs(detW) < 1e-6 else 2
        good = (rW == rX == expect)
        p5_all.append(good)
        print(f"  {name:<22}{rW:>10}{rX:>14}{str(np.round(detW, 6)):>26}{iv:>12.6f}"
              f"{str(is_null):>8}")

    print(f"\n  rank(W^dag W) == rank(W) in every row, and == 1 exactly where det W == 0:"
          f"  {ok(all(p5_all))}")

    print(f"\n  Zero divisor q = 1 + h*i (Lemma 4.1(c)) in detail:")
    Xn = mat(dagger_closure(W_NULL))[0]
    pn = paravector(Xn)
    print(f"    N(q) = quat_norm_sq(q) = {quat_norm_sq(W_NULL).item():+.6f}"
          f"   (1 + h^2 = 0, so q is a zero divisor)")
    print(f"    W^dag W paravector (t,x,y,z) = {pn}")
    print(f"    eigenvalues of W^dag W       = {np.linalg.eigvalsh(Xn)}")
    print(f"    t = {pn[0]:.6f}   |x_vec| = {np.linalg.norm(pn[1:]):.6f}"
          f"   t^2-|x|^2 = {float(interval(pn)):.3e}")
    on_cone = abs(float(interval(pn))) < TOL and rank(Xn) == 1 and pn[0] > 0.5
    print(f"    NULL (on the cone) but NOT zero: t > 0 while the interval vanishes: "
          f"{ok(on_cone)}")
    print("  >> The Born weight stays positive exactly where the reduced norm dies. That is")
    print("     the C5 complementarity of exp_two_closures.py, read geometrically: the")
    print("     output slides onto the boundary of the cone but does not fall out of it.")

    ranks_b = np.array([rank(Xb_out[i]) for i in range(Xb_out.shape[0])])
    ranks_in = np.array([rank(Mb[i]) for i in range(Mb.shape[0])])
    batch_rank = bool((ranks_b == 2).all() and (ranks_in == 2).all())
    print(f"\n  batch of 200 random W: rank(W) == 2 for all: {ok(bool((ranks_in == 2).all()))}"
          f"   rank(W^dag W) == 2 for all: {ok(bool((ranks_b == 2).all()))}")

    # -----------------------------------------------------------------------
    banner("P6. CENTRAL W = w*1: the output is |w|^2 * IDENTITY -- rank 2 says nothing")

    w = to_np(W_CENTRAL)[0, 0]
    Xc = mat(dagger_closure(W_CENTRAL))[0]
    gc = grade_norms(Xc)
    target = (abs(w) ** 2) * s0
    is_id = np.allclose(Xc, target, atol=TOL)
    pure_g0 = bool(gc[1] < TOL and gc[2] < TOL and gc[3] < TOL)
    print(f"  w = {w:+.4f}   |w|^2 = {abs(w) ** 2:.6f}")
    print(f"  W^dag W =\n{np.round(Xc, 6)}")
    print(f"  W^dag W == |w|^2 * IDENTITY:                                 {ok(is_id)}"
          f"   max|diff| = {np.abs(Xc - target).max():.2e}")
    print(f"  grade norms: g0 = {gc[0]:.6f}  g1 = {gc[1]:.2e}  g2 = {gc[2]:.2e}"
          f"  g3 = {gc[3]:.2e}")
    print(f"  purely grade-0 (zero grade-1, so a point on the cone AXIS):  {ok(pure_g0)}")
    print(f"  rank(W^dag W) = {rank(Xc)}   rank(identity) = {rank(s0)}   same: "
          f"{ok(rank(Xc) == rank(s0))}")
    central = all(np.allclose(mat(W_CENTRAL)[0] @ X - X @ mat(W_CENTRAL)[0], 0, atol=TOL)
                  for X in (s1, s2, s3, Mb[0]))
    print(f"  W = w*1 is central (commutes with everything):               {ok(central)}")
    print("  >> With scalar H and S in Axiom 2 the source current W has NO quaternionic part,")
    print("     so W^dag W is |w|^2 times the identity. It is rank 2 because the identity is")
    print("     rank 2, not because any algebraic structure survived. The rank is not")
    print("     evidence. Contrast the generic row of P2: grade-1 norm "
          f"{grade_norms(mat(dagger_closure(W_GENERIC))[0])[1]:.4f} there, 0 here.")

    # -----------------------------------------------------------------------
    banner("N1. NEGATIVE: grade-0 is the TIME axis, so the Born weight is NOT a scalar")

    print("  The (1,3) Hermitian slice carries the Lorentz action X -> L X L^dag with")
    print("  L in SL(2,C). det L = 1 forces det X, i.e. t^2 - |x|^2, to be invariant.")
    print("  It does NOT fix t. Boost along e3: L(r) = diag(e^{r/2}, e^{-r/2}).\n")

    def boost(r):
        return np.array([[np.exp(r / 2), 0], [0, np.exp(-r / 2)]], dtype=np.complex128)

    rs = np.concatenate([[0.0], np.sort(RNG.uniform(0.2, 1.5, size=3))])
    n1_all = []
    for name, W in [("generic W = G + S", W_GENERIC), ("random W", W_random)]:
        X0 = mat(dagger_closure(W))[0]
        p0 = paravector(X0)
        iv0 = float(interval(p0))
        print(f"  --- {name}:  rest-frame (t,x,y,z) = {np.round(p0, 5)}")
        print(f"      {'r':>8}{'det L':>10}{'t':>12}{'|x_vec|':>12}{'t^2-|x|^2':>14}"
              f"{'t cosh r + z sinh r':>22}{'t/t_0':>10}")
        for r in rs:
            L = boost(r)
            Xr = L @ X0 @ L.conj().T
            pr = paravector(Xr)
            ivr = float(interval(pr))
            closed = float(p0[0] * np.cosh(r) + p0[3] * np.sinh(r))
            detL = float(np.real(np.linalg.det(L)))
            inv_ok = abs(ivr - iv0) < 1e-3 * max(1.0, abs(iv0))
            form_ok = abs(float(pr[0]) - closed) < 1e-8 * max(1.0, abs(closed))
            n1_all += [inv_ok, form_ok, abs(detL - 1.0) < TOL_EXACT]
            print(f"      {r:>8.4f}{detL:>10.4f}{pr[0]:>12.6f}"
                  f"{np.linalg.norm(pr[1:]):>12.6f}{ivr:>14.6f}{closed:>22.6f}"
                  f"{pr[0] / p0[0]:>10.4f}")
        t_moves = abs(float(paravector(boost(rs[-1]) @ X0 @ boost(rs[-1]).conj().T)[0])
                      - float(p0[0])) > 0.05
        n1_all.append(t_moves)
        print(f"      t CHANGES under the boost:                               {ok(t_moves)}")
        print(f"      t^2 - |x|^2 is INVARIANT across all r:                   "
              f"{ok(all(abs(float(interval(paravector(boost(r) @ X0 @ boost(r).conj().T))) - iv0)
                        < 1e-3 * max(1.0, abs(iv0)) for r in rs))}")
        # PSD / forward-cone membership survives the boost: the cone is Lorentz invariant.
        fwd = all(np.linalg.eigvalsh(boost(r) @ X0 @ boost(r).conj().T).min() > -TOL
                  and paravector(boost(r) @ X0 @ boost(r).conj().T)[0] > 0 for r in rs)
        n1_all.append(fwd)
        print(f"      still PSD and still FORWARD at every r (the cone IS invariant): "
              f"{ok(fwd)}\n")

    print("  >> The interval is invariant; the time component is not. t = <W^dag W>_0 is the")
    print("     Born weight. So the Born weight is frame dependent, and the Lorentz-invariant")
    print("     content of the closure is |N(W)|^2 -- which is the BAR closure, the one the")
    print("     paper rejects for Born because it is complex and vanishes on the null cone.")
    print("     The two closures split the job: the bar is invariant but not positive, the")
    print("     dagger is positive but not invariant. Neither is both.")

    # -----------------------------------------------------------------------
    banner("N2. NEGATIVE: the boost is an INSIDE job -- L X L^dag == W'^dag W', W' = W L^dag")

    print("  L X L^dag with X = W^dag W equals (W L^dag)^dag (W L^dag). So the frame change")
    print("  is an ordinary right multiplication of the source current by an algebra element.")
    print("  Nothing external is needed to move the Born weight.\n")

    n2_all = []
    print(f"  {'r':>8}{'t of W^dag W boosted':>24}{'t of (W L^dag)^dag (W L^dag)':>32}"
          f"{'match?':>10}")
    W = W_GENERIC
    M = mat(W)[0]
    X0 = mat(dagger_closure(W))[0]
    for r in rs:
        L = boost(r)
        lhs = L @ X0 @ L.conj().T
        Wp = M @ L.conj().T          # W' = W L^dag, still a biquaternion (M2(C) is the algebra)
        rhs = Wp.conj().T @ Wp
        same = np.allclose(lhs, rhs, atol=TOL)
        n2_all.append(same)
        print(f"  {r:>8.4f}{paravector(lhs)[0]:>24.6f}{paravector(rhs)[0]:>32.6f}"
              f"{ok(same):>10}")
    print(f"\n  W' = W L^dag is inside the algebra, and N(W') = N(W) det(L^dag) = N(W):")
    Lb = boost(rs[-1])
    Wp = M @ Lb.conj().T
    print(f"    det W  = {np.round(np.linalg.det(M), 6)}"
          f"    det W' = {np.round(np.linalg.det(Wp), 6)}"
          f"    equal: {ok(np.isclose(np.linalg.det(M), np.linalg.det(Wp), atol=TOL))}")
    print(f"    sum_n |W_n|^2  = {float((W.real ** 2 + W.imag ** 2).sum(-1).item()):.6f}")
    print(f"    sum_n |W'_n|^2 = {float(np.real(np.trace(Wp.conj().T @ Wp))):.6f}"
          f"   -> the Born weight moved, the reduced norm did not.")

    # -----------------------------------------------------------------------
    banner("SUMMARY")

    pos = (all(p1_all) and batch_herm and batch_cal and ctrl and all(p2_all) and batch_para
           and g0_live and g1_live and ctrl2 and all(p3_all) and batch_psd and batch_fwd
           and all(p4_all) and batch_iv and nonneg and all(p5_all) and on_cone and batch_rank
           and is_id and pure_g0 and central)
    neg = all(n1_all) and all(n2_all)

    print(f"  POSITIVE (P1-P6): {ok(pos)}")
    print("    W^dag W is always Hermitian PSD, hence a PARAVECTOR: grades 2 and 3 vanish")
    print("    identically, grades 0 and 1 survive. It lies in the CLOSED FORWARD cone of the")
    print("    (1,3) slice with t^2 - |x_vec|^2 = det(W^dag W) = |det W|^2 = |N(W)|^2 >= 0,")
    print("    NULL exactly on the zero divisors (rank drops 2 -> 1). The Born weight is the")
    print("    time component t = <W^dag W>_0 = sum_n |W_n|^2. This is a real sharpening of")
    print("    Cor 6.2 / Cor 6.4: positivity is not an assumption, it is cone membership.")
    print("    For the theorem's own central W = w*1 the output is |w|^2 * IDENTITY -- the")
    print("    cone axis -- and its rank 2 is the identity's rank, carrying no information.")
    print(f"\n  NEGATIVE (N1-N2): {ok(neg)}")
    print("    Grade-0 IS the timelike direction. Under X -> L X L^dag with det L = 1 the")
    print("    interval t^2 - |x|^2 is invariant and t is not, so the Born weight is a TIME")
    print("    COMPONENT, not a Lorentz scalar. The boost is reachable inside the algebra as")
    print("    W -> W L^dag. This is the ordinary Dirac-current situation (rho = psi^dag psi")
    print("    is also a time component) and is repairable the ordinary way: current")
    print("    conservation plus an integral over a spacelike slice. That apparatus is exactly")
    print("    the external partition integral Thm 6.1 claims not to need.")


if __name__ == "__main__":
    main()
