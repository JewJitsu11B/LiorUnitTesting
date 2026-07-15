"""
NEGATIVE RESULT. The proposed quaternionic POLAR form of the source current,

    J = rho * exp(n_hat * theta) = exp[ -H/tau + n_hat * S/hbar_eff ],

with n_hat a fixed unit imaginary quaternion, is offered as a replacement for the
additive form of Eq. (J-source),

    W = A exp(-H/tau) + B exp(i S/hbar_eff),

on the grounds that the additive notation "obscures" the structure. It fails three
ways. All three are reported here as negatives and kept as negatives.

CLAIMS VERIFIED:
  P1  polar and additive are DIFFERENT OBJECTS, not two notations for one thing.
      In the polar form e^u MULTIPLIES the phase; in the additive form the two
      channels sit side by side. Tabulate Re/Im of each; they disagree.

  P2  (DECISIVE) the polar form has NO INTERFERENCE.
      P2a: sweep v = S/hbar_eff over [0, 2pi]. The polar Born weight column is
           CONSTANT; the additive column varies. Both columns are computed with
           the cal Born closure <q qdag>_0, not by hand.
      P2b: polar Born weight = e^{2u} to tolerance, independent of v, because
           |e^{n v}| = |cos v + n_hat sin v| = 1.
      P2c: symbolic. |rho e^{i th}|^2 = rho^2 (theta is GONE) while
           |A e^u + B e^{i v}|^2 = A^2 e^{2u} + 2 A B e^u cos(v) + B^2.
           d/dv of the polar weight is identically 0; d/dv of the additive
           weight is -2 A B e^u sin(v), not identically 0.
      Structural: a polar form is modulus TIMES phase, and the modulus-squared
      of a product kills the phase identically. Interference comes from ADDITION
      of amplitudes, never from polar decomposition of one. So the additive
      notation is the ONLY one of the two that produces the cross term the
      physics needs.

  P3  n_hat CONTRADICTS the manuscript's own centrality argument.
      CAL_Unified_Manual.tex lines 540-541: the phase unit is "the privileged
      commuting imaginary of the C factor, not a quaternionic imaginary".
      Lines 604-607: it "must not depend on a preferred quaternionic direction".
      born_axiomatic.tex Axiom 2 says the same.
      P3a: the central i = h*1 commutes with i_q, j_q, k_q.
      P3b: n_hat = i_q does NOT commute with j_q or k_q.
      P3c: over seeded random biquaternions, h*1 commutes with 100 percent of
           them (it is central); i_q commutes with 0 percent.
      So n_hat IS a preferred quaternionic direction, foreclosed by name.

  P4  J = rho e^{n theta} lives entirely in the SLICE C_n = span{1, n_hat}:
      2 of the algebra's 8 real dimensions, and isomorphic to C. Compute
      exp(u*1 + v*n_hat) by matrix exponential through the cal M2(C) rep and
      show the j and k components are exactly zero. The polar form does not
      rescue the quaternionic content; it makes the collapse explicit.

  P5  "Laplacian J = 0" is TRUE but TRIVIAL. For F = z, z^2, -2 log(z), and for
      seeded random holomorphic polynomials, lap(Re e^F) = lap(Im e^F) = 0.
      This is just "holomorphic implies harmonic" (Delta = 4 d_z d_zbar, and
      d_zbar kills a holomorphic function). It holds for EVERY holomorphic F,
      hence is not evidence for this one. Control: a NON-holomorphic F has
      nonzero Laplacian, confirming holomorphy is what is doing the work.

Run from inside "py tests":  python exp_polar_form_interference.py
"""
import os
import sys

import numpy as np
import sympy as sp
import torch

# cal/ lives at the repo root, one level above "py tests"; this script is run from inside
# "py tests" as: python exp_polar_form_interference.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cal.biquaternion import (quat_mul, hermitian_conj, biquat_to_matrix,
                              matrix_to_biquat, CDTYPE)

torch.manual_seed(0)
RNG = np.random.default_rng(20260714)

# cal uses complex64; float32 precision throughout the torch path.
TOL = 1e-4

# Biquaternion basis in cal coordinates (..., 4) = (1, i_q, j_q, k_q).
ONE = torch.tensor([1.0 + 0j, 0, 0, 0], dtype=CDTYPE)
I_Q = torch.tensor([0, 1.0 + 0j, 0, 0], dtype=CDTYPE)
J_Q = torch.tensor([0, 0, 1.0 + 0j, 0], dtype=CDTYPE)
K_Q = torch.tensor([0, 0, 0, 1.0 + 0j], dtype=CDTYPE)

# The CENTRAL imaginary of the C factor of C (x) H: h times the identity.
I_CENTRAL = torch.tensor([1j, 0, 0, 0], dtype=CDTYPE)

# The proposed polar phase unit: a fixed unit imaginary QUATERNION.
N_HAT = I_Q


def ok(b):
    return "PASS" if b else "**FAIL**"


def born0(q):
    """Born weight <q qdag>_0 = sum_n |q_n|^2, via the cal closure."""
    return quat_mul(q, hermitian_conj(q))[..., 0].real


def commutator(p, q):
    return quat_mul(p, q) - quat_mul(q, p)


def commutes(p, q, tol=TOL):
    return bool(commutator(p, q).abs().max() < tol)


def biquat_exp(q):
    """exp(q) through the cal M2(C) representation."""
    return matrix_to_biquat(torch.linalg.matrix_exp(biquat_to_matrix(q)))


def rand_biquat(n):
    return (torch.randn(n, 4) + 1j * torch.randn(n, 4)).to(CDTYPE)


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def precondition():
    """biquat_to_matrix must be an algebra homomorphism for matrix_exp to be
    the algebra exponential. Checked before P4 relies on it."""
    banner("P0. PRECONDITION: cal's M2(C) rep is an algebra homomorphism")
    p, q = rand_biquat(200), rand_biquat(200)
    lhs = biquat_to_matrix(quat_mul(p, q))
    rhs = biquat_to_matrix(p) @ biquat_to_matrix(q)
    hom = torch.allclose(lhs, rhs, atol=1e-3, rtol=1e-4)
    rt = torch.allclose(matrix_to_biquat(biquat_to_matrix(p)), p, atol=TOL)
    print(f"  M(p q) = M(p) M(q):                    {ok(hom)}  "
          f"max|diff|={(lhs - rhs).abs().max():.2e}")
    print(f"  matrix_to_biquat(biquat_to_matrix(q)) = q:  {ok(rt)}")
    print("  -> matrix_exp through the rep IS the algebra exponential. P4 may use it.")
    return hom and rt


def p1_different_objects():
    banner("P1. polar rho*e^{n theta} and additive A e^u + B e^{i v} DIFFER")
    print("  polar    : e^{u + n v} = e^u e^{n v} = e^u cos v  +  n * e^u sin v")
    print("  additive : A e^u + B e^{i v} = (A e^u + B cos v)  +  i * (B sin v)")
    print()
    print(f"  {'u':>6}{'v':>6}{'polar Re':>12}{'polar Im':>12}"
          f"{'add Re':>12}{'add Im':>12}{'same?':>8}")
    all_differ = True
    for uv, vv in [(0.3, 0.7), (-0.5, 2.1), (1.0, np.pi)]:
        pr, pim = np.exp(uv) * np.cos(vv), np.exp(uv) * np.sin(vv)
        ar, ai = np.exp(uv) + np.cos(vv), np.sin(vv)
        same = np.allclose([pr, pim], [ar, ai], atol=1e-9)
        all_differ = all_differ and not same
        print(f"  {uv:>6.2f}{vv:>6.2f}{pr:>12.4f}{pim:>12.4f}"
              f"{ar:>12.4f}{ai:>12.4f}{str(same):>8}")
    print(f"\n  P1  polar is NOT a re-notation of additive:  {ok(all_differ)}")
    print("      In the polar form e^u MULTIPLIES the phase; in the additive form the")
    print("      two channels sit side by side. They are different fields.")
    return all_differ


def p2_no_interference():
    banner("P2. DECISIVE: the polar form has NO INTERFERENCE. At all.")
    print("  |e^{u + n v}|^2 = |e^u|^2 * |e^{n v}|^2 = e^{2u} * 1 = e^{2u}")
    print("  because |e^{n v}| = |cos v + n_hat sin v| = sqrt(cos^2 v + sin^2 v) = 1.")
    print("  Both columns below are the cal Born closure <q qdag>_0 on the")
    print("  biquaternion, computed identically. Only the FORM of q differs.\n")

    u_val, A, B = 0.3, 1.0, 1.0
    vs = np.linspace(0.0, 2.0 * np.pi, 7)

    polar_col, add_col = [], []
    print(f"  {'v = S/hbar':>12}{'polar |J|^2':>14}{'additive |W|^2':>17}"
          f"{'analytic e^{2u}':>18}")
    for vv in vs:
        # polar: J = exp(u*1 + v*n_hat), a genuine biquaternion exponential.
        J = biquat_exp((u_val * ONE + float(vv) * N_HAT).to(CDTYPE))
        pj = float(born0(J))
        # additive: W = A e^u * 1 + B e^{h v} * 1, a complex SCALAR biquaternion,
        # exactly as Eq. (J-source) has it (manuscript line 544).
        Wc = A * np.exp(u_val) + B * np.exp(1j * vv)
        W = torch.tensor([complex(Wc), 0, 0, 0], dtype=CDTYPE)
        aw = float(born0(W))
        polar_col.append(pj)
        add_col.append(aw)
        print(f"  {vv:>12.4f}{pj:>14.6f}{aw:>17.6f}{np.exp(2*u_val):>18.6f}")

    polar_col, add_col = np.array(polar_col), np.array(add_col)
    polar_spread = polar_col.max() - polar_col.min()
    add_spread = add_col.max() - add_col.min()
    p2a = bool(polar_spread < TOL and add_spread > 1.0)
    p2b = bool(np.allclose(polar_col, np.exp(2 * u_val), atol=TOL))

    print(f"\n  P2a polar column CONSTANT (spread={polar_spread:.2e} < {TOL:.0e}) while")
    print(f"      additive column VARIES (spread={add_spread:.4f}):        {ok(p2a)}")
    print(f"  P2b polar weight = e^{{2u}} = {np.exp(2*u_val):.6f} for every v:    {ok(p2b)}")
    print("      The action v = S/hbar_eff does not enter the Born weight at all:")
    print("      |rho e^{n theta}|^2 = rho^2 = e^{-2H/tau}. The additive column's")
    print("      variation IS the interference.")

    # ---- symbolic ----
    print("\n  P2c symbolic confirmation:")
    u, v = sp.symbols('u v', real=True)
    A_s, B_s = sp.symbols('A B', real=True, positive=True)
    rho, th = sp.symbols('rho theta', real=True, positive=True)

    polar_mod2 = sp.simplify(sp.Abs(rho * sp.exp(sp.I * th))**2)
    theta_gone = (polar_mod2 == rho**2) and (th not in polar_mod2.free_symbols)

    W_s = A_s * sp.exp(u) + B_s * sp.exp(sp.I * v)
    add_mod2 = sp.simplify(sp.expand(W_s * sp.conjugate(W_s)))
    closed = A_s**2 * sp.exp(2*u) + 2*A_s*B_s*sp.exp(u)*sp.cos(v) + B_s**2
    cross_ok = sp.simplify(add_mod2 - closed) == 0

    d_polar = sp.simplify(sp.diff(polar_mod2, th))
    d_add = sp.simplify(sp.diff(closed, v))
    d_add_expected = sp.simplify(d_add - (-2*A_s*B_s*sp.exp(u)*sp.sin(v))) == 0
    deriv_ok = (d_polar == 0) and d_add_expected and (d_add != 0)

    print(f"      |rho e^{{n theta}}|^2         = {polar_mod2}")
    print(f"          theta eliminated:                            {ok(theta_gone)}")
    print(f"      |A e^u + B e^{{i v}}|^2       = {add_mod2}")
    print(f"          equals A^2 e^{{2u}} + 2 A B e^u cos(v) + B^2:   {ok(cross_ok)}")
    print(f"      d/d(theta) polar weight   = {d_polar}   (identically zero)")
    print(f"      d/dv       additive weight= {d_add}")
    print(f"          polar has no v-dependence, additive does:    {ok(deriv_ok)}")

    print("\n  >> This is structural, not incidental. A polar form is modulus TIMES")
    print("     phase, and the modulus-squared of a product kills the phase")
    print("     identically. Interference comes from ADDITION of amplitudes, never")
    print("     from polar decomposition of one. So the additive notation is not")
    print("     'obscuring' the structure: it is the ONLY one of the two that")
    print("     produces the cross term 2 A B e^u cos(v) the physics needs.")
    return p2a and p2b and theta_gone and cross_ok and deriv_ok


def p3_centrality():
    banner("P3. n_hat is a PREFERRED quaternionic direction, foreclosed by name")
    print("  CAL_Unified_Manual.tex lines 540-541: the unit i is 'the privileged")
    print("  commuting imaginary of the C factor, not a quaternionic imaginary'.")
    print("  Lines 604-607: 'the imaginary unit of e^{iS/hbar} must commute with")
    print("  everything in the rest of the algebra, since the phase is a global")
    print("  object on the action and must not depend on a preferred quaternionic")
    print("  direction.' born_axiomatic.tex Axiom 2 says the same.\n")

    print("  Does the CENTRAL i = h*1 commute with the quaternion units?")
    central_all = True
    for nm, U in [('i_q', I_Q), ('j_q', J_Q), ('k_q', K_Q)]:
        c = commutes(I_CENTRAL, U)
        central_all = central_all and c
        print(f"    [i_central, {nm}] = 0 ?  {c}   "
              f"max|comm|={commutator(I_CENTRAL, U).abs().max():.2e}")

    print("\n  Does the proposed n_hat = i_q commute with the others?")
    nhat_fails = []
    for nm, U in [('i_q', I_Q), ('j_q', J_Q), ('k_q', K_Q)]:
        c = commutes(N_HAT, U)
        nhat_fails.append(not c)
        print(f"    [n_hat, {nm}]     = 0 ?  {c}   "
              f"max|comm|={commutator(N_HAT, U).abs().max():.2e}")

    p3a = central_all
    p3b = nhat_fails[1] and nhat_fails[2]   # fails against j_q and k_q
    print(f"\n  P3a central i commutes with i_q, j_q, k_q:            {ok(p3a)}")
    print(f"  P3b n_hat = i_q does NOT commute with j_q or k_q:     {ok(p3b)}")

    # P3c: centrality against seeded random elements of the whole algebra.
    Q = rand_biquat(500)
    n = Q.shape[0]
    c_central = commutator(I_CENTRAL.expand(n, 4), Q).abs().amax(dim=-1)
    c_nhat = commutator(N_HAT.expand(n, 4), Q).abs().amax(dim=-1)
    frac_central = float((c_central < 1e-3).float().mean()) * 100.0
    frac_nhat = float((c_nhat < 1e-3).float().mean()) * 100.0
    p3c = bool(frac_central == 100.0 and frac_nhat == 0.0)
    print(f"\n  Over {n} seeded random biquaternions:")
    print(f"    h*1 commutes with {frac_central:6.2f} percent of them  "
          f"(max|comm|={c_central.max():.2e})")
    print(f"    i_q commutes with {frac_nhat:6.2f} percent of them  "
          f"(median|comm|={c_nhat.median():.3f})")
    print(f"  P3c h*1 is central, i_q is not:                       {ok(p3c)}")

    print("\n  >> The central i commutes with everything. A quaternion unit does not.")
    print("     n_hat IS a preferred quaternionic direction: the exact thing the")
    print("     manuscript says centrality is forced to avoid. The polar form")
    print("     contradicts the stated justification for using C (x) H at all.")
    return p3a and p3b and p3c


def p4_slice_collapse():
    banner("P4. J = rho e^{n theta} collapses into the slice C_n = span{1, n_hat}")
    print("  C_n = {a + n_hat b : a, b real} is a 2-real-dimensional subalgebra of H")
    print("  (4 real dims), sitting inside C (x) H (8 real dims). It is isomorphic")
    print("  to C. Components below are over the cal basis (1, i_q, j_q, k_q).\n")

    all_collapse = True
    for uv, vv in [(0.3, 0.7), (-0.4, 2.0), (1.0, np.pi)]:
        J = biquat_exp((uv * ONE + vv * N_HAT).to(CDTYPE))
        comps = J.numpy()
        jk_zero = bool(np.allclose([comps[2], comps[3]], 0, atol=TOL))
        # the 1 and n_hat components should be real e^u cos v, e^u sin v
        expect = np.allclose([comps[0].real, comps[1].real],
                             [np.exp(uv)*np.cos(vv), np.exp(uv)*np.sin(vv)], atol=TOL)
        all_collapse = all_collapse and jk_zero and expect
        print(f"    u={uv:>5}, v={vv:>6.3f}:  J = exp(u + n_hat v) over (1, i_q, j_q, k_q):")
        print(f"      {np.round(comps, 5)}")
        print(f"      j and k components zero? {jk_zero}   "
              f"matches (e^u cos v, e^u sin v)? {expect}")

    n_used = 2
    print(f"\n  P4  J uses {n_used} of the algebra's 8 real dimensions:      {ok(all_collapse)}")
    print("      The j_q and k_q generators never appear, and the components are")
    print("      real, so the complex factor never appears either. This is the same")
    print("      collapse flagged at the Born step: the biquaternion machinery is")
    print("      decorative there. The polar form does not rescue it. It makes the")
    print("      collapse explicit.")
    return all_collapse


def p5_laplacian_trivial():
    banner("P5. 'Laplacian J = 0' is TRUE, and trivial")
    x, y = sp.symbols('x y', real=True)
    z = x + sp.I * y

    cases = [('F = z', z), ('F = z^2', z**2), ('F = -2 log(z)', -2 * sp.log(z))]
    # seeded random holomorphic polynomials: the point is it holds for EVERY
    # holomorphic F, not for this particular one.
    for _ in range(2):
        c = RNG.integers(-3, 4, size=3)
        Fr = sp.Integer(int(c[0])) * z**2 + sp.Integer(int(c[1])) * z + sp.Integer(int(c[2]))
        cases.append((f'F = {sp.sstr(sp.expand(Fr)).replace("I*y + x", "z")}'[:26], Fr))

    print("  Holomorphic F (the claim's cases, plus seeded random ones):")
    all_harmonic = True
    for name, F in cases:
        Jf = sp.exp(F)
        lap_re = sp.simplify(sp.diff(sp.re(Jf), x, 2) + sp.diff(sp.re(Jf), y, 2))
        lap_im = sp.simplify(sp.diff(sp.im(Jf), x, 2) + sp.diff(sp.im(Jf), y, 2))
        harmonic = (lap_re == 0) and (lap_im == 0)
        all_harmonic = all_harmonic and harmonic
        print(f"    {name:<28} lap(Re e^F) = {lap_re}   lap(Im e^F) = {lap_im}   "
              f"{ok(harmonic)}")

    # Control. Delta = 4 d_z d_zbar, so BOTH holomorphic (d_zbar F = 0) and
    # ANTI-holomorphic (d_z F = 0) exponents give a harmonic e^F. zbar is
    # anti-holomorphic, so it is harmonic too and is NOT a valid control -- it is
    # kept here because it sharpens the point: the harmonicity is weaker still
    # than "holomorphic implies harmonic". A real control must be neither
    # holomorphic nor anti-holomorphic, e.g. |z|^2 = z zbar.
    print("\n  CONTROL, dropping holomorphy (is holomorphy doing the work?):")
    ctrl_nonzero = True
    for name, F, is_ctrl in [('F = zbar (ANTI-holomorphic)', x - sp.I*y, False),
                             ('F = |z|^2 (neither)', x**2 + y**2, True)]:
        Jf = sp.exp(F)
        lap_re = sp.simplify(sp.diff(sp.re(Jf), x, 2) + sp.diff(sp.re(Jf), y, 2))
        nonzero = lap_re != 0
        if is_ctrl:
            ctrl_nonzero = ctrl_nonzero and nonzero
        note = '' if is_ctrl else '   <- also harmonic, as d_z kills it'
        print(f"    {name:<28} lap(Re e^F) = {lap_re}   nonzero? {nonzero}{note}")

    print(f"\n  P5  every holomorphic F gives lap(Re e^F) = lap(Im e^F) = 0:  "
          f"{ok(all_harmonic)}")
    print(f"      a neither-holo-nor-antiholo control IS non-harmonic:      "
          f"{ok(ctrl_nonzero)}")
    print("\n  >> Yes, Laplacian J = 0. But that is just 'holomorphic implies")
    print("     harmonic' (Delta = 4 d_z d_zbar, and d_zbar kills a holomorphic")
    print("     function). It holds for EVERY holomorphic F, as the random cases")
    print("     show, and fails as soon as holomorphy is dropped, as the control")
    print("     shows. It is not evidence for this F, and it buys nothing the Born")
    print("     closure needs: by P2 the modulus of e^F is e^{Re F} and the phase")
    print("     has already dropped out.")
    return all_harmonic and ctrl_nonzero


def main():
    print("=" * 78)
    print("POLAR FORM vs ADDITIVE SOURCE CURRENT -- interference, centrality, slice")
    print("NEGATIVE RESULT: the polar form fails three ways.")
    print("=" * 78)

    results = {
        'P0 rep is a homomorphism (precondition)': precondition(),
        'P1 polar != additive (different objects)': p1_different_objects(),
        'P2 polar has NO interference (decisive)': p2_no_interference(),
        'P3 n_hat is a preferred quaternionic direction': p3_centrality(),
        'P4 polar J collapses to the 2-dim slice C_n': p4_slice_collapse(),
        'P5 Laplacian J = 0 is true but trivial': p5_laplacian_trivial(),
    }

    banner("SUMMARY")
    for k, v in results.items():
        print(f"  {ok(v):>8}  {k}")

    print("\n  VERDICT (kept as a negative):")
    print("  The polar form J = rho e^{n_hat theta} is not a clarifying re-notation")
    print("  of A e^{-H/tau} + B e^{iS/hbar_eff}. It (1) destroys the interference")
    print("  cross term that the Born weight depends on, (2) reintroduces the")
    print("  preferred quaternionic direction that the manuscript's own centrality")
    print("  argument at lines 540-541 and 604-607 explicitly forecloses, and (3)")
    print("  confines J to a 2-of-8-real-dimensional slice isomorphic to C. The")
    print("  additive notation does not obscure the structure; it is the only one")
    print("  of the two that carries it. The 'Laplacian J = 0' observation is true")
    print("  of every holomorphic F and is not evidence for this one.")

    if not all(results.values()):
        raise SystemExit("At least one claim did not reproduce. See **FAIL** above.")


if __name__ == "__main__":
    main()
