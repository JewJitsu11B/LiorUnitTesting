"""
Are the two CAL channel drivers HOLOMORPHIC DUALS (harmonic conjugates)?

The proposal: with f := -2 log(psi) for holomorphic psi,
    Re f = -2 ln|psi| = -ln|psi|^2 = H^D   (the thermal driver -- Step 1 of thm:Born_fp)
    Im f = -2 arg(psi)                     (the phase driver)
so the thermal channel's harmonic conjugate is the wavefunction's PHASE, which is exactly
the object quantum interference is made of. This script tests that proposal as mathematics,
and then tests it as a description of the manuscript. The two verdicts disagree.

CLAIMS VERIFIED (pure numpy/sympy float64; cal package not needed):
  H1  The mathematics CHECKS OUT. For psi = z, z^2+1, exp(z), exp(z)/(z^2+1):
        laplacian(ln|psi|) = 0, and BOTH Cauchy-Riemann relations hold,
        u_x - v_y = 0 and u_y + v_x = 0, for u = ln|psi|, v = arg(psi).
      Confirmed symbolically and at seeded random sample points. So H^D is harmonic and its
      holomorphic dual is arg(psi). The structure the author proposes is real and is good.
  H2  Under that reading the phase channel genuinely interferes. For holomorphic psi = z + i
      along the line y = 0, W = A|psi|^{2/tau} + B exp(-2i arg(psi)/hbar) has a
      position-dependent phase; |W|^2 VARIES over the tabulated line and has a TRUE NULL
      (exactly 0) at x = 0, where |psi| = 1 and exp(-2i arg psi) = exp(-i pi) = -1.
      The frozen-action objection does not survive THIS reading.

NEGATIVE FINDINGS (reported and kept as negatives, not reframed):
  M1  REFUTED as a description of the manuscript. A multi-agent audit of the current
      CAL_Unified_Manual.tex found NO Cauchy-Riemann condition, analyticity requirement, or
      Hilbert-transform relation anywhere: "Cauchy" 0 hits, "Riemann" 0 hits, "harmonic
      conjugate" 0 hits, "Hilbert transform" 0 hits; "harmonic" occurs only as harmonic gauge
      / harmonic oscillator. sec:source introduces H^D and S^D as two separately named
      drivers with no relation asserted between them. Re-run live when the .tex is reachable
      (see MANUAL_TEX below); otherwise the recorded audit counts are reported as recorded.
  M2  The manuscript takes the OPPOSITE road. It writes S := B exp(i S^nu/hbar_eff)
      = B |psi|^{2/tau}. Sympy: for REAL S^nu the modulus |B exp(i*real)| = B identically, so
      that equation forces S^nu IMAGINARY:
        exp(i S^nu/hbar) = |psi|^{2/tau}  =>  S^nu = -2i hbar_eff log|psi| / tau.
      That is a Wick rotation. An imaginary S is NOT the harmonic conjugate of H^D; it is i
      times it. Tabulated side by side: the harmonic-conjugate reading gives the phase channel
      modulus 1 with a varying argument; the manuscript's line gives modulus |psi|^{2/tau}
      with NO phase. They cannot both hold, and Steps 3-4 need the second.
  D1  DECISIVE, and it is the manuscript's own closure that does the damage. def:J-closed
      sets "At closure the two coincide, theta_R = theta_I =: varphi(x,t)". With
      theta_R = -H^D/tau = (2/tau) u and theta_I = S^D/hbar = (-2/hbar) v, closure says
      v = c u for the REAL constant c = -hbar/tau. Imposing that together with
      Cauchy-Riemann gives, symbolically,
          (1 + c^2) u_x = 0,
      and 1 + c^2 has NO real root, so u_x = u_y = 0: u is CONSTANT and J is constant.
      Verified for c = 1 (the literal reading) and for seeded random real c (so it is not an
      artifact of dropping the tau and hbar scale factors). The only roots of 1 + c^2 are
      c = -i, +i: the ratio theta_I/theta_R must be IMAGINARY, which is precisely the Wick
      rotation of M2 and abandons the real harmonic-conjugate structure the claim rests on.
      So holomorphy and def:J-closed are mutually exclusive except for constant fields.
  D2  The weaker reading does not rescue it either. If closure is asked to hold only where
      theta_R = theta_I rather than identically, the closure set for psi = z is a log spiral
      r = exp(-2 theta): a measure-zero CURVE. The fraction of grid points satisfying closure
      shrinks LINEARLY with the tolerance (the signature of a curve, not of an open set), so
      varphi(x,t) is not well defined as a FIELD on any open set and eq:J is not a field
      equation there.

Not covered here (see exp_two_closures.py, C1-C7): the two biquaternion closures, the reduced
norm, multiplicativity, and the null cone. This script adds only the holomorphic-dual /
harmonic-conjugate analysis of the two channel drivers.
"""
import os
import re

import numpy as np
import sympy as sp

rng = np.random.default_rng(7)
np.set_printoptions(precision=6, suppress=True)

TOL = 1e-10          # pure float64 / exact sympy sections

# Optional live re-run of the M1 grep audit. The manuscript lives in the SEPARATE manuscript
# repo, not in this test repo, so it is not reachable from here in general. Point
# CAL_MANUAL_TEX at CAL_Unified_Manual.tex to re-run the audit live; otherwise the recorded
# audit counts below are reported as recorded.
MANUAL_TEX = os.environ.get("CAL_MANUAL_TEX", "")

# Audit record: counts confirmed against CAL_Unified_Manual.tex at the time of writing.
RECORDED_AUDIT = [("Cauchy", 0), ("Riemann", 0), ("harmonic conjugate", 0),
                  ("Hilbert transform", 0)]

x, y = sp.symbols("x y", real=True)
z = x + sp.I * y


def ok(b):
    return "PASS" if b else "**FAIL**"


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------------------
# H1. The mathematics: H^D = -ln|psi|^2 is harmonic, and its dual is arg(psi).
# ---------------------------------------------------------------------------
def h1_harmonic_conjugates():
    banner("H1. The holomorphic-dual claim CHECKS OUT as mathematics")
    print("  f := -2 log(psi) is holomorphic wherever psi is holomorphic and nonzero, so")
    print("      Re f = -2 ln|psi| = -ln|psi|^2 = H^D    <-- the THERMAL driver (Step 1)")
    print("      Im f = -2 arg(psi)                      <-- the PHASE driver")
    print("  If that is right then u := ln|psi| is harmonic and v := arg(psi) is its")
    print("  harmonic conjugate. Checking symbolically on concrete holomorphic psi:\n")

    cases = [("psi = z", z),
             ("psi = z^2+1", z**2 + 1),
             ("psi = exp(z)", sp.exp(z)),
             ("psi = z^3-2z", z**3 - 2 * z),
             ("psi = exp(z)/(z^2+1)", sp.exp(z) / (z**2 + 1))]

    print(f"  {'psi':<22}{'lap(ln|psi|)':>14}{'CR1: u_x-v_y':>16}{'CR2: u_y+v_x':>16}")
    all_ok = True
    for name, psi_e in cases:
        # expand_log splits log(a/b) into log(a) - log(b), which sympy's simplify can then
        # actually reduce; without it the quotient case leaves unevaluated re(...)/im(...)
        # atoms and simplify stalls. force=True is legitimate HERE because the split is exact
        # for u = ln|psi| (ln|ab| = ln|a| + ln|b| always) and is correct for v = arg(psi) up
        # to an additive 2*pi*k branch constant, which is annihilated by the derivatives that
        # the Laplacian and both CR relations are built from.
        u, v = sp.expand_log(sp.log(psi_e), force=True).as_real_imag()   # u = ln|psi|, v = arg psi
        lap_u = sp.simplify(sp.diff(u, x, 2) + sp.diff(u, y, 2))
        cr1 = sp.simplify(sp.diff(u, x) - sp.diff(v, y))
        cr2 = sp.simplify(sp.diff(u, y) + sp.diff(v, x))
        good = (lap_u == 0 and cr1 == 0 and cr2 == 0)
        all_ok &= good
        print(f"  {name:<22}{str(lap_u):>14}{str(cr1):>16}{str(cr2):>16}   {ok(good)}")

    print(f"\n  H1 symbolic: harmonic AND both Cauchy-Riemann relations hold:  {ok(all_ok)}")

    # Independent numerical corroboration at seeded random points, so the symbolic
    # simplification is not the only witness.
    print("\n  Numerical corroboration at seeded random points (central differences, h=1e-5):")
    h = 1e-5
    num_ok = True
    for name, psi_fn in [("psi = z", lambda w: w),
                         ("psi = z^2+1", lambda w: w**2 + 1),
                         ("psi = exp(z)", lambda w: np.exp(w))]:
        pts = rng.uniform(0.5, 1.5, size=(6, 2))       # away from zeros/branch cut
        u_fn = lambda a, b: np.log(np.abs(psi_fn(a + 1j * b)))
        v_fn = lambda a, b: np.angle(psi_fn(a + 1j * b))
        lap_m = cr1_m = cr2_m = 0.0
        for a, b in pts:
            uxx = (u_fn(a + h, b) - 2 * u_fn(a, b) + u_fn(a - h, b)) / h**2
            uyy = (u_fn(a, b + h) - 2 * u_fn(a, b) + u_fn(a, b - h)) / h**2
            ux = (u_fn(a + h, b) - u_fn(a - h, b)) / (2 * h)
            uy = (u_fn(a, b + h) - u_fn(a, b - h)) / (2 * h)
            vx = (v_fn(a + h, b) - v_fn(a - h, b)) / (2 * h)
            vy = (v_fn(a, b + h) - v_fn(a, b - h)) / (2 * h)
            lap_m = max(lap_m, abs(uxx + uyy))
            cr1_m = max(cr1_m, abs(ux - vy))
            cr2_m = max(cr2_m, abs(uy + vx))
        good = (lap_m < 1e-3 and cr1_m < 1e-6 and cr2_m < 1e-6)   # lap: h^-2 differencing
        num_ok &= good
        print(f"    {name:<14} max|lap u|={lap_m:.2e}  max|u_x-v_y|={cr1_m:.2e}  "
              f"max|u_y+v_x|={cr2_m:.2e}  {ok(good)}")
    print(f"\n  H1 numerical: {ok(num_ok)}")
    print("  >> The author is RIGHT that H^D has a holomorphic dual, and that the dual is")
    print("     the PHASE arg(psi). f = -2 log(psi) is the holomorphic function whose real")
    print("     and imaginary readings are the two channel drivers. This is a good structure.")
    return all_ok and num_ok


# ---------------------------------------------------------------------------
# H2. Under that reading the phase channel really does interfere.
# ---------------------------------------------------------------------------
def h2_interference():
    banner("H2. Under the harmonic-conjugate reading, |W|^2 varies and has a true null")
    print("  theta_R = -H^D/tau = (2/tau) ln|psi|      theta_I = S^D/hbar = -2 arg(psi)/hbar")
    print("  so the two channels are the MODULUS and the ARGUMENT of psi:")
    print("      W = A|psi|^{2/tau} + B exp(-2i arg(psi)/hbar)")
    print("  arg(psi) varies with position, so the phase channel carries a genuine,")
    print("  position-dependent phase. Tabulating along y = 0 for holomorphic psi = z + i,")
    print("  with A = B = 1, tau = 2, hbar = 1:\n")

    A = B = 1.0
    tau, hbar = 2.0, 1.0
    xs = np.linspace(-2, 2, 9)
    psi_v = xs + 1j * 1.0                              # psi = z + i on the line y = 0

    print(f"  {'x':>7}{'|psi|':>9}{'arg(psi)':>11}{'phase channel':>26}{'|W|^2':>10}")
    w2 = []
    for xv, pv in zip(xs, psi_v):
        ch = B * np.exp(-2j * np.angle(pv) / hbar)
        Wv = A * abs(pv) ** (2 / tau) + ch
        w2.append(abs(Wv) ** 2)
        print(f"  {xv:>7.2f}{abs(pv):>9.4f}{np.angle(pv):>11.4f}"
              f"{str(np.round(ch, 4)):>26}{abs(Wv) ** 2:>10.4f}")
    w2 = np.array(w2)

    varies = (w2.max() - w2.min()) > 1e-6
    i0 = int(np.argmin(np.abs(xs)))                    # x = 0 is on the grid
    null_val = w2[i0]
    true_null = null_val < TOL
    print(f"\n  |W|^2 range: min={w2.min():.4e}  max={w2.max():.4f}  spread={w2.max()-w2.min():.4f}")
    print(f"  |W|^2 VARIES along the line:                           {ok(varies)}")
    print(f"  TRUE NULL at x = 0: |W|^2 = {null_val:.3e}                {ok(true_null)}")
    print(f"    at x = 0, psi = i: |psi| = {abs(psi_v[i0]):.4f}, arg = pi/2, so the phase")
    print(f"    channel is exp(-i pi) = {np.exp(-1j*np.pi):.1f}, and W = 1 + (-1) = 0 exactly.")
    print("  >> Real destructive interference at a point, from a real position-dependent")
    print("     phase. The 'the action is frozen' objection does NOT survive this reading.")
    return varies and true_null


# ---------------------------------------------------------------------------
# M1. But this is not what the manuscript says.
# ---------------------------------------------------------------------------
def m1_manuscript_audit():
    banner("M1. NEGATIVE: the manuscript never asserts any of this")
    live = bool(MANUAL_TEX) and os.path.isfile(MANUAL_TEX)
    if live:
        with open(MANUAL_TEX, "r", encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        print(f"  Live grep audit of {MANUAL_TEX}\n")
        counts = [(term, len(re.findall(re.escape(term), src, flags=re.IGNORECASE)))
                  for term, _ in RECORDED_AUDIT]
    else:
        print("  Manuscript .tex not reachable from this repo (set CAL_MANUAL_TEX to re-run")
        print("  the grep live). Reporting the RECORDED audit counts as recorded:\n")
        counts = RECORDED_AUDIT

    for term, n in counts:
        print(f"    grep -i {term!r:<22} -> {n} hits")
    clean = all(n == 0 for _, n in counts)
    print(f"\n  No Cauchy-Riemann / analyticity / Hilbert-transform machinery present: {ok(clean)}")
    if live:
        agree = counts == RECORDED_AUDIT
        print(f"  Live counts match the recorded audit:                  {ok(agree)}")
        clean = clean and agree
    print("  'harmonic' occurs only as harmonic gauge / harmonic oscillator; 'harmonic")
    print("  conjugate' and 'Hilbert transform' never appear. sec:source introduces H^D and")
    print("  S^D as two separately named drivers with NO relation asserted between them.")
    print("  >> The holomorphic-dual structure is a good idea that is NOT in the manuscript.")
    print("     It would have to be added, with analyticity of psi stated as a hypothesis.")
    return clean


# ---------------------------------------------------------------------------
# M2. The manuscript takes the opposite road: it forces S^nu imaginary.
# ---------------------------------------------------------------------------
def m2_wick_rotation():
    banner("M2. NEGATIVE: the manuscript's own line forces S^nu IMAGINARY (a Wick rotation)")
    print("  CAL_Unified_Manual_pre-fold.tex l.2294-2297 writes:")
    print("      S := B exp(i S^nu / hbar_eff) = B |psi|^{2/tau}\n")

    hb, tau, B = sp.symbols("hbar_eff tau B", real=True, positive=True)
    m = sp.symbols("m", real=True, positive=True)          # m := |psi|
    Snu_r = sp.symbols("S_nu_real", real=True)

    mod_real = sp.simplify(sp.Abs(B * sp.exp(sp.I * Snu_r / hb)))
    forced = sp.simplify(mod_real - B) == 0
    print(f"  Step 1. For REAL S^nu, sympy gives |B exp(i S^nu/hbar_eff)| = {mod_real}")
    print(f"          The modulus is identically B, independent of S^nu:  {ok(forced)}")
    print("          So 'B exp(i S^nu/hbar) = B m^{2/tau}' forces m^{2/tau} = 1, i.e.")
    print("          |psi| = 1 EVERYWHERE -- unless S^nu is complex.\n")

    Snu_c = sp.symbols("S_nu", complex=True)
    sol = sp.solve(sp.Eq(sp.exp(sp.I * Snu_c / hb), m ** (2 / tau)), Snu_c)
    print(f"  Step 2. Solving exp(i S^nu/hbar_eff) = m^{{2/tau}} for S^nu:")
    print(f"          S^nu = {sol}")
    expected = -2 * sp.I * hb * sp.log(m) / tau
    matches = any(sp.simplify(s - expected) == 0 for s in sol)
    print(f"          i.e. S^nu = -2i hbar_eff log|psi| / tau:           {ok(matches)}")
    is_imag = all(sp.simplify(sp.re(sp.expand(s.rewrite(sp.log)))) == 0 for s in sol)
    print(f"          PURELY IMAGINARY (zero real part):                 {ok(is_imag)}")
    print("\n  >> Step 2 of the manuscript requires the action to be IMAGINARY. That is a")
    print("     Wick rotation, and it is the OPPOSITE of the harmonic-conjugate reading:")
    print("       harmonic-conjugate reading:  S^D = -2 arg(psi)                 (REAL, a phase)")
    print("       the manuscript's line needs: S^nu = -2i hbar log|psi| / tau    (IMAGINARY)")
    print("     An imaginary S is NOT the harmonic conjugate of H^D. It is i times it.")
    return forced and matches and is_imag


# ---------------------------------------------------------------------------
# M3. The two readings side by side. They cannot both hold.
# ---------------------------------------------------------------------------
def m3_side_by_side():
    banner("M3. The two readings tabulated side by side (psi = z + i, y = 0, tau = 2, hbar = 1)")
    print("  Harmonic-conjugate reading: phase channel = B exp(-2i arg psi)   -> modulus 1")
    print("  The manuscript's line:      phase channel = B |psi|^{2/tau}      -> no phase\n")

    tau, hbar, B = 2.0, 1.0, 1.0
    xs = np.linspace(-2, 2, 5)
    psi_v = xs + 1j * 1.0

    print(f"  {'x':>7}|{'  harmonic conjugate: B exp(-2i arg psi)':<40}|"
          f"{'  l.2295: B |psi|^(2/tau)':<28}")
    print(f"  {'':>7}|{'modulus':>12}{'argument':>12}{'complex?':>14}|"
          f"{'modulus':>12}{'argument':>12}")
    hc_mods, hc_args, ms_args = [], [], []
    for xv, pv in zip(xs, psi_v):
        hc = B * np.exp(-2j * np.angle(pv) / hbar)
        ms = B * abs(pv) ** (2 / tau)
        hc_mods.append(abs(hc))
        hc_args.append(np.angle(hc))
        ms_args.append(np.angle(complex(ms, 0.0)))
        print(f"  {xv:>7.2f}|{abs(hc):>12.4f}{np.angle(hc):>12.4f}"
              f"{str(abs(np.imag(hc)) > 1e-9):>14}|{ms:>12.4f}{np.angle(complex(ms,0.0)):>12.4f}")

    hc_mods, hc_args, ms_args = map(np.array, (hc_mods, hc_args, ms_args))
    hc_mod_const1 = np.allclose(hc_mods, 1.0, atol=TOL)
    hc_arg_varies = (hc_args.max() - hc_args.min()) > 1e-6
    ms_no_phase = np.allclose(ms_args, 0.0, atol=TOL)
    print(f"\n  harmonic-conjugate reading: modulus is identically 1:  {ok(hc_mod_const1)}")
    print(f"  harmonic-conjugate reading: argument VARIES:           {ok(hc_arg_varies)}"
          f"   (spread {hc_args.max()-hc_args.min():.4f} rad)")
    print(f"  l.2295 reading: argument is identically 0 (NO phase):  {ok(ms_no_phase)}")
    print("  >> Mutually exclusive. One has modulus 1 and a varying phase; the other has")
    print("     modulus |psi|^{2/tau} and no phase. Steps 3-4 need the SECOND: they conclude")
    print("     W = G - S = (A-B)|psi|^{2/tau}, real, and W^dag W = (A-B)^2 |psi|^{4/tau},")
    print("     arithmetic that only works if S is a real MAGNITUDE. The manuscript chose the")
    print("     reading that kills the interference.")
    return hc_mod_const1 and hc_arg_varies and ms_no_phase


# ---------------------------------------------------------------------------
# D1. The decisive one: def:J-closed trivializes holomorphy.
# ---------------------------------------------------------------------------
def d1_closure_trivializes():
    banner("D1. DECISIVE NEGATIVE: def:J-closed + Cauchy-Riemann => the field is CONSTANT")
    print("  CAL_Unified_Manual.tex def:J-closed (label l.1054, body l.1052-1064) states:")
    print("    'Write theta_R := -H^D/tau and theta_I := S^D/hbar_eff ... At closure the two")
    print("     coincide, theta_R = theta_I =: varphi(x,t)'")
    print("  With u := ln|psi| and v := arg(psi) (the H1 pair):")
    print("    theta_R = -H^D/tau  = (2/tau) u        theta_I = S^D/hbar = (-2/hbar) v")
    print("  so closure says v = c u for the REAL constant c = -hbar/tau.\n")

    ux, uy, vx, vy = sp.symbols("u_x u_y v_x v_y", real=True)
    c = sp.symbols("c", real=True)        # the physical ratio -hbar/tau: REAL by construction
    c_any = sp.symbols("c_any")           # unrestricted, only to exhibit the complex roots

    print("  Imposing Cauchy-Riemann AND closure simultaneously:")
    print("    CR:       u_x - v_y = 0,   u_y + v_x = 0")
    print("    closure:  v = c u  =>  v_x = c u_x,  v_y = c u_y\n")

    # Elimination, shown explicitly.
    lhs = sp.simplify(sp.expand(ux - c * (-c * ux)))       # u_x = c u_y and u_y = -c u_x
    factored = sp.factor(lhs)
    print("    substitute: u_x = v_y = c u_y   and   u_y = -v_x = -c u_x")
    print(f"    chain:      u_x = c(-c u_x) = -c^2 u_x   =>   {factored} = 0")
    real_roots = sp.solve(sp.Eq(1 + c**2, 0), c)              # c declared real
    all_roots = sp.solve(sp.Eq(1 + c_any**2, 0), c_any)       # c unrestricted
    print(f"    roots of 1 + c^2 over the REALS:     {real_roots if real_roots else '[]  (NONE)'}")
    print(f"    roots of 1 + c^2 with c unrestricted: {all_roots}")
    no_real_escape = (len(real_roots) == 0 and all_roots == [-sp.I, sp.I])
    print(f"    1 + c^2 has NO real root, so u_x = 0, hence u_y = -c u_x = 0: {ok(no_real_escape)}")

    # Independent confirmation: solve the full linear system for the derivatives.
    print("\n  Independent confirmation, solving the 4-equation system for the derivatives:")
    eqs = [sp.Eq(ux, vy), sp.Eq(uy, -vx), sp.Eq(vx, c * ux), sp.Eq(vy, c * uy)]
    sol = sp.solve(eqs, [ux, uy, vx, vy], dict=True)
    print(f"    solve({{u_x=v_y, u_y=-v_x, v_x=c u_x, v_y=c u_y}}, [u_x,u_y,v_x,v_y])")
    print(f"      -> {sol}")
    only_trivial = (len(sol) == 1 and all(sp.simplify(val) == 0 for val in sol[0].values()))
    print(f"    the ONLY solution is u_x = u_y = v_x = v_y = 0:        {ok(only_trivial)}")

    # c = 1: the literal "theta_R = theta_I" reading.
    print("\n  D1a. The literal reading, c = 1 (theta_R = theta_I with u = v):")
    eqs1 = [e.subs(c, 1) for e in eqs]
    sol1 = sp.solve(eqs1, [ux, uy, vx, vy], dict=True)
    triv1 = (len(sol1) == 1 and all(sp.simplify(v_) == 0 for v_ in sol1[0].values()))
    print(f"    u_x = u_y = v_y = -v_x and u_y = -u_x  ->  {sol1}")
    print(f"    forced constant:                                      {ok(triv1)}")

    # Seeded random real c: not an artifact of dropping the scale factors.
    print("\n  D1b. Seeded random REAL c (so this is NOT an artifact of dropping tau, hbar):")
    triv_all = True
    for cv in rng.uniform(-5, 5, size=5):
        cval = sp.Rational(float(np.round(cv, 6))).limit_denominator(10**6)
        eqc = [e.subs(c, cval) for e in eqs]
        solc = sp.solve(eqc, [ux, uy, vx, vy], dict=True)
        triv = (len(solc) == 1 and all(sp.simplify(v_) == 0 for v_ in solc[0].values()))
        triv_all &= triv
        print(f"    c = {float(cval):>9.6f}  (equiv. hbar/tau = {-float(cval):>9.6f})  ->  "
              f"u_x = u_y = 0 forced: {ok(triv)}")
    print(f"\n  D1b: every real c forces the constant solution:         {ok(triv_all)}")

    print("\n  >> The manuscript's OWN closure condition trivializes the holomorphy claim.")
    print("     If u = v (or v = c u for any real c) AND Cauchy-Riemann holds, then")
    print("     u_x = u_y = 0: u is CONSTANT, varphi is constant, and J = A e^varphi +")
    print("     B e^{i varphi} is a CONSTANT. No interference, no dynamics, no Born rule.")
    print("     Holomorphy and def:J-closed are mutually exclusive except for constant fields.")
    print(f"\n     The only roots of 1 + c^2 are {all_roots}: the ratio theta_I/theta_R would have")
    print("     to be IMAGINARY. That is not a rescue -- it is exactly the Wick rotation M2")
    print("     found at l.2295, and it abandons the real harmonic-conjugate structure the")
    print("     whole claim rests on. The two negatives are the same negative twice.")
    return no_real_escape and only_trivial and triv1 and triv_all


# ---------------------------------------------------------------------------
# D2. The weaker "closure holds on a set" reading does not rescue it.
# ---------------------------------------------------------------------------
def d2_closure_set_is_a_curve():
    banner("D2. NEGATIVE: reading closure as 'holds where it holds' gives a measure-zero curve")
    print("  If def:J-closed is read not as an identity but as holding only where")
    print("  theta_R = theta_I, then for psi = z (tau = 2, hbar = 1):")
    print("      theta_R = ln r      theta_I = -2 theta")
    print("  and closure is ln r = -2 theta, i.e. the log spiral r = exp(-2 theta).")
    print("  A curve, not an open set. Measuring that directly on a grid:\n")

    g = np.linspace(-2, 2, 601)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    TH = np.arctan2(Y, X)
    mask = R > 1e-6
    resid = np.where(mask, np.log(np.where(mask, R, 1.0)) + 2 * TH, np.nan)

    print(f"  {'tolerance':>12}{'fraction of grid points with |theta_R - theta_I| < tol':>56}")
    fracs = []
    tols = [1e-1, 1e-2, 1e-3]
    for t in tols:
        f = float(np.nanmean(np.abs(resid) < t))
        fracs.append(f)
        print(f"  {t:>12.0e}{f:>56.6f}")
    ratios = [fracs[i] / fracs[i + 1] for i in range(len(fracs) - 1)]
    print(f"\n  successive ratios as tol shrinks 10x: "
          f"{', '.join(f'{r:.2f}' for r in ratios)}   (a CURVE gives ~10, an OPEN SET ~1)")
    is_curve = all(4.0 < r < 25.0 for r in ratios)
    print(f"  the closure set scales like a 1-D curve, not a 2-D region:  {ok(is_curve)}")

    # On the spiral itself closure does hold, exactly.
    th = np.linspace(-1.0, 1.0, 5)
    r_sp = np.exp(-2 * th)
    resid_sp = np.log(r_sp) + 2 * th
    on_curve = np.allclose(resid_sp, 0.0, atol=TOL)
    print(f"\n  On the spiral r = exp(-2 theta), closure holds exactly:     {ok(on_curve)}"
          f"   max|resid|={np.abs(resid_sp).max():.2e}")
    print("  >> So the weaker reading is not a rescue either. varphi(x,t) is defined by")
    print("     def:J-closed as a FIELD on (x,t); on a measure-zero spiral it is not a field,")
    print("     and eq:J is not a field equation. Either closure holds on an open set -- and")
    print("     D1 makes the field constant -- or it holds on a curve and eq:J does not")
    print("     define varphi(x,t) at all.")
    return is_curve and on_curve


def main():
    h1 = h1_harmonic_conjugates()
    h2 = h2_interference()
    m1 = m1_manuscript_audit()
    m2 = m2_wick_rotation()
    m3 = m3_side_by_side()
    d1 = d1_closure_trivializes()
    d2 = d2_closure_set_is_a_curve()

    banner("SUMMARY: the claim is MIXED, and it splits cleanly")
    print(f"  POSITIVE (H1-H2): as MATHEMATICS the holomorphic-dual proposal holds: {ok(h1 and h2)}")
    print("    f = -2 log(psi) is holomorphic; Re f = H^D (thermal), Im f = -2 arg(psi)")
    print("    (phase). H^D is harmonic, its dual IS the wavefunction phase, and under that")
    print("    reading |W|^2 varies with a true null at x = 0. It is a good structure and it")
    print("    would kill the frozen-action objection. The author is right about the math.")
    print(f"\n  NEGATIVE (M1-M3): it is NOT what the manuscript does:                  {ok(m1 and m2 and m3)}")
    print("    M1  No Cauchy-Riemann, analyticity, or Hilbert-transform machinery appears")
    print("        anywhere; H^D and S^D are introduced as unrelated drivers.")
    print("    M2  l.2295 forces S^nu = -2i hbar_eff log|psi| / tau: a Wick rotation. An")
    print("        imaginary S is not the harmonic conjugate of H^D, it is i times it.")
    print("    M3  The two readings are mutually exclusive, and Steps 3-4 need the one")
    print("        without a phase.")
    print(f"\n  DECISIVE (D1-D2): the manuscript's own closure kills the claim:        {ok(d1 and d2)}")
    print("    D1  def:J-closed sets theta_R = theta_I. With Cauchy-Riemann this gives")
    print("        (1 + c^2) u_x = 0, and 1 + c^2 has no real root, so u_x = u_y = 0: the")
    print("        field is CONSTANT and J is constant. Holomorphy and def:J-closed are")
    print("        mutually exclusive except for constant fields.")
    print("    D2  Weakening closure to a level set gives a measure-zero spiral, on which")
    print("        varphi(x,t) is not a field and eq:J is not a field equation.")
    print("\n  VERDICT: the proposed structure is sound mathematics and is NOT the manuscript's")
    print("  structure. Adopting it would require stating analyticity of psi as a hypothesis,")
    print("  deleting the Wick rotation at l.2295, and REPLACING def:J-closed, which as")
    print("  written forces the constant field. It cannot be presented as what the manuscript")
    print("  already meant. The gap is a rewrite, not a reading.")

    all_ok = all([h1, h2, m1, m2, m3, d1, d2])
    print(f"\n  All checks in this script behaved as documented:            {ok(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
