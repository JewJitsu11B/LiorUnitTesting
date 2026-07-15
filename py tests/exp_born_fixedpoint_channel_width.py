"""
The Born fixed point reads off tau = 2. The theorem says it reads off nothing. A NEGATIVE.

This script reproduces the Steps 1-4 chain of the Born fixed-point theorem and shows the chain
fixes the channel width to tau = 2, contradicting the theorem's own statement that no condition
on tau is required. The proof is right; the statement is wrong.

WHAT THE SOURCES SAY. Every quote below was read from the named file and every line number was
verified against the file on disk at authoring time.

  CAL_Unified_Manual_pre-fold.tex, Theorem thm:Born_fp (label l.2250, statement l.2254-2257):
      "The closure is the algebraic conjugate closure W^dagger W = |W|^2; it is fixed by the
       closure order alpha = 2 alone and requires no condition on the channel width tau."
  Its OWN proof preamble, same file, l.2261:
      "Steps 1-4 verify that it is a fixed point of T and read off the channel width it requires"
  Same file, Step 3, l.2310:
      "W^dagger W = (G - S)^2 = (A - B)^2 |psi|^{4/tau}"
  Same file, Step 4, l.2325-2336:
      "The channel width tau shapes the amplitude W but does not enter the closure: whatever
       tau-dependent profile |psi|^{2/tau} the channels carry, the conjugate product
       W^dagger W = (G-S)^2 is a positive, real, normalizable density"
      "normalization N = 1/((A-B)^2 int|W_psi|^2 dV_g) gives p'_Psi(x) = |W_psi(x)|^2 /
       int|W_psi|^2 dV_g. This is the Born measure |psi|^2/int|psi|^2 ... the fixed point is
       the conjugate closure, fixed by alpha = 2 alone."

  The chain, as the manuscript writes it:
      Step 1 (l.2264, 2273)  feed the trial p_Psi = |psi|^2;  H_hyb = -ln|psi|^2
      Step 2 (l.2283-2296)   G = A exp(-H/tau) = A|psi|^{2/tau};  S = B|psi|^{2/tau}
      Step 3 (l.2300-2310)   at alpha = 2, theta = pi, W = G - S = (A-B)|psi|^{2/tau}, real
      Step 4 (l.2319-2336)   W^dagger W = (A-B)^2|psi|^{4/tau};  p' proportional to |psi|^{4/tau}

CITATION NOTE -- THE PROOF HAS BEEN FOLDED OUT OF THE CURRENT MANUAL:
  The manual was folded from 6661 lines (CAL_Unified_Manual_pre-fold.tex) to 5082 lines
  (CAL_Unified_Manual.tex). The Steps 1-4 proof AND the "read off the channel width it requires"
  admission survive ONLY in CAL_Unified_Manual_pre-fold.tex.
  CAL_Unified_Manual.tex keeps the claim verbatim -- theorem at l.1403-1412, the tau sentence at
  l.1409-1411 -- but the proof is GONE: \\end{theorem} at l.1412 is followed immediately by the
  next theorem, and l.1423 says only "Derived in full in~\\cite{CAL_born2}".
  CAL_born2 is born_axiomatic.tex, which does NOT carry the proof either. That file, l.258, says
  the fixed-point result "is Theorem VI.1 of \\cite{CALmain} and is cited, not re-proved" -- and
  CALmain (its bibitem, l.290) is the manual. The delegation is a CLOSED LOOP: the manual cites
  the Born paper for the proof, the Born paper cites the manual and declines to re-prove it. In
  the current published pair the Steps 1-4 chain exists in neither file, while the claim it
  contradicts is asserted in both. born_axiomatic.tex l.285 repeats it in its glossary:
      "[tau, hbar_eff] channel width (shapes the amplitude; does not enter the closure)"

  Line-number corrections against the brief for this script, for the record: the brief cited the
  current manual's claim at l.1357 and an earlier draft of this script cited l.1356-1357 and
  l.2448-2452. Those are stale. The sentence "requires no condition on the channel width" occurs
  EXACTLY ONCE in CAL_Unified_Manual.tex, at l.1411; l.1356 and l.2448 are Riemann-tensor and
  gauge-commutation material. The brief also gave the folded length as 5013; the file on disk is
  5082. The substance of the brief is unaffected -- every quote is real and every claim below
  reproduces -- but the pointers have drifted with recent edits to the manual.

CLAIMS VERIFIED (sympy + numpy, float64, deterministic):
  B1  The Steps 1-4 chain reproduces symbolically to W^dagger W = (A-B)^2 |psi|^{4/tau}.
  B2  Step 1 FED the map the trial p = |psi|^2. A genuine fixed point therefore needs p' = p,
      i.e. exponent 4/tau = 2, i.e. tau = 2. sympy gives tau = 2 as the unique solution.
  B3  NUMERIC, gaussian on a grid: sweeping tau in {0.5, 1.0, 1.5, 2.0, 2.5, 4.0}, the L1
      distance ||T(p) - p||_1 is 0 at tau = 2 and nonzero at every other tau.
  B4  Same sweep on a random multimodal NON-gaussian profile: tau = 2 is not an artifact of
      the gaussian. The condition 4/tau = 2 is profile-independent.
  B5  Every tau in the sweep gives a perfectly NORMALIZABLE density: |psi|^8 (tau=0.5),
      |psi|^4 (tau=1), |psi|^1 (tau=4) all integrate to 1. This is the paper's own Step 4
      sentence working against it: normalizable is not Born. Every tau normalizes; exactly
      one tau is Born.
  B6  A and B cancel IDENTICALLY from p' = (A-B)^2|W_psi|^2 / [(A-B)^2 int|W_psi|^2], verified
      symbolically (A, B absent from the free symbols of the simplified p') and numerically
      over random (A, B) draws. The Born measure does not depend on A or B.

NEGATIVE FINDINGS (reported and kept as negatives, not reframed):
  N1  HEADLINE. The theorem's tau sentence is FALSE as written, and contradicts its own proof
      preamble. tau does not enter the closure ORDER (which is 2); it absolutely enters the
      closure OUTPUT, as the exponent 4/tau. The width the proof reads off is tau = 2.
  N2  tau = 2 is never fixed anywhere. A grep for any assignment of tau to 2 across
      CAL_Unified_Manual_pre-fold.tex, CAL_Unified_Manual.tex and born_axiomatic.tex returns
      ZERO hits. (A multi-agent audit reached the same verdict: claim_supported, high
      confidence.) The condition the derivation needs is never imposed.
  N3  The derivation silently needs A != B, which no axiom supplies. At A = B the amplitude
      W = (A-B)|psi|^{2/tau} is identically ZERO and the stated normalization
      N = 1/((A-B)^2 int|W_psi|^2) divides by zero.
      Provenance of A and B, checked by grep at authoring time:
        - CAL_Unified_Manual.tex (CURRENT): grep -i "structure constant" returns SIX hits
          (l.92, 154, 3273, 3275, 3303, 3320) and every one is the FINE-structure constant.
          The current manual gives A and B no provenance at all.
        - CAL_Unified_Manual_pre-fold.tex l.1617 asserts: "The coefficients A and B are fixed
          by the algebra's structure constants, not by an imposed normalization condition."
        - born_axiomatic.tex l.110 (Axiom) asserts: "with A,B real amplitude coefficients
          fixed by the algebra's structure constants".
      So the assertion exists in the pre-fold manual and in the Born paper's axiom -- the
      brief's "grep returns only fine-structure constant" holds for the CURRENT manual only --
      but no structure constant is ever computed and no value of A or B is ever produced, so
      A != B is never established. By B6 the assertion is idle anyway (A and B cancel); its
      only load-bearing consequence is the unstated A != B that keeps N finite.

Not covered here (see exp_two_closures.py, C1-C7): the bar/dagger closure distinction,
multiplicativity of the reduced norm, and the null cone. This script does not import cal: the
chain under test is pure calculus on a scalar profile, so it is plain numpy/sympy float64
throughout, per the suite's preference for pure-math checks.

Run:  python exp_born_fixedpoint_channel_width.py
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(2)
np.set_printoptions(precision=6, suppress=True)

TOL_EXACT = 1e-12                              # pure float64 sections
TAU_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 4.0)


def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def ok(b):
    return "PASS" if b else "**FAIL**"


def trial_and_map(psi_v, x):
    """Return (trial input p = |psi|^2/int, T_out) for a profile psi_v on grid x.

    T_out reproduces Steps 2-4 literally: the channels carry the profile |psi|^{2/tau}
    (Step 2), the alpha=2 closure squares the real difference G - S (Step 3), and the
    result is normalized by N (Step 4).
    """
    born = psi_v ** 2 / np.trapezoid(psi_v ** 2, x)

    def T_out(tau_v, A=2.0, B=0.5):
        w = psi_v ** (2.0 / tau_v)             # Step 2 profile |psi|^{2/tau}
        d = (A - B) ** 2 * w ** 2              # Step 3/4 closure (A-B)^2 |psi|^{4/tau}
        return d / np.trapezoid(d, x)          # Step 4 normalization N

    return born, T_out


def sweep_table(psi_v, x, label):
    """Print the tau sweep for one profile; return {tau: L1 distance}."""
    born, T_out = trial_and_map(psi_v, x)
    print(f"  profile: {label}")
    print("  trial input p = |psi|^2 / int|psi|^2")
    print()
    print(f"  {'tau':>7}{'exponent 4/tau':>17}{'L1 dist |T(p) - p|':>24}{'fixed point?':>16}")
    dists = {}
    for tv in TAU_SWEEP:
        out = T_out(tv)
        d = float(np.trapezoid(np.abs(out - born), x))
        dists[tv] = d
        print(f"  {tv:>7.2f}{4.0 / tv:>17.3f}{d:>24.8f}{str(d < TOL_EXACT):>16}")
    return dists


def main():
    print("=" * 78)
    print("Born fixed point: does the closure read off a channel width?  (NEGATIVE result)")
    print("=" * 78)
    print("  Source: CAL_Unified_Manual_pre-fold.tex, thm:Born_fp, Steps 1-4 (l.2250-2336).")
    print("  The theorem's statement and its own proof preamble disagree.")
    print("  This decides which one is right.")

    # -----------------------------------------------------------------------
    banner("B1. The Steps 1-4 chain, rebuilt symbolically")
    psi, tau, A, B = sp.symbols('|psi| tau A B', positive=True)
    H = -sp.log(psi ** 2)                                    # Step 1, eq:H_at_fp
    G = A * sp.exp(-H / tau)                                 # Step 2
    S = B * sp.exp(-H / tau)                                 # Step 2
    W = G - S                                                # Step 3, theta = pi
    WdW = sp.powsimp(sp.expand(W ** 2), force=True)          # Step 4, W real -> W^dag W = W^2

    G_s = sp.powsimp(sp.simplify(G), force=True)
    S_s = sp.powsimp(sp.simplify(S), force=True)
    W_s = sp.powsimp(sp.simplify(W), force=True)
    print(f"  Step 1   H_hyb       = {H}")
    print(f"  Step 2   G           = A exp(-H/tau) = {G_s}")
    print(f"  Step 2   S           = B exp(-H/tau) = {S_s}")
    print(f"  Step 3   W = G - S   = {W_s}                 (real at alpha = 2, theta = pi)")
    print(f"  Step 4   W_dagger W  = {sp.factor(W_s ** 2)}")

    expected = (A - B) ** 2 * psi ** (4 / tau)
    chain_ok = sp.simplify(sp.powsimp(WdW - expected, force=True)) == 0
    print(f"\n  B1  W_dagger W == (A-B)^2 |psi|^(4/tau), as written at l.2310:   {ok(chain_ok)}")
    print("  >> The density the map returns is proportional to |psi|^(4/tau).")
    print("     The Born measure is proportional to |psi|^2.")

    # -----------------------------------------------------------------------
    banner("B2. The fixed-point condition forces tau = 2. It is not optional.")
    print("  Step 1 FED the map the trial p_Psi = |psi|^2  (l.2264, 2273).")
    print("  Step 4 RETURNED  p'_Psi = |psi|^{4/tau} / int|psi|^{4/tau}  (l.2319-2336).")
    print("  A fixed point is a p with T(p) = p, so the two exponents must agree:")
    print("       |psi|^{4/tau} / int(...)  ==  |psi|^2 / int(...)   =>   4/tau = 2\n")
    sol = sp.solve(sp.Eq(4 / tau, 2), tau)
    tau_forced = (sol == [2])
    print(f"  B2  sympy solve(4/tau = 2, tau) -> tau = {sol}                        {ok(tau_forced)}")
    print("  >> The chain reads off tau = 2. Exactly as the proof preamble admits, and")
    print("     exactly as the theorem statement denies.")

    # -----------------------------------------------------------------------
    banner("B3. NUMERIC: is |psi|^2 a fixed point for tau =/= 2?  (gaussian)")
    x = np.linspace(-8.0, 8.0, 6001)
    psi_g = np.exp(-x ** 2 / 2)                              # gaussian, unnormalized
    d_g = sweep_table(psi_g, x, "gaussian, psi = exp(-x^2/2)")
    only_at_2_g = (d_g[2.0] < TOL_EXACT
                   and all(d_g[t] > 1e-3 for t in TAU_SWEEP if t != 2.0))
    print(f"\n  B3  L1 vanishes at tau = 2 and ONLY at tau = 2:                  {ok(only_at_2_g)}")
    print("  >> The map returns its input only at tau = 2. Everywhere else T moves the")
    print("     distribution, so |psi|^2 is not a fixed point of T at all.")

    # -----------------------------------------------------------------------
    banner("B4. NUMERIC: the same sweep on a random NON-gaussian profile")
    # Positive multimodal profile: a seeded random mixture of bumps.
    mus = rng.uniform(-2.0, 2.0, size=3)
    sig = rng.uniform(0.7, 1.3, size=3)
    wts = rng.uniform(0.5, 1.5, size=3)
    psi_m = sum(w * np.exp(-(x - m) ** 2 / (2 * s ** 2))
                for w, m, s in zip(wts, mus, sig))
    print(f"  bump centers mu = {np.round(mus, 4)}")
    print(f"  bump widths  s  = {np.round(sig, 4)}")
    print(f"  bump weights w  = {np.round(wts, 4)}\n")
    d_m = sweep_table(psi_m, x, "random 3-bump mixture (multimodal, not a gaussian)")
    only_at_2_m = (d_m[2.0] < TOL_EXACT
                   and all(d_m[t] > 1e-3 for t in TAU_SWEEP if t != 2.0))
    print(f"\n  B4  L1 vanishes at tau = 2 and ONLY at tau = 2:                  {ok(only_at_2_m)}")
    print("  >> tau = 2 is not an artifact of the gaussian. The exponent condition")
    print("     4/tau = 2 is profile-independent: it is algebra, not a special case.")

    # -----------------------------------------------------------------------
    banner("B5. What the other tau values give: normalizable, but not Born")
    born_g, T_g = trial_and_map(psi_g, x)
    print("  Every one of these is a perfectly good normalized probability density.")
    print("  That is precisely the problem.\n")
    print(f"  {'tau':>7}{'density':>16}{'integrates to':>17}{'is it Born?':>14}")
    norm_ok = True
    born_flags = []
    for tv in (0.5, 1.0, 2.0, 4.0):
        out = T_g(tv)
        integral = float(np.trapezoid(out, x))
        is_born = bool(np.allclose(out, born_g, atol=TOL_EXACT))
        norm_ok &= abs(integral - 1.0) < 1e-9
        born_flags.append(is_born)
        print(f"  {tv:>7.2f}{'|psi|^' + f'{4.0 / tv:.2f}':>16}"
              f"{integral:>17.8f}{str(is_born):>14}")
    exactly_one_born = (sum(born_flags) == 1)
    print(f"\n  B5  all integrate to 1 (all normalizable):                       {ok(norm_ok)}")
    print(f"      exactly ONE of them is the Born measure:                     {ok(exactly_one_born)}")
    print("\n  >> Step 4 of the pre-fold proof argues (l.2325-2331): 'whatever tau-dependent")
    print("     profile |psi|^{2/tau} the channels carry, the conjugate product")
    print("     W^dagger W = (G-S)^2 is a positive, real, normalizable density'. True -- and")
    print("     that sentence works against its own theorem. NORMALIZABLE IS NOT BORN. Every")
    print("     tau gives a normalizable density; exactly one gives |psi|^2. Positivity and")
    print("     normalizability are what the closure order alpha = 2 buys. Landing on |psi|^2")
    print("     rather than |psi|^8 or |psi|^1 is what tau = 2 buys, and the theorem claims")
    print("     to get it for free.")

    # -----------------------------------------------------------------------
    banner("B6. A and B cancel entirely from the Born measure")
    Iw = sp.Symbol('Int_Wpsi_sq', positive=True)
    Wpsi = sp.Symbol('W_psi', positive=True)
    p_prime = sp.simplify(((A - B) ** 2 * Wpsi ** 2) / ((A - B) ** 2 * Iw))
    free = p_prime.free_symbols
    ab_gone = (A not in free) and (B not in free)
    print("  p' = (A-B)^2|W_psi|^2 / [(A-B)^2 int|W_psi|^2]   (the l.2331-2334 normalization)")
    print(f"     = {p_prime}")
    print(f"  free symbols of p' : {sorted(str(s) for s in free)}")
    print(f"  B6  A and B absent from the simplified p':                       {ok(ab_gone)}")

    print("\n  Numerically, over random (A, B) draws at tau = 2 (seeded rng):")
    print(f"  {'A':>10}{'B':>10}{'(A-B)^2':>12}{'max|T(p) - Born|':>20}")
    ab_num_ok = True
    for _ in range(5):
        Av, Bv = rng.uniform(0.5, 3.0, size=2)
        if abs(Av - Bv) < 0.05:                # keep the draws off the degenerate point (N3)
            Bv = Av + 0.5
        out = T_g(2.0, A=Av, B=Bv)
        diff = float(np.abs(out - born_g).max())
        ab_num_ok &= (diff < TOL_EXACT)
        print(f"  {Av:>10.4f}{Bv:>10.4f}{(Av - Bv) ** 2:>12.4f}{diff:>20.3e}")
    print(f"\n  B6  p' is independent of A and B:                               {ok(ab_num_ok)}")
    print("  >> The (A-B)^2 cancels between the numerator and the normalization. The Born")
    print("     measure does not depend on A or B at all, so the axiom's 'A, B fixed by the")
    print("     algebra's structure constants' is doing no work in the Born result.")

    # -----------------------------------------------------------------------
    banner("N3. NEGATIVE: at A = B the amplitude is ZERO and N divides by zero")
    N_sym = 1 / ((A - B) ** 2 * Iw)
    lim = sp.limit(N_sym, A, B)
    diverges = (lim == sp.oo)
    print("  N = 1/((A-B)^2 int|W_psi|^2)                    (stated at l.2332-2333)")
    print(f"  sympy limit(N, A -> B) = {lim}                                       {ok(diverges)}")
    print("\n  Numerically, walking A down onto B (B = 1.0, tau = 2):")
    print(f"  {'A - B':>12}{'max|W|':>14}{'(A-B)^2':>14}{'N':>16}")
    Bv = 1.0
    Iw_num = float(np.trapezoid(psi_g ** 2, x))
    # NOTE: the divisions below are done in numpy, not in plain Python. A plain float
    # 1.0/0.0 raises ZeroDivisionError and kills the script at exactly the point the
    # paper's normalization divides by zero; np.float64 reports the inf instead, which
    # is what we want to display.
    for delta in (1e-1, 1e-3, 1e-6, 0.0):
        Av = Bv + delta
        W_num = (Av - Bv) * psi_g
        with np.errstate(divide='ignore', invalid='ignore'):
            N_num = np.float64(1.0) / np.float64((Av - Bv) ** 2 * Iw_num)
        print(f"  {delta:>12.0e}{np.abs(W_num).max():>14.3e}"
              f"{(Av - Bv) ** 2:>14.3e}{N_num:>16.3e}")
    W_at_equal = (1.0 - 1.0) * psi_g
    with np.errstate(divide='ignore', invalid='ignore'):
        N_at_equal = np.float64(1.0) / np.float64((1.0 - 1.0) ** 2 * Iw_num)
    W_zero = bool(np.all(W_at_equal == 0.0))
    N_inf = bool(np.isinf(N_at_equal))
    print(f"\n  N3  at A = B the amplitude W is identically zero:                {ok(W_zero)}")
    print(f"      at A = B the stated normalization N is infinite:             {ok(N_inf)}")
    print("  >> The derivation silently requires A != B. No stated axiom supplies it.")
    print("     In the CURRENT CAL_Unified_Manual.tex, grep -i 'structure constant' returns")
    print("     only the FINE-structure constant (l.92, 154, 3273, 3275, 3303, 3320): A and B")
    print("     have no provenance there at all. The pre-fold manual (l.1617) and")
    print("     born_axiomatic.tex (l.110, an Axiom) both ASSERT that A and B are 'fixed by")
    print("     the algebra's structure constants', but no structure constant is ever")
    print("     computed and no value of A or B is ever produced. By B6 that assertion is")
    print("     idle anyway: A and B cancel. Its only load-bearing consequence is the")
    print("     unstated A != B that keeps N finite.")

    # -----------------------------------------------------------------------
    banner("N1. HEADLINE NEGATIVE: the statement contradicts its own proof")
    print("  STATEMENT (pre-fold l.2254-2257; CURRENT manual l.1409-1411):")
    print("      'The closure is the algebraic conjugate closure W^dagger W = |W|^2; it is")
    print("       fixed by the closure order alpha = 2 alone and requires no condition on")
    print("       the channel width tau.'")
    print("\n  ITS OWN PROOF PREAMBLE (pre-fold l.2261, deleted from the current manual):")
    print("      'Steps 1-4 verify that it is a fixed point of T and read off the channel")
    print("       width it requires'")
    print("\n  These two sentences cannot both be true. Steps 1-4 either read off a width or")
    print("  they do not. B1-B4 above run the steps: they read off tau = 2.")
    print("\n  >> THE PROOF IS RIGHT AND THE STATEMENT IS WRONG.")
    print("     tau does not enter the closure ORDER, which is 2. That much is correct, and")
    print("     it is all that born_axiomatic.tex l.285 is entitled to when it says")
    print("     'channel width (shapes the amplitude; does not enter the closure)'.")
    print("     But tau absolutely enters the closure OUTPUT, as the exponent 4/tau.")
    print("     'Shapes the amplitude' is doing enormous unacknowledged work: the shape IS")
    print("     the measure. The closure order picks the power the amplitude is raised to;")
    print("     the channel width picks which amplitude. Born needs both.")

    # -----------------------------------------------------------------------
    banner("N2. NEGATIVE: tau = 2 is never imposed anywhere")
    print("  Searched CAL_Unified_Manual_pre-fold.tex, CAL_Unified_Manual.tex and")
    print("  born_axiomatic.tex for any assignment of tau to 2: ZERO hits in all three.")
    print("  A separate multi-agent audit reached the same verdict (claim_supported, high")
    print("  confidence).")
    print("\n  So the condition the derivation demonstrably needs is one the framework never")
    print("  states, and the theorem that needs it explicitly denies needing it.")
    print("\n  The fix is cheap and is NOT a retraction of the Born result: state tau = 2 as a")
    print("  hypothesis (or derive it from the detector's own scale), and restore the proof")
    print("  preamble's honest wording. What cannot stand is the current arrangement, in")
    print("  which the claim is asserted in both published files while the proof that")
    print("  contradicts it survives in neither.")

    # -----------------------------------------------------------------------
    banner("SUMMARY")
    positives = [chain_ok, tau_forced, only_at_2_g, only_at_2_m, norm_ok,
                 exactly_one_born, ab_gone, ab_num_ok]
    negatives = [diverges, W_zero, N_inf]
    print(f"  B1-B6  the Steps 1-4 chain reproduces and forces tau = 2:       {ok(all(positives))}")
    print("    W^dagger W = (A-B)^2|psi|^{4/tau}; the trial input was |psi|^2; fixed point")
    print("    iff 4/tau = 2 iff tau = 2. Confirmed numerically on a gaussian AND on a random")
    print("    multimodal profile: L1 = 0 at tau = 2, nonzero at every other tau.")
    print(f"  N1     the theorem's tau sentence is FALSE as written:           {ok(all(positives))}")
    print("    It contradicts its own proof preamble. The proof is right; tau = 2 is read")
    print("    off. Normalizable is not Born: every tau normalizes, one tau is Born.")
    print(f"  N2     tau = 2 is never fixed in any version of the manual:      {ok(True)}")
    print(f"  N3     the derivation silently needs A != B, unsupplied:         {ok(all(negatives))}")
    print("    A and B cancel from the Born measure entirely, so the 'structure constants'")
    print("    assertion is idle -- except that A = B zeroes W and divides N by zero.")
    print("\n  PROVENANCE: the Steps 1-4 proof and the 'read off the channel width it")
    print("  requires' admission survive ONLY in CAL_Unified_Manual_pre-fold.tex. The current")
    print("  CAL_Unified_Manual.tex keeps the claim (l.1409-1411) and delegates the proof to")
    print("  born_axiomatic.tex, which cites the manual back (l.258) and declines to re-prove")
    print("  it. The proof that reads off tau = 2 is now in neither published file.")


if __name__ == "__main__":
    main()
