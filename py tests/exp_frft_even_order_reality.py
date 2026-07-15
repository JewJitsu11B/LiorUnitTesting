"""
Even-order reality of the fractional Fourier transform: what it does and does not buy.

The manual (CAL_Unified_Manual.tex, lines 1200-1204) identifies the rotational content of
the fractional integral with the fractional Fourier transform
    F^alpha = exp(-i (alpha pi / 2) N),        N = the number operator,
whose four canonical corners are identity (alpha=0), Fourier (alpha=1), parity (alpha=2),
inverse Fourier (alpha=3). Built here on the real, orthonormal Hermite eigenbasis as
    F^alpha = sum_n exp(-i alpha pi n / 2) |h_n><h_n|.

The claim under test is MIXED. One half reproduces exactly; the other half is refuted.

CLAIMS VERIFIED (confirmed half):
  R0: basis sanity. The Hermite functions are orthonormal on the grid, and F^alpha
      reconstructs its input at alpha=0 (truncation is not doing the work below).
  R1: the mode multiplier exp(-i alpha pi n / 2) is real for EVERY mode n exactly when
      alpha is EVEN, where it is (+/-1)^n. At odd alpha it is (-/+ i)^n, complex.
  R2: consequently F^alpha maps REAL functions to REAL functions iff alpha is even.
      Swept over alpha = 0,1,2,3,4 on a seeded real test function, printing max|Im|.
  R3: the corners are the stated ones. F^2 f(x) = f(-x) (parity) to precision, F^1
      agrees with the direct quadrature Fourier transform, F^4 = identity.

CLAIMS REFUTED (the gap):
  R4: even alpha only PRESERVES reality, it does not CREATE it. A complex action channel
      B exp(i S(x)) with a genuinely varying action S comes out COMPLEX at EVERY alpha,
      the even ones included. F^2 sends B e^{iS(x)} to B e^{iS(-x)}: parity-flipped, and
      still complex. Verified as an identity, not just as a nonzero max|Im|.
  R5: the mechanism. At even alpha the multiplier is real, so F^alpha is a REAL-LINEAR
      operator: it commutes with Re and Im and therefore cannot move anything between
      them. Checked directly: F^2(Re f) = Re(F^2 f) and F^2(Im f) = Im(F^2 f).
      Reality at alpha=2 is INHERITED from a real input, never produced.
  R6: the consequence for Axiom 2. born_axiomatic.tex Axiom 2 calls the action channel
      "an imaginary unitary (action) channel", B exp(i S / hbar_eff). Its input is
      complex by construction, so even alpha does nothing for it: the thermal channel
      A exp(-H / tau) goes in real and comes out real, the action channel goes in complex
      and comes out complex. A real S does not help; e^{i * real} is a phase.

Pure-math check: numpy float64 throughout, no torch, tolerances at 1e-8.
Seeded and deterministic: np.random.default_rng(11).
"""
import numpy as np
from numpy.polynomial.hermite import hermval
from math import factorial, pi, sqrt

RNG = np.random.default_rng(11)

# ---- Hermite eigenbasis on a grid ------------------------------------------
# NMAX modes need a grid past the classical turning point sqrt(2*NMAX + 1) ~ 11.0,
# so [-14, 14] contains the whole basis and the quadrature is spectrally accurate.
NMAX = 60
x = np.linspace(-14.0, 14.0, 4001)
dx = x[1] - x[0]

TOL = 1e-8
CORNERS = {0: "identity", 1: "Fourier", 2: "parity", 3: "inverse Fourier", 4: "identity"}


def hermite_fn(n, xs):
    """The n-th Hermite function h_n, real and L2-normalised."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    norm = 1.0 / sqrt(2.0**n * factorial(n) * sqrt(pi))
    return norm * hermval(xs, c) * np.exp(-xs**2 / 2)


HBASIS = np.array([hermite_fn(n, x) for n in range(NMAX)])   # real, orthonormal
MODES = np.arange(NMAX)


def frft(f, alpha):
    """F^alpha f = sum_n exp(-i alpha pi n / 2) <h_n|f> h_n."""
    coef = (HBASIS @ f) * dx
    mult = np.exp(-1j * alpha * pi * MODES / 2)
    return (mult * coef) @ HBASIS


def max_im(v):
    return float(np.max(np.abs(np.imag(v))))


def is_real(v, tol=1e-6):
    scale = max(1.0, float(np.max(np.abs(v))))
    return max_im(v) < tol * scale


def ok(b):
    return "PASS" if b else "**FAIL**"


def banner(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    # Seeded inputs. The real test function is a random combination of Hermite modes,
    # so it is exactly real and exactly inside the truncated basis: the reality result
    # below cannot be an artifact of a hand-picked function or of truncation error.
    coef_real = RNG.normal(size=24) / (1.0 + MODES[:24])
    f_real = coef_real @ HBASIS[:24]

    # The action channel B exp(i S(x)): a real amplitude envelope B carrying a genuinely
    # varying action S(x) (linear drift plus a chirp plus a seeded cubic wiggle).
    B = 0.7 * np.exp(-x**2 / 4.0)
    s1, s2, s3 = RNG.uniform(0.8, 1.4), RNG.uniform(0.2, 0.4), RNG.uniform(0.01, 0.03)
    S_of_x = s1 * x + s2 * x**2 + s3 * x**3
    chan = B * np.exp(1j * S_of_x)

    # ---- R0 ----------------------------------------------------------------
    banner("R0. Basis sanity: orthonormality and the alpha=0 reconstruction")
    gram = (HBASIS @ HBASIS.T) * dx
    gram_err = float(np.max(np.abs(gram - np.eye(NMAX))))
    rec_real = float(np.max(np.abs(frft(f_real, 0) - f_real)))
    rec_chan = float(np.max(np.abs(frft(chan, 0) - chan)))
    print(f"  grid: {len(x)} points on [{x[0]:.0f}, {x[-1]:.0f}], dx = {dx:.5f}, "
          f"NMAX = {NMAX} Hermite modes")
    print(f"  max |<h_m|h_n> - delta_mn|            = {gram_err:.3e}   {ok(gram_err < 1e-8)}")
    print(f"  max |F^0 f_real - f_real|             = {rec_real:.3e}   {ok(rec_real < 1e-8)}")
    print(f"  max |F^0 chan - chan|                 = {rec_chan:.3e}   {ok(rec_chan < 1e-6)}")
    print("  >> Both inputs are represented by the truncated basis. Truncation is not")
    print("     driving any result below.")

    # ---- R1 ----------------------------------------------------------------
    print()
    banner("R1. CONFIRMED: the multiplier exp(-i alpha pi n / 2) is real iff alpha is even")
    header = f"  {'alpha':>7}  " + "".join(f"{'n=' + str(n):>13}" for n in range(4))
    print("  multiplier on the n-th Hermite mode:")
    print()
    print(header + "    all real?")
    r1_ok = True
    for a in (0, 1, 2, 3, 4):
        m = np.exp(-1j * a * pi * np.arange(4) / 2)
        allr = bool(np.all(np.abs(m.imag) < 1e-12))
        row = "".join(f"{np.round(v, 3)!s:>13}" for v in m)
        print(f"  {a:>7}  " + row + f"    {allr}")
        r1_ok = r1_ok and (allr == (a % 2 == 0))
    print()
    print(f"  real-for-every-mode <=> alpha even, over alpha = 0..4:   {ok(r1_ok)}")
    print("  >> Even alpha: the multiplier is (+/-1)^n, REAL for every mode.")
    print("     Odd alpha:  the multiplier is (-/+ i)^n, complex. As claimed.")

    # ---- R2 ----------------------------------------------------------------
    print()
    banner("R2. CONFIRMED: F^alpha maps REAL to REAL iff alpha is even")
    print(f"  input: a seeded real function (24 Hermite modes). Real? {is_real(f_real)}")
    print()
    print(f"  {'alpha':>7}{'F^alpha f real?':>18}{'max |Im|':>14}   corner")
    r2_ok = True
    for a in (0, 1, 2, 3, 4):
        g = frft(f_real, a)
        real_out = is_real(g)
        print(f"  {a:>7}{str(real_out):>18}{max_im(g):>14.3e}   {CORNERS[a]}")
        r2_ok = r2_ok and (real_out == (a % 2 == 0))
    print()
    print(f"  real output <=> alpha even:                             {ok(r2_ok)}")

    # ---- R3 ----------------------------------------------------------------
    print()
    banner("R3. CONFIRMED: the corners are identity / Fourier / parity / inverse Fourier")
    g2 = frft(f_real, 2)
    par_err = float(np.max(np.abs(g2 - f_real[::-1])))
    g4 = frft(f_real, 4)
    id_err = float(np.max(np.abs(g4 - f_real)))
    # direct quadrature Fourier transform, kernel exp(-i x u) / sqrt(2 pi)
    ft_direct = (np.exp(-1j * np.outer(x, x)) @ f_real) * dx / sqrt(2 * pi)
    g1 = frft(f_real, 1)
    ft_err = float(np.max(np.abs(g1 - ft_direct)))
    g3 = frft(f_real, 3)
    ift_err = float(np.max(np.abs(g3 - np.conj(ft_direct))))
    print(f"  alpha=2 is parity:    max |F^2 f(x) - f(-x)|       = {par_err:.3e}   "
          f"{ok(par_err < TOL)}")
    print(f"  alpha=1 is Fourier:   max |F^1 f - FT[f]|          = {ft_err:.3e}   "
          f"{ok(ft_err < 1e-6)}")
    print(f"  alpha=3 is inv-Four:  max |F^3 f - conj(FT[f])|    = {ift_err:.3e}   "
          f"{ok(ift_err < 1e-6)}")
    print(f"  alpha=4 is identity:  max |F^4 f - f|              = {id_err:.3e}   "
          f"{ok(id_err < TOL)}")
    print("  (FT[f] is the direct quadrature transform with kernel exp(-i x u)/sqrt(2 pi);")
    print("   f is real, so the inverse Fourier corner is its conjugate.)")
    print("  >> At alpha = 2 the transform IS parity, and parity of a real function is real.")

    # ---- R4 ----------------------------------------------------------------
    print()
    banner("R4. REFUTED: even alpha PRESERVES reality, it does not CREATE it")
    print("  Axiom 2 of born_axiomatic.tex calls the action channel 'an imaginary unitary")
    print("  (action) channel', B exp(i S / hbar_eff). That input is COMPLEX. Feed it in:")
    print()
    print(f"  S(x) = {s1:.3f} x + {s2:.3f} x^2 + {s3:.4f} x^3   (a genuinely varying action)")
    print(f"  input: B(x) exp(i S(x)), a varying phase. Real? {is_real(chan)}")
    print()
    print(f"  {'alpha':>7}{'F^alpha[chan] real?':>22}{'max |Im|':>14}   corner")
    r4_ok = True
    for a in (0, 1, 2, 3, 4):
        g = frft(chan, a)
        real_out = is_real(g)
        print(f"  {a:>7}{str(real_out):>22}{max_im(g):>14.4f}   {CORNERS[a]}")
        r4_ok = r4_ok and (not real_out)
    print()
    print(f"  complex at EVERY alpha, even ones included:             {ok(r4_ok)}")
    chan_par_err = float(np.max(np.abs(frft(chan, 2) - chan[::-1])))
    print(f"  and F^2 is exactly the parity flip of the channel:")
    print(f"      max |F^2[B e^(iS(x))] - B(-x) e^(i S(-x))| = {chan_par_err:.3e}   "
          f"{ok(chan_par_err < 1e-6)}")
    print("  >> F^2 sends B e^{iS(x)} to B e^{iS(-x)}: parity-flipped, still complex.")
    print("     A real multiplier acting on a complex vector gives a complex vector.")

    # ---- R5 ----------------------------------------------------------------
    print()
    banner("R5. REFUTED, the mechanism: at even alpha F^alpha is REAL-LINEAR")
    print("  A real-linear operator commutes with Re and Im, so it cannot move anything")
    print("  between them. Checking that on the complex channel at alpha = 2:")
    print()
    re_then = frft(np.real(chan).astype(complex), 2)
    then_re = np.real(frft(chan, 2))
    im_then = frft(np.imag(chan).astype(complex), 2)
    then_im = np.imag(frft(chan, 2))
    c_re = float(np.max(np.abs(re_then - then_re)))
    c_im = float(np.max(np.abs(im_then - then_im)))
    print(f"  max |F^2(Re chan) - Re(F^2 chan)|     = {c_re:.3e}   {ok(c_re < 1e-6)}")
    print(f"  max |F^2(Im chan) - Im(F^2 chan)|     = {c_im:.3e}   {ok(c_im < 1e-6)}")
    # the same commutator at an odd alpha, where the multiplier is complex: it must FAIL
    re_then1 = frft(np.real(chan).astype(complex), 1)
    then_re1 = np.real(frft(chan, 1))
    c_re1 = float(np.max(np.abs(re_then1 - then_re1)))
    print(f"  contrast at alpha=1 (complex multiplier, NOT real-linear):")
    print(f"      max |F^1(Re chan) - Re(F^1 chan)| = {c_re1:.3e}   "
          f"(large, as it must be: {ok(c_re1 > 1e-2)})")
    print("  >> At even alpha, Re and Im propagate independently. Whatever imaginary part")
    print("     went in comes back out, rotated within Im. Reality at alpha=2 is INHERITED.")

    # ---- R6 ----------------------------------------------------------------
    print()
    banner("R6. The consequence for Axiom 2: the action channel is complex on input")
    print("  Under the holomorphic duality f = -2 ln(psi): H = Re f = -ln|psi|^2 and")
    print("  S = Im f = -2 arg(psi) are both REAL harmonic functions. The drivers are real.")
    print("  But the CHANNELS built from them are not both real:")
    print("      thermal:  exp(-H / tau)   = |psi|^(2/tau)           REAL")
    print("      action:   exp(i S / hbar) = exp(-2i arg(psi)/hbar)  COMPLEX, modulus 1")
    print()
    psi = np.exp(-0.3 * x**2) * np.exp(1j * 0.9 * x)
    cases = [
        ("thermal channel  A e^(-H/tau)", np.exp(-0.5 * x**2).astype(complex)),
        ("action channel   B e^(iS(x))", chan),
        ("wavefunction     psi = |psi| e^(i arg)", psi),
    ]
    print(f"  {'input':<40}{'real in?':>10}{'F^2 real out?':>16}")
    r6_ok = True
    for label, v in cases:
        g = frft(v, 2)
        rin, rout = is_real(v), is_real(g)
        print(f"  {label:<40}{str(rin):>10}{str(rout):>16}")
        r6_ok = r6_ok and (rin == rout)
    print()
    print(f"  real out <=> real in, at alpha = 2:                     {ok(r6_ok)}")
    print("  >> The thermal channel goes in real and comes out real. The action channel")
    print("     goes in complex and comes out complex. The i in front of S is what makes")
    print("     it complex, and no even alpha removes it: e^(i * real) is a phase.")

    # ---- verdict -----------------------------------------------------------
    print()
    banner("VERDICT")
    confirmed = gram_err < 1e-8 and rec_real < 1e-8 and r1_ok and r2_ok and par_err < TOL
    refuted = r4_ok and c_re < 1e-6 and c_im < 1e-6 and r6_ok
    print(f"  CONFIRMED  the multiplier is real for every mode iff alpha is even, and so")
    print(f"             F^alpha maps real to real iff alpha is even; F^2 is parity.   "
          f"{ok(confirmed)}")
    print(f"  REFUTED    'even order makes complex exponentials real' as a general claim.")
    print(f"             Even alpha is real-LINEAR: it preserves reality, never creates")
    print(f"             it. The Axiom 2 action channel is complex on input and stays")
    print(f"             complex at every alpha, alpha = 2 included.                   "
          f"{ok(refuted)}")
    print()
    print("  The statement that IS true:  F^alpha preserves the reality of a real f, iff")
    print("                              alpha is even.")
    print("  The statement the argument needs: F^2 makes the COMPLEX action channel real.")
    print("  These are different statements, and the second one is false.")


if __name__ == "__main__":
    main()
