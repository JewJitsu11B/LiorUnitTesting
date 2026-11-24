"""Test of the manual's cosmological-constant relation (CAL Unified Manual, Eqs. 78-81):

    rho_Lambda = M_Pl^4 * (7/135) * (t_Pl / t)^alpha,   alpha = 2 - 1/280 + 1/14000 + 1/140000

Arithmetic, the inversion for t, sensitivity to each input, and the input value of rho_Lambda.
Constants: CODATA 2018 / 2022 (they agree to the digits used); Planck 2018 cosmology."""
from fractions import Fraction
import mpmath as mp

mp.mp.dps = 40
PASS = []


def check(label, ok, text):
    PASS.append(bool(ok))
    print(("PASS" if ok else "FAIL"), label, "::", text)


# --- exponent
al_frac = 2 - Fraction(1, 280) + Fraction(1, 14000) + Fraction(1, 140000)
check("Eq 79 arithmetic", al_frac == Fraction(279511, 140000), "alpha = %s = %.10f" % (al_frac, float(al_frac)))
check("Eq 79 pieces", Fraction(1, 280) * Fraction(1, 50) == Fraction(1, 14000) and Fraction(1, 14 * 10 ** 4) == Fraction(1, 140000),
      "(1/280)(1/50) = 1/14000 ; 1/(14 (3+7)^4) = 1/140000")
alpha = mp.mpf(al_frac.numerator) / al_frac.denominator

# --- constants
GeV = mp.mpf("1.602176634e-10")          # J
hbar = mp.mpf("1.054571817e-34")         # J s
c = mp.mpf("299792458")
G = mp.mpf("6.67430e-11")
MPl = mp.sqrt(hbar * c ** 5 / G) / GeV   # GeV (non-reduced)
tPl = mp.sqrt(hbar * G / c ** 5)
Gyr = mp.mpf("3.15576e16")
hbarc = hbar * c / GeV                   # GeV m
print("M_Pl = %s GeV   t_Pl = %s s" % (mp.nstr(MPl, 8), mp.nstr(tPl, 8)))


def age(rho, a, M=MPl, tp=tPl):
    return tp * (M ** 4 * mp.mpf(7) / 135 / rho) ** (1 / a)


# --- the manual's inversion, with the manual's rounded inputs and with full constants
rho_man = mp.mpf("2.888e-47")
t_man = age(rho_man, alpha, mp.mpf("1.221e19"), mp.mpf("5.391e-44"))
t_full = age(rho_man, alpha)
check("Eq 81, manual inputs", abs(t_man / Gyr - mp.mpf("13.79")) < 0.01, "t = %s s = %s Gyr (manual: 4.351e17 s = 13.79 Gyr)" % (mp.nstr(t_man, 5), mp.nstr(t_man / Gyr, 6)))
print("     with unrounded CODATA constants: %s Gyr" % mp.nstr(t_full / Gyr, 6))
t2 = age(rho_man, 2, mp.mpf("1.221e19"), mp.mpf("5.391e-44"))
check("alpha = 2 gives 10.79 Gyr", abs(t2 / Gyr - mp.mpf("10.79")) < 0.01, "t = %s Gyr" % mp.nstr(t2 / Gyr, 6))

# --- sensitivity
X = mp.mpf("1.221e19") ** 4 * mp.mpf(7) / 135 / rho_man
dlnt_dalpha = -mp.log(X) / alpha ** 2
print("\nsensitivity: d ln t / d alpha = %s ; d ln t / d ln rho = %s ; d ln t / d ln M_Pl = %s"
      % (mp.nstr(dlnt_dalpha, 6), mp.nstr(-1 / alpha, 6), mp.nstr(4 / alpha, 6)))
age_obs, age_err = mp.mpf("13.787"), mp.mpf("0.020")
a_exact = mp.findroot(lambda a: age(rho_man, a, mp.mpf("1.221e19"), mp.mpf("5.391e-44")) / Gyr - age_obs, alpha)
print("alpha that returns 13.787 Gyr exactly (manual inputs): %s (manual: 1.9965068)" % mp.nstr(a_exact, 9))
print("the age error +-0.020 Gyr alone corresponds to alpha +- %s" % mp.nstr(age_err / age_obs / abs(dlnt_dalpha), 3))
for name, a in [("-1/280 only", 2 - mp.mpf(1) / 280), ("-1/280 + 1/14000", 2 - mp.mpf(1) / 280 + mp.mpf(1) / 14000), ("all three terms", alpha),
                ("1.9965701 (digits transposed)", mp.mpf("1.9965701"))]:
    tt = age(rho_man, a, mp.mpf("1.221e19"), mp.mpf("5.391e-44")) / Gyr
    print("   alpha = %-12s %-30s t = %s Gyr   (%+.2f sigma of the age error alone)" % (mp.nstr(a, 9), name, mp.nstr(tt, 6), float((tt - age_obs) / age_err)))

# --- the input: rho_Lambda from Planck 2018 base-LCDM (TT,TE,EE+lowE+lensing)
print("\nrho_Lambda from Planck 2018: H0 = 67.36 +- 0.54 km/s/Mpc, Omega_Lambda = 0.6847 +- 0.0073")
Mpc = mp.mpf("3.0856775814913673e22")


def rho_lambda(H0, OL):
    H = H0 * 1000 / Mpc
    rho_c = 3 * H ** 2 / (8 * mp.pi * G)             # kg/m^3
    return OL * rho_c * c ** 2 / GeV * hbarc ** 3    # GeV^4


rP = rho_lambda(mp.mpf("67.36"), mp.mpf("0.6847"))
rel = mp.sqrt((2 * mp.mpf("0.54") / mp.mpf("67.36")) ** 2 + (mp.mpf("0.0073") / mp.mpf("0.6847")) ** 2)
print("   rho_Lambda = %s GeV^4  (+- %s percent)   manual: 2.888e-47 ; ratio manual/Planck = %s"
      % (mp.nstr(rP, 5), mp.nstr(100 * rel, 3), mp.nstr(rho_man / rP, 5)))
tP = age(rP, alpha) / Gyr
print("   age from that rho_Lambda and the algebraic alpha: %s +- %s Gyr   against 13.787 +- 0.020"
      % (mp.nstr(tP, 5), mp.nstr(tP * rel / alpha, 2)))
print("   H0 that would give 2.888e-47 at Omega_Lambda = 0.6847: %s km/s/Mpc"
      % mp.nstr(mp.mpf("67.36") * mp.sqrt(rho_man / rP), 5))
print("   alpha that returns 13.787 Gyr from the Planck rho_Lambda: %s"
      % mp.nstr(mp.findroot(lambda a: age(rP, a) / Gyr - age_obs, alpha), 9))
for H0, OL, tag in [("67.36", "0.6847", "Planck 2018"), ("73.04", "0.6847", "SH0ES H0, Planck Omega_L"), ("70", "0.7", "round numbers")]:
    rr = rho_lambda(mp.mpf(H0), mp.mpf(OL))
    print("   %-26s rho_Lambda = %s GeV^4  ->  t = %s Gyr" % (tag, mp.nstr(rr, 5), mp.nstr(age(rr, alpha) / Gyr, 5)))

print("\n%d of %d arithmetic checks passed" % (sum(PASS), len(PASS)))
