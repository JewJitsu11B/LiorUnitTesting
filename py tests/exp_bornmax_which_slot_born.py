import numpy as np
from math import lgamma, log
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

np.set_printoptions(precision=6, suppress=True)

print("="*72)
print("PART 1   Which slot produces the Born rule?")
print("="*72)

# a small normalized amplitude psi over K states
psi   = np.array([0.10, 0.20, 0.30, 0.60, 0.70])
pborn = psi**2
pborn = pborn / pborn.sum()                 # target Born weights |psi|^2
print("target Born weights |psi|^2      :", pborn)

# the two closure forms the paper states:
H_det = -np.log(pborn)                       # detector cost  -> -ln|psi|^2   (line 1151)
H_nu  = np.zeros_like(pborn)                 # entropy at nu=1: -ln nu = 0     (line 1062-3)
print("detector cost  H_det = -ln|psi|^2:", H_det)
print("entropy H^nu at closure (nu=1)   :", H_nu, " (annihilates)")

def weights(numer, denom):
    w = np.exp(-numer/denom)
    return w / w.sum()

# ---- Assignment A : DETECTOR in cost/numerator, tau in temperature/denominator
tau = 1.0
P_A = weights(H_det, tau)
print("\nAssignment A  (paper's channel eqn  theta_R = -H_det/tau):")
print("  P ∝ exp(-H_det/tau), tau=1       :", P_A)
print("  equals Born |psi|^2 ?            :", np.allclose(P_A, pborn),
      "  max err =", f"{np.max(np.abs(P_A-pborn)):.2e}")
print("  tau is a real temperature (P ∝ |psi|^(2/tau)) -- sharpness knob:")
for t in (0.5, 1.0, 2.0):
    Pt = weights(H_det, t)
    print(f"    tau={t}:  {Pt}   participation ratio {1.0/np.sum(Pt**2):.3f}")

# ---- Assignment B : DETECTOR in temperature/denominator, entropy as cost/numerator
P_B = weights(H_nu, H_det)
print("\nAssignment B  (proposed: detector in temp slot, entropy in cost slot):")
print("  P ∝ exp(-H^nu/H_det) at closure  :", P_B)
print("  numerator H^nu=0 -> exp(0)=1 -> uniform 1/K =", round(1/len(psi),6))
print("  equals Born |psi|^2 ?            :", np.allclose(P_B, pborn))

# even OFF closure (nu<1) B cannot reproduce Born, and it INVERTS the ordering:
nu = 0.7
H_nu_off = (-log(nu))*np.ones_like(pborn)    # scalar order -> uniform numerator
P_B_off  = weights(H_nu_off, H_det)
print("  off-closure (nu=0.7)             :", P_B_off, " Born?", np.allclose(P_B_off,pborn))
print("   -> larger cost H_det (smaller |psi|^2) becomes MORE likely: sign inverted.")

print("\n" + "="*72)
print("PART 2   The kernel's role: scale-free K(tau) -> Gamma-spread temp -> exp_q")
print("="*72)

def exp_q(u, q):                              # exp_q(-u) = [1+(q-1)u]^{-1/(q-1)}
    return (1 + (q-1)*u)**(-1.0/(q-1))

def gamma_pdf(b, a, s):                        # Gamma(shape a, scale s)
    return np.exp((a-1)*np.log(b) - b/s - lgamma(a) - a*np.log(s))

beta0 = 1.0                                    # mean inverse temperature <beta>
b = np.linspace(1e-6, 80.0, 800000)
for alpha_K in (1.0, 2.0, 5.0):
    q = 1 + 1.0/alpha_K                        # paper: q = 1 + 1/alpha_K   (line 1165)
    a, s = alpha_K, beta0/alpha_K              # rel. variance 1/a = 1/alpha_K, mean a*s=beta0
    f = gamma_pdf(b, a, s)
    print(f"\nalpha_K={alpha_K}  ->  q = 1+1/alpha_K = {q:.4f}   (norm check {trapz(f,b):.4f})")
    for u in (0.3, 1.0, 3.0):
        Bnum = trapz(f*np.exp(-b*u), b)        # < e^{-beta u} >_Gamma  (superstatistics)
        Bq   = exp_q(beta0*u, q)               # exp_q(-<beta> u)
        Bana = (1 + u*s)**(-a)                 # closed form (1+us)^{-a}
        print(f"   u={u}:  <e^-bu>_Gamma={Bnum:.6f}   exp_q={Bq:.6f}   analytic={Bana:.6f}"
              f"   match={np.isclose(Bnum,Bq,rtol=1e-4) and np.isclose(Bq,Bana)}")

print("\nSummary:  cost/energy = detector (gives Born);  temperature = tau;")
print("          kernel K(tau)=(1+tau)^-alpha_K sets only the deformation q=1+1/alpha_K.")
