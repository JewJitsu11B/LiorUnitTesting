"""
A/B: detector as COST in both arms; TEMPERATURE = variable-order entropy (arm A)
vs Shannon entropy (control, arm B). Functions ported line-for-line from
testpy/cal/entropy.py (torch -> numpy). No reinvention.
"""
import numpy as np
np.set_printoptions(precision=5, suppress=True)

# ---- exact ports of cal/entropy.py ----
def causal_kernel_matrix(N, alpha_K=0.5):
    x = np.arange(N); y = np.arange(N)
    dist = np.abs(x[:, None] - y[None, :])
    K = (1.0 + dist) ** (-alpha_K)
    mask = (y[None, :] < x[:, None]).astype(float)   # y < x (past)
    return K * mask

def nu_field(p, K, w_d=0.5, w_c=0.8):
    ratio = p[None, :] / np.clip(p[:, None], 1e-10, None)   # p(y)/p(x)
    dens = w_d * np.log(1.0 + ratio)
    return 1.0 + dens + w_c * K

def voe(p, nu, phi=None, form="nist"):
    N = nu.shape[0]
    if phi is None:
        phi = np.ones((N, N)) / N
    if form == "nist":                       # H = -Sum p^{nu+1} ln(nu) phi   (my var-order entropy)
        p_bil = np.clip(p, 1e-10, None)
        log_nu = np.log(np.clip(nu, 1e-4, None))
        p_pow = np.exp((nu + 1.0) * np.log(p_bil))
        return -(p_pow * log_nu * phi).sum(axis=-1)
    if form == "mixed_renyi_tsallis":        # the detector (Renyi outside / Tsallis inside)
        p_bil = np.clip(p, 1e-10, None)
        log_nu = np.log(np.clip(nu, 1e-4, None))
        p_pow_nu = np.exp(nu * np.log(p_bil))
        integrand = (p_pow_nu * (-log_nu) * phi).sum(axis=-1)
        p_phi = np.clip((p_bil * phi).sum(axis=-1), 1e-10, None)
        nu_bar = (nu * p_bil * phi).sum(axis=-1) / p_phi
        sign = np.sign(integrand)
        log_abs = np.log(np.clip(np.abs(integrand), 1e-10, None))
        return sign * log_abs / np.clip(1.0 - nu_bar, 1e-6, None)
    p_clamped = np.clip(p, 1e-10, None)
    if form == "shannon":                    # H = -Sum p ln p phi     (control baseline)
        return -(p_clamped[None, :] * np.log(p_clamped[None, :]) * phi).sum(axis=-1)
    # renyi
    nu_bar = nu.mean(axis=-1)
    p_pow = np.exp(nu * np.log(p_clamped[None, :]))
    integral = (p_pow * phi).sum(axis=-1)
    return np.log(np.clip(integral, 1e-10, None)) / np.clip(1.0 - nu_bar, 1e-6, None)

# ---- build one belief density and its field ----
rng = np.random.default_rng(0)
N = 8
raw = rng.random(N) + 0.05
p = raw / raw.sum()                       # Born target |psi|^2
K = causal_kernel_matrix(N)
nu = nu_field(p, K)                        # (N,N) field
p_bil = np.broadcast_to(p[None, :], (N, N)).copy()   # p(x,y) = p(y)

# COST = detector.  Operative cost = its closure value -ln p (what the BornMax uses,
# and what gave Born at T=1 in test A).  Also show the raw functional for scale.
cost = -np.log(np.clip(p, 1e-10, None))
det_func = voe(p_bil, nu, form="mixed_renyi_tsallis")

# TEMPERATURES
T_var = voe(p_bil, nu, form="nist")        # arm A: my variable-order entropy
T_sha = voe(p,     nu, form="shannon")     # arm B: Shannon control

print("nu field:  min %.4f  max %.4f  mean %.4f  (>=1 => -ln nu <= 0)" % (nu.min(), nu.max(), nu.mean()))
print("p (Born)          :", p)
print("cost = -ln p      :", cost)
print("detector functional (raw, diagnostic):", det_func)
print("T_var (var-order) :", T_var, "  sign:", np.sign(T_var).astype(int))
print("T_sha (shannon)   :", T_sha, "  sign:", np.sign(T_sha).astype(int))

def softmax_w(cost, T):
    z = -cost / T
    z = z - z.max()
    w = np.exp(z)
    return w / w.sum()

def report(name, P):
    err = np.max(np.abs(P - p))
    pr = 1.0 / np.sum(P ** 2)
    order_ok = bool(np.all(np.argsort(P) == np.argsort(p)))
    corr = float(np.corrcoef(P, p)[0, 1])
    print(f"\n{name}")
    print(f"   max|P-Born| = {err:.4f}   PR = {pr:.3f}   same_order_as_Born = {order_ok}   corr(P,Born) = {corr:+.3f}")
    print("   P =", P)

print("\n" + "="*70)
report("ANCHOR  T=1 (constant): P = exp(-cost) = |psi|^2", softmax_w(cost, 1.0))
report("ARM A   T = variable-order entropy (per position)", softmax_w(cost, T_var))
report("ARM B   T = Shannon entropy (control)", softmax_w(cost, T_sha))
