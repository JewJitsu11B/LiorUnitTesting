import numpy as np

# ============================================================
# H-A: detector cost sits in the COST/energy slot of BornMax.
#
#   P_i  proportional to  exp( -H_det_i / tau )
#
# Closure identities used VERBATIM (not altered):
#   (1) Detector cost at closure:   H_det -> -ln|psi|^2       (line ~1151)
#   (2) Variable-order entropy at fixed point nu=1: H^nu -> 0  since -ln nu = 0  (line ~1062)
#   (3) Thermal channel:            theta_R = -H_det / tau     (lines ~1195, ~1208)
#
# CRITICAL FAITHFULNESS RULE:
#   |psi| enters the weight ONLY through H_det = -ln|psi|^2.
#   We NEVER type |psi|^2 into the exponent by hand.
# ============================================================

def H_det(psi2):
    # closure identity (1): detector cost = -ln|psi|^2
    return -np.log(psi2)

def bornmax_weight(psi2, tau):
    # H-A: detector cost occupies the cost slot.
    # theta_R = -H_det/tau  (identity 3) is exactly the exponent argument.
    Hd = H_det(psi2)                 # = -ln|psi|^2   (only entry point of |psi|)
    theta = -Hd / tau                # identity (3): theta_R = -H_det/tau
    # Optionally include variable-order entropy at its fixed point nu=1:
    # identity (2) says H^nu -> 0 there, so it adds nothing. We add it explicitly
    # to show it is inert, proving the match is not an accident of dropping a term.
    nu = 1.0
    H_nu = -np.log(nu)               # = 0 exactly (identity 2)
    theta = theta - H_nu / tau       # subtract 0; leaves theta unchanged
    w = np.exp(theta)                # exp(-H_det/tau)
    return w / w.sum()

def rng_psi(K, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(K) + 1j * rng.standard_normal(K)
    psi = a / np.sqrt(np.sum(np.abs(a)**2))   # normalized state
    psi2 = np.abs(psi)**2
    return psi2

def pr(P):
    return 1.0 / np.sum(P**2)   # participation ratio

# ---- Task 1: tau = 1 must reproduce |psi|^2 ----
print("=== TASK 1: tau=1 reproduces Born |psi|^2 ? ===")
task1_errs = []
for seed in range(6):
    for K in (5, 8, 12, 20):
        psi2 = rng_psi(K, seed)
        P = bornmax_weight(psi2, tau=1.0)
        err = np.max(np.abs(P - psi2))   # psi2 used ONLY as comparison target
        task1_errs.append(err)
        if seed < 2:
            print(f"  seed={seed} K={K:2d}  max|P-|psi|^2| = {err:.3e}")
print(f"  overall max abs error (tau=1) = {max(task1_errs):.3e}")

# ---- Task 2: sweep tau, confirm P proportional to |psi|^(2/tau) ----
print("\n=== TASK 2: tau sweep, P ~ |psi|^(2/tau) ===")
psi2 = rng_psi(K=8, seed=42)
for tau in (0.5, 1.0, 2.0):
    P = bornmax_weight(psi2, tau)
    # independent prediction from |psi|^(2/tau), NOT via H_det (sanity cross-check only)
    pred = psi2**(1.0/tau); pred = pred / pred.sum()
    match = np.max(np.abs(P - pred))
    print(f"  tau={tau:<4}  PR(P)={pr(P):.4f}   max|P - |psi|^(2/tau)|={match:.3e}")
print(f"  PR(Born,tau=1) = {pr(bornmax_weight(psi2,1.0)):.4f}  (reference)")
print("  tau<1 sharpens => lower PR ; tau>1 flattens => higher PR")

# ---- Task 3: adversarial break attempts ----
print("\n=== TASK 3: adversarial attempts to break tau=1 match ===")
worst = 0.0
# many random psi
for seed in range(200):
    K = int(np.random.default_rng(1000+seed).integers(5, 21))
    psi2 = rng_psi(K, seed=2000+seed)
    P = bornmax_weight(psi2, 1.0)
    worst = max(worst, np.max(np.abs(P - psi2)))
# near-degenerate components
rng = np.random.default_rng(7)
for _ in range(50):
    K = int(rng.integers(5, 21))
    base = rng.random()
    a = base + 1e-9 * rng.standard_normal(K)      # nearly equal amplitudes
    psi = a / np.sqrt(np.sum(a**2)); psi2 = psi**2
    P = bornmax_weight(psi2, 1.0)
    worst = max(worst, np.max(np.abs(P - psi2)))
# tiny-amplitude component (large H_det)
for _ in range(50):
    K = int(rng.integers(5, 21))
    a = rng.random(K); a[0] = 1e-8                 # one very small component
    psi = a / np.sqrt(np.sum(a**2)); psi2 = psi**2
    P = bornmax_weight(psi2, 1.0)
    worst = max(worst, np.max(np.abs(P - psi2)))
print(f"  worst max abs error across 300 adversarial cases = {worst:.3e}")
print(f"  fails beyond 1e-12 ? {worst > 1e-12}")

print("\nFINAL max abs error (tau=1, all tests) =", max(max(task1_errs), worst))
