"""
EXPLORATORY EXTENSION (NOT a manuscript claim).

Decision-theoretic reading of the CAL BornMax weight:
    P_i = softmax(-cost_i / tau),   cost_i = -ln p_i + lambda * V_i

  - p_i  : Born density |psi|^2 (detector closure cost -ln p reproduces Born at tau=1, lambda=0).
  - V_i  : value/threat potential (chosen vector). High V = costly/threatening; we REWARD by
           lowering effective cost, so we use +lambda*V with V interpreted as a *penalty*, and a
           REWARD is encoded as NEGATIVE V (low cost). To make "high value = attractive" explicit
           we phrase the potential as a threat: high V_i pushes the state DOWN. For high-reward we
           give a state a large NEGATIVE V so its cost drops and it becomes the argmax target.

  P_i = softmax( (ln p_i - lambda V_i) / tau ).

Entropy port (causal_kernel_matrix / nu_field / voe) is reused verbatim from the verified
/tmp/temp_slot_ab.py port of testpy/cal/entropy.py; the value term is the only new ingredient.

Free energy: F(tau) = -tau * ln Sum_i exp(-cost_i/tau)  = log-sum-exp / entropic risk measure.
    tau large  -> F -> mean cost (risk-neutral)
    tau -> 0   -> F -> min cost   (extreme value; argmax of -cost = value-weighted decision)
"""
import numpy as np
np.set_printoptions(precision=5, suppress=True)

# ---------- verified port (verbatim from /tmp/temp_slot_ab.py) ----------
def causal_kernel_matrix(N, alpha_K=0.5):
    x = np.arange(N); y = np.arange(N)
    dist = np.abs(x[:, None] - y[None, :])
    K = (1.0 + dist) ** (-alpha_K)
    mask = (y[None, :] < x[:, None]).astype(float)
    return K * mask

def nu_field(p, K, w_d=0.5, w_c=0.8):
    ratio = p[None, :] / np.clip(p[:, None], 1e-10, None)
    dens = w_d * np.log(1.0 + ratio)
    return 1.0 + dens + w_c * K

# ---------- new ingredient: value/threat cost ----------
def make_cost(p, V, lam):
    """cost_i = -ln p_i + lam * V_i."""
    return -np.log(np.clip(p, 1e-10, None)) + lam * V

def softmax_neg_cost(cost, tau):
    """P_i = softmax(-cost_i / tau)."""
    z = -cost / tau
    z = z - z.max()
    w = np.exp(z)
    return w / w.sum()

def free_energy(cost, tau):
    """F(tau) = -tau ln Sum exp(-cost/tau)  (entropic risk / log-sum-exp)."""
    z = -cost / tau
    m = z.max()
    return -tau * (m + np.log(np.exp(z - m).sum()))

def participation_ratio(P):
    return 1.0 / np.sum(P ** 2)

# ---------- build a Born density (same construction as the port) ----------
rng = np.random.default_rng(0)
N = 8
raw = rng.random(N) + 0.05
p = raw / raw.sum()
K = causal_kernel_matrix(N)
nu = nu_field(p, K)   # field exists / is consistent with the manuscript machinery (not used in cost here)

born_mode = int(np.argmax(p))
print("Born density p =", p)
print("Born mode = argmax p =", born_mode, " p[mode] = %.4f" % p[born_mode])
print("nu field: min %.4f max %.4f mean %.4f" % (nu.min(), nu.max(), nu.mean()))

# =====================================================================
# Choose a value/threat vector V.
# Pick a LOW-probability state and give it HIGH REWARD (large NEGATIVE V => low cost).
# All other states get V = 0 (neutral). This is the high-risk-high-reward gamble.
# =====================================================================
low_p_state = int(np.argmin(p))
V = np.zeros(N)
V[low_p_state] = -1.0   # this state is the "reward magnitude 1" direction; lambda scales it
print("\nLow-p state (the gamble) =", low_p_state, " p = %.4f" % p[low_p_state])
print("Value vector V (reward = negative cost):", V)

# ---------------------------------------------------------------------
# DEMO 1 & 2: tau sweep at a fixed moderate lambda, show broad -> argmax collapse.
# Use lambda in the risk-neutral regime first (lambda = 0): collapse target must be Born mode.
# ---------------------------------------------------------------------
print("\n" + "=" * 72)
print("DEMO 1/2: tau sweep. cost = -ln p (lambda = 0, pure Born).")
print("=" * 72)
demos = []
for tau in [5.0, 1.0, 0.05]:
    cost = make_cost(p, V, lam=0.0)
    P = softmax_neg_cost(cost, tau)
    target = int(np.argmax(P))
    eq_born = (target == born_mode)
    F = free_energy(cost, tau)
    print(f"tau={tau:>5}: PR={participation_ratio(P):.3f}  collapse_target={target}  "
          f"==Born_mode? {eq_born}  F={F:+.4f}")
    print("          P =", P)
    demos.append(("lambda=0, tau=%.2f" % tau, tau, target, eq_born, False, F))

# Now turn ON value at a lambda where the flip has occurred, and repeat the tau sweep.
lam_on = 3.0
print("\n" + "-" * 72)
print(f"DEMO 1/2 (value ON, lambda={lam_on}): cost = -ln p + lambda*V.  argmax(ln p - lambda V).")
print("-" * 72)
for tau in [5.0, 1.0, 0.05]:
    cost = make_cost(p, V, lam=lam_on)
    P = softmax_neg_cost(cost, tau)
    target = int(np.argmax(P))
    eq_born = (target == born_mode)
    eq_val = (target == low_p_state)
    F = free_energy(cost, tau)
    print(f"tau={tau:>5}: PR={participation_ratio(P):.3f}  collapse_target={target}  "
          f"==Born_mode? {eq_born}  ==value_state? {eq_val}  F={F:+.4f}")
    print("          P =", P)
    demos.append(("lambda=%.1f, tau=%.2f" % (lam_on, tau), tau, target, eq_born, eq_val, F))

# ---------------------------------------------------------------------
# DEMO 3: HIGH-RISK-HIGH-REWARD flip. Sweep lambda; at tau->0 the collapse target
# is argmax(ln p - lambda V). Find the lambda threshold where the winner switches
# from the Born mode to the low-p high-value state.
# ---------------------------------------------------------------------
print("\n" + "=" * 72)
print("DEMO 3: high-risk-high-reward. tau->0 collapse target = argmax(ln p - lambda V).")
print("=" * 72)

def collapse_target(lam):
    score = np.log(np.clip(p, 1e-10, None)) - lam * V   # -cost, the thing argmax picks at tau->0
    return int(np.argmax(score))

# Analytic threshold: the value state overtakes the Born mode when
#   ln p_gamble - lam*V_gamble  >=  ln p_mode - lam*V_mode
# With V_mode = 0, V_gamble = -1 (reward), this is:
#   ln p_gamble + lam  >=  ln p_mode   =>  lam >= ln p_mode - ln p_gamble = ln(p_mode/p_gamble)
lam_star_analytic = np.log(p[born_mode] / p[low_p_state])
print(f"Analytic threshold lambda* = ln(p_mode/p_gamble) = ln({p[born_mode]:.4f}/{p[low_p_state]:.4f}) "
      f"= {lam_star_analytic:.4f}")

lams = np.linspace(0.0, 3.0, 3001)
targets_small_tau = np.array([collapse_target(l) for l in lams])
# also verify via actual softmax at tiny tau
tau_tiny = 0.01
targets_softmax = []
for l in lams[::100]:
    cost = make_cost(p, V, l)
    targets_softmax.append(int(np.argmax(softmax_neg_cost(cost, tau_tiny))))

switch_idx = np.argmax(targets_small_tau != born_mode)
lam_star_numeric = lams[switch_idx]
print(f"Numeric threshold (grid, argmax score): first lambda where target != Born mode = "
      f"{lam_star_numeric:.4f}")
print(f"  below threshold target = {targets_small_tau[0]} (Born mode {born_mode})")
print(f"  above threshold target = {targets_small_tau[-1]} (value state {low_p_state})")
print(f"  softmax@tau={tau_tiny} agrees at sampled points: "
      f"{all((t==born_mode) or (t==low_p_state) for t in targets_softmax)}")

# Show the flip explicitly at three lambdas straddling the threshold, at small tau.
print("\nFlip demonstration at tau=0.05 across the threshold:")
for lam in [lam_star_analytic - 0.5, lam_star_analytic, lam_star_analytic + 0.5]:
    lam = max(lam, 0.0)
    cost = make_cost(p, V, lam)
    P = softmax_neg_cost(cost, 0.05)
    target = int(np.argmax(P))
    eq_born = (target == born_mode)
    eq_val = (target == low_p_state)
    print(f"  lambda={lam:.4f}: collapse_target={target}  ==Born? {eq_born}  ==value_state? {eq_val}")
    demos.append(("flip lambda=%.3f, tau=0.05" % lam, 0.05, target, eq_born, eq_val,
                  free_energy(cost, 0.05)))

# ---------------------------------------------------------------------
# DEMO 4: free energy = entropic risk measure. Show tau large -> mean cost,
# tau -> 0 -> min cost, for the lambda-ON gamble cost.
# ---------------------------------------------------------------------
print("\n" + "=" * 72)
print("DEMO 4: free energy F(tau) = -tau ln Sum exp(-cost/tau) = entropic risk / LSE.")
print("=" * 72)
cost = make_cost(p, V, lam=lam_on)
print("cost vector (lambda=%.1f):" % lam_on, cost)
print("mean(cost) =", np.mean(cost), "  min(cost) =", np.min(cost))
for tau in [50.0, 5.0, 1.0, 0.1, 0.01]:
    F = free_energy(cost, tau)
    print(f"  tau={tau:>6}: F={F:+.5f}")
print("=> tau large: F -> mean(cost) (risk-neutral average).")
print("=> tau -> 0 : F -> min(cost)  (extreme value; argmax of -cost = value-weighted choice).")

# ---------------------------------------------------------------------
# Machine-readable summary for the structured object.
# ---------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY (for structured object)")
print("=" * 72)
print("BORN_MODE=%d" % born_mode)
print("GAMBLE_STATE=%d" % low_p_state)
print("P_MODE=%.6f  P_GAMBLE=%.6f" % (p[born_mode], p[low_p_state]))
print("LAMBDA_STAR_ANALYTIC=%.6f" % lam_star_analytic)
print("LAMBDA_STAR_NUMERIC=%.6f" % lam_star_numeric)
cost0 = make_cost(p, V, lam=lam_on)
print("F(tau=50)=%.5f  MEAN_COST=%.5f" % (free_energy(cost0, 50.0), np.mean(cost0)))
print("F(tau=0.01)=%.5f  MIN_COST=%.5f" % (free_energy(cost0, 0.01), np.min(cost0)))
