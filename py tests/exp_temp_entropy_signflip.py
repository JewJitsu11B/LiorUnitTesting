"""
Entropy-in-TEMPERATURE-slot collapse study.

cost_i = -ln p_i  (detector closure value; reproduces Born at T=1).
P_i = softmax(-cost_i / T_i),  T_i = per-position variable-order entropy (nist form).

Branches:
 1. BASELINE nu>=1 (nu_field as-is)             -> T_var<0 (neg temp) -> argmin p collapse
 2. FLIP via flip_density=True (code C5 control) -> report nu regime + collapse
 3. FLIP via sub-unitary field nu' = 2 - nu      -> T_var>0? argmax p collapse?
 4. FLIP via negating the lens (+ln nu; T -> -T_var) -> collapse target?

Ported line-for-line from testpy/cal/entropy.py (torch->numpy). No reinvention.
"""
import numpy as np
np.set_printoptions(precision=5, suppress=True)

# ---------- exact ports of cal/entropy.py ----------
def causal_kernel_matrix(N, alpha_K=0.5):
    x = np.arange(N); y = np.arange(N)
    dist = np.abs(x[:, None] - y[None, :])
    K = (1.0 + dist) ** (-alpha_K)
    mask = (y[None, :] < x[:, None]).astype(float)   # y < x (past)
    return K * mask

def nu_field(p, K, w_d=0.5, w_c=0.8, flip_density=False):
    ratio = p[None, :] / np.clip(p[:, None], 1e-10, None)   # p(y)/p(x)
    dens = w_d * np.log(1.0 + ratio)
    if flip_density:
        dens = -dens
    return 1.0 + dens + w_c * K

def voe_nist(p_bil, nu, phi=None, negate_lens=False):
    """H = -Sum p^{nu+1} ln(nu) phi   (variable-order entropy, nist form).
    negate_lens: use +ln(nu) instead of -ln(nu) i.e. return -H (branch 4)."""
    N = nu.shape[0]
    if phi is None:
        phi = np.ones((N, N)) / N
    p = np.clip(p_bil, 1e-10, None)
    log_nu = np.log(np.clip(nu, 1e-4, None))
    p_pow = np.exp((nu + 1.0) * np.log(p))
    H = -(p_pow * log_nu * phi).sum(axis=-1)
    return -H if negate_lens else H

# ---------- BornMax mechanics ----------
def softmax_w(cost, T):
    z = -cost / T
    z = z - z.max()
    w = np.exp(z)
    return w / w.sum()

def summarize(P, p):
    collapse_index = int(np.argmax(P))
    argmax_p_index = int(np.argmax(p))
    corr = float(np.corrcoef(P, p)[0, 1]) if np.std(P) > 1e-12 else 0.0
    pr = float(1.0 / np.sum(P ** 2))
    return collapse_index, argmax_p_index, corr, pr

def run_branch(name, p, T, cost):
    P = softmax_w(cost, T)
    ci, ai, corr, pr = summarize(P, p)
    T_sign = "mixed"
    if np.all(T > 0):   T_sign = "positive"
    elif np.all(T < 0): T_sign = "negative"
    argmin_p = int(np.argmin(p))
    return {
        "name": name, "T": T, "T_sign": T_sign, "P": P,
        "collapse_index": ci, "argmax_p_index": ai, "argmin_p_index": argmin_p,
        "collapses_to_argmax_p": ci == ai,
        "collapses_to_argmin_p": ci == argmin_p,
        "corr": corr, "pr": pr,
    }

# ---------- sweep ----------
seeds = [0, 1, 2, 3, 4, 5, 6]
Ns = [6, 8, 12]

branch_names = ["1_baseline", "2_flip_density", "3_subunitary", "4_negate_lens"]
agg = {b: {"nu_gt1_frac": [], "T_sign": [], "collapse_index": [], "argmax_p_index": [],
           "argmin_p_index": [], "eq_argmax": [], "eq_argmin": [], "corr": [], "pr": []}
       for b in branch_names}

# store one detailed printout per branch for the first (seed=0,N=8) config
detail = {}

for N in Ns:
    for seed in seeds:
        rng = np.random.default_rng(seed)
        raw = rng.random(N) + 0.05
        p = raw / raw.sum()
        K = causal_kernel_matrix(N)
        cost = -np.log(np.clip(p, 1e-10, None))
        p_bil = np.broadcast_to(p[None, :], (N, N)).copy()

        # nu fields
        nu_base = nu_field(p, K)                       # nu >= 1
        nu_flip = nu_field(p, K, flip_density=True)    # C5 control
        nu_sub  = 2.0 - nu_base                         # sub-unitary where nu>1

        # temperatures
        T1 = voe_nist(p_bil, nu_base)                       # baseline
        T2 = voe_nist(p_bil, nu_flip)                       # flip_density lens
        T3 = voe_nist(p_bil, nu_sub)                        # sub-unitary
        T4 = voe_nist(p_bil, nu_base, negate_lens=True)     # negate lens (= -T1)

        results = {
            "1_baseline":      (run_branch("1_baseline", p, T1, cost), nu_base),
            "2_flip_density":  (run_branch("2_flip_density", p, T2, cost), nu_flip),
            "3_subunitary":    (run_branch("3_subunitary", p, T3, cost), nu_sub),
            "4_negate_lens":   (run_branch("4_negate_lens", p, T4, cost), nu_base),
        }

        for b, (r, nuf) in results.items():
            frac_gt1 = float(np.mean(nuf > 1.0))
            agg[b]["nu_gt1_frac"].append(frac_gt1)
            agg[b]["T_sign"].append(r["T_sign"])
            agg[b]["collapse_index"].append(r["collapse_index"])
            agg[b]["argmax_p_index"].append(r["argmax_p_index"])
            agg[b]["argmin_p_index"].append(r["argmin_p_index"])
            agg[b]["eq_argmax"].append(r["collapses_to_argmax_p"])
            agg[b]["eq_argmin"].append(r["collapses_to_argmin_p"])
            agg[b]["corr"].append(r["corr"])
            agg[b]["pr"].append(r["pr"])

            if seed == 0 and N == 8:
                detail[b] = (r, nuf, p, cost)

# ---------- detailed printout for seed=0,N=8 ----------
print("="*78)
print("DETAIL: seed=0, N=8")
print("="*78)
r0 = detail["1_baseline"]
p = r0[2]; cost = r0[3]
print("p (Born)      :", p)
print("cost = -ln p  :", cost)
print("argmax_p =", int(np.argmax(p)), "  argmin_p =", int(np.argmin(p)))
for b in branch_names:
    r, nuf, p, cost = detail[b]
    print(f"\n--- {b} ---")
    print(f"  nu: min {nuf.min():.4f} max {nuf.max():.4f} mean {nuf.mean():.4f}  frac(nu>1)={np.mean(nuf>1.0):.3f}")
    print(f"  T : {r['T']}")
    print(f"  T_sign={r['T_sign']}")
    print(f"  P : {r['P']}")
    print(f"  collapse_index={r['collapse_index']}  argmax_p={r['argmax_p_index']}  argmin_p={r['argmin_p_index']}")
    print(f"  ==argmax_p: {r['collapses_to_argmax_p']}   ==argmin_p: {r['collapses_to_argmin_p']}")
    print(f"  corr(P,Born)={r['corr']:+.4f}  PR={r['pr']:.4f}")

# ---------- aggregate summary ----------
print("\n" + "="*78)
print("AGGREGATE across seeds", seeds, "and N", Ns, f"  ({len(seeds)*len(Ns)} configs each)")
print("="*78)
from collections import Counter
for b in branch_names:
    a = agg[b]
    tsigns = Counter(a["T_sign"])
    eq_ax = np.mean(a["eq_argmax"])
    eq_am = np.mean(a["eq_argmin"])
    print(f"\n{b}")
    print(f"  frac(nu>1): mean {np.mean(a['nu_gt1_frac']):.3f}  (min {np.min(a['nu_gt1_frac']):.3f} max {np.max(a['nu_gt1_frac']):.3f})")
    print(f"  T_sign distribution: {dict(tsigns)}")
    print(f"  collapse==argmax_p : {eq_ax*100:.1f}% of configs")
    print(f"  collapse==argmin_p : {eq_am*100:.1f}% of configs")
    print(f"  corr(P,Born): mean {np.mean(a['corr']):+.4f}  (min {np.min(a['corr']):+.4f} max {np.max(a['corr']):+.4f})")
    print(f"  PR: mean {np.mean(a['pr']):.4f}  (min {np.min(a['pr']):.4f} max {np.max(a['pr']):.4f})")

# robustness flags: is the collapse target the SAME across all seeds/N per branch?
print("\n" + "="*78)
print("ROBUSTNESS")
print("="*78)
robust = True
for b in branch_names:
    a = agg[b]
    ax_consistent = len(set(a["eq_argmax"])) == 1
    am_consistent = len(set(a["eq_argmin"])) == 1
    ts_consistent = len(set(a["T_sign"])) == 1
    print(f"  {b}: T_sign_consistent={ts_consistent}  eq_argmax_consistent={ax_consistent}  eq_argmin_consistent={am_consistent}")
    if not (ts_consistent and ax_consistent and am_consistent):
        robust = False
print(f"\nROBUST_ACROSS_SEEDS_AND_N = {robust}")
