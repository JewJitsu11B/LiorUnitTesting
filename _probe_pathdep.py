"""
RIGOROUS path-dependence (genuine non-Markov) probe for Claim B.

The reference test compares step_dist(x, pi_early) vs step_dist(x, pi_late) from ONE run.
That only shows the transition kernel CHANGES OVER TIME (time-inhomogeneity). A
time-inhomogeneous Markov chain x_{t+1} ~ K_t(.|x_t) would pass that test trivially
while still being Markov: at fixed (x, t), the next-step law is identical regardless of
how you arrived at x.

GENUINE non-Markov (path-dependence) means: conditioning on (x_t = x, time = t), the
next-step distribution STILL depends on the past path. The clean test:

  - Run many seeded agents.
  - Find pairs of agents that occupy the SAME state x at the SAME time t.
  - Split into:
      DIFF-history pairs: arrived at (x,t) via different past paths
      SAME-history pairs: arrived at (x,t) via identical past paths
  - For each pair, compute the divergence (KL / total variation) between their
    ACTUAL next-step distributions step_dist(x, pi) given each agent's hidden pi.
  - If DIFF-history pairs diverge significantly MORE than SAME-history pairs,
    the process is genuinely path-dependent (non-Markov) in the observed x.

We hold t fixed (so time-inhomogeneity is controlled out) and x fixed (so it is not
just "different states"). Any remaining divergence is pure path-dependence.
"""
import numpy as np

N = 60
ALPHA = 0.3
BETA = 3.0
COSTD = 0.15
T = 600
DEPLETE = 0.02
XS = np.arange(N)


def landscape():
    v = np.zeros(N)
    for a, f, ph in [(1.0, 2, 0.3), (0.7, 5, 1.1), (0.6, 9, 2.0), (0.4, 3, 0.7)]:
        v += a * np.sin(2 * np.pi * f * XS / N + ph)
    return v - v.min() + 0.1


def step_dist(x, pi):
    logits = BETA * (pi - COSTD * np.abs(XS - x))
    p = np.exp(logits - logits.max())
    return p / p.sum()


def kl(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def run_full(seed, learn=True):
    """Return per-timestep (x_t, pi_t snapshot, path-up-to-t hash) for the full trajectory."""
    rng = np.random.default_rng(seed)
    v = landscape()
    pi = np.full(N, v.mean())
    x = N // 2
    # record state at each decision time t: x_t (current), pi BEFORE the t-th move,
    # and the path prefix that led here.
    rec_x = []        # x_t at time t
    rec_pi = []       # pi at time t (hidden state that drives the next-step dist)
    rec_prefix = []   # tuple of path so far (x_0..x_t)
    prefix = [x]
    for t in range(T):
        rec_x.append(x)
        rec_pi.append(pi.copy())
        rec_prefix.append(tuple(prefix))
        x = int(rng.choice(N, p=step_dist(x, pi)))
        r = v[x]
        if learn:
            pi[x] += ALPHA * (r - pi[x])
        v[x] = max(0.01, v[x] - DEPLETE)
        prefix.append(x)
    return rec_x, rec_pi, rec_prefix


def main():
    rng_seeds = range(400)  # many agents
    print("=" * 80)
    print("RIGOROUS path-dependence probe (controls out time-inhomogeneity)")
    print("=" * 80)

    for learn in (True, False):
        agents = [run_full(s, learn=learn) for s in rng_seeds]
        nA = len(agents)

        # Build index: (t, x) -> list of (agent_id, prefix, pi)
        from collections import defaultdict
        bucket = defaultdict(list)
        for aid, (rec_x, rec_pi, rec_prefix) in enumerate(agents):
            for t in range(T):
                bucket[(t, rec_x[t])].append((aid, rec_prefix[t], rec_pi[t]))

        diff_kl, diff_tv = [], []
        same_kl, same_tv = [], []
        n_diff_pairs = n_same_pairs = 0
        # cap pairs per bucket to keep it tractable but representative
        for (t, x), members in bucket.items():
            if len(members) < 2:
                continue
            m = len(members)
            for i in range(m):
                for j in range(i + 1, m):
                    aid_i, pref_i, pi_i = members[i]
                    aid_j, pref_j, pi_j = members[j]
                    di = step_dist(x, pi_i)
                    dj = step_dist(x, pi_j)
                    k = kl(di, dj)
                    t_ = tv(di, dj)
                    if pref_i == pref_j:
                        same_kl.append(k); same_tv.append(t_); n_same_pairs += 1
                    else:
                        diff_kl.append(k); diff_tv.append(t_); n_diff_pairs += 1

        tag = "LEARN-ON" if learn else "LEARN-OFF (frozen pi)"
        print(f"\n  [{tag}]  agents={nA}, T={T}")
        print(f"    same-(t,x) pairs total: diff-history={n_diff_pairs}  same-history={n_same_pairs}")
        if n_diff_pairs:
            print(f"    DIFF-history next-step divergence:  mean KL={np.mean(diff_kl):.5f}  "
                  f"mean TV={np.mean(diff_tv):.5f}  (n={n_diff_pairs})")
        if n_same_pairs:
            print(f"    SAME-history next-step divergence:  mean KL={np.mean(same_kl):.5f}  "
                  f"mean TV={np.mean(same_tv):.5f}  (n={n_same_pairs})")
        else:
            print(f"    SAME-history next-step divergence:  (no same-history collisions found)")
        if n_diff_pairs and n_same_pairs:
            print(f"    RATIO diff/same (TV): {np.mean(diff_tv)/(np.mean(same_tv)+1e-12):.2f}x")
        elif n_diff_pairs:
            print(f"    -> same-history pairs would have IDENTICAL pi by determinism => TV=0 exactly.")
            print(f"       diff-history TV={np.mean(diff_tv):.5f} > 0  => path-dependent at fixed (t,x).")

    print()
    print("  Structural fact: x alone is NOT a Markov state. pi (learned policy) and v")
    print("  (depleted lattice) are hidden variables carrying the full history. The process")
    print("  is Markov ONLY in the augmented state (x, pi, v). Same-history agents share")
    print("  identical (pi, v) by determinism, hence identical next-step law; different-history")
    print("  agents at the same (t, x) generally do not.")


if __name__ == "__main__":
    main()
