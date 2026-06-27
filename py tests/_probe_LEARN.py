"""Probe: does bias-update LEARNING beat frozen controls across a sweep,
and does it approach the perfect-prior ceiling?

Model copied faithfully from exp_expectancy_walk.py:
  x_{t+1} ~ softmax(beta*(pi - COSTD*|XS-x|))
  r = v[x]; if learn: pi[x] += alpha*(r-pi[x]); v[x] -= DELETE (depletion)

Frozen controls (learn=False) differ only in the FIXED pi:
  - 'meanprior' : pi = v.mean() everywhere (original control / incomplete info)
  - 'perfect'   : pi = TRUE landscape v (oracle ceiling)
  - 'uniform'   : pi = 0 everywhere (bad/uninformative prior)
"""
import numpy as np
import itertools

N = 60
T = 600
COSTD = 0.15
XS = np.arange(N)


def landscape():
    v = np.zeros(N)
    for a, f, ph in [(1.0, 2, 0.3), (0.7, 5, 1.1), (0.6, 9, 2.0), (0.4, 3, 0.7)]:
        v += a * np.sin(2 * np.pi * f * XS / N + ph)
    return v - v.min() + 0.1


def step_dist(x, pi, beta):
    logits = beta * (pi - COSTD * np.abs(XS - x))
    p = np.exp(logits - logits.max())
    return p / p.sum()


def run(seed, alpha, beta, delete, mode):
    """mode in {'learn','meanprior','perfect','uniform'}"""
    rng = np.random.default_rng(seed)
    v = landscape()
    v_true = v.copy()
    if mode == "perfect":
        pi = v_true.copy()
    elif mode == "uniform":
        pi = np.zeros(N)
    else:  # learn or meanprior both start at the mean prior
        pi = np.full(N, v.mean())
    x = N // 2
    total = 0.0
    learn = (mode == "learn")
    for t in range(T):
        x = int(rng.choice(N, p=step_dist(x, pi, beta)))
        r = v[x]
        total += r
        if learn:
            pi[x] += alpha * (r - pi[x])
        v[x] = max(0.01, v[x] - delete)
    return total


def main():
    alphas = [0.1, 0.3, 0.6]
    betas = [1.0, 3.0, 6.0]
    deletes = [0.0, 0.02, 0.05]
    seeds = list(range(24))

    ratios_vs_mean = []
    ratios_vs_uniform = []
    frac_of_perfect = []
    rows = []

    for alpha, beta, delete in itertools.product(alphas, betas, deletes):
        learn = np.mean([run(s, alpha, beta, delete, "learn") for s in seeds])
        meanp = np.mean([run(s, alpha, beta, delete, "meanprior") for s in seeds])
        perf = np.mean([run(s, alpha, beta, delete, "perfect") for s in seeds])
        unif = np.mean([run(s, alpha, beta, delete, "uniform") for s in seeds])

        r_mean = learn / meanp
        r_unif = learn / unif
        f_perf = learn / perf
        ratios_vs_mean.append(r_mean)
        ratios_vs_uniform.append(r_unif)
        frac_of_perfect.append(f_perf)
        rows.append((alpha, beta, delete, learn, meanp, perf, unif, r_mean, r_unif, f_perf))

    print("alpha beta  DEL | learnON  meanFROZEN perfect  uniform | L/mean L/unif L/perf")
    for (a, b, d, L, M, P, U, rm, ru, fp) in rows:
        print(f"{a:4.1f} {b:4.1f} {d:4.2f} | {L:8.1f} {M:9.1f} {P:8.1f} {U:8.1f} | "
              f"{rm:5.3f} {ru:5.3f} {fp:5.3f}")

    print()
    rm = np.array(ratios_vs_mean); ru = np.array(ratios_vs_uniform); fp = np.array(frac_of_perfect)
    print(f"N cells = {len(rows)} (=3x3x3), seeds/cell = {len(seeds)}")
    print(f"L/meanFrozen  ratio: min={rm.min():.3f} med={np.median(rm):.3f} max={rm.max():.3f}  "
          f"({(rm>1.0).sum()}/{len(rm)} cells >1.0)")
    print(f"L/uniform     ratio: min={ru.min():.3f} med={np.median(ru):.3f} max={ru.max():.3f}  "
          f"({(ru>1.0).sum()}/{len(ru)} cells >1.0)")
    print(f"L/perfect     frac : min={fp.min():.3f} med={np.median(fp):.3f} max={fp.max():.3f}  "
          f"({(fp<=1.02).sum()}/{len(fp)} cells <=1.02 i.e. below/at ceiling)")


if __name__ == "__main__":
    main()
