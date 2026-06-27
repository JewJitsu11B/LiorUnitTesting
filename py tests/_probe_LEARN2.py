"""Follow-up: (1) confirm softmax shift-invariance makes uniform==meanprior,
(2) add a genuinely BAD non-constant frozen prior (anti-correlated with truth),
(3) isolate the clean ceiling test at DELETE=0 where the oracle pi stays valid.
"""
import numpy as np
import itertools

N = 60; T = 600; COSTD = 0.15; XS = np.arange(N)


def landscape():
    v = np.zeros(N)
    for a, f, ph in [(1.0, 2, 0.3), (0.7, 5, 1.1), (0.6, 9, 2.0), (0.4, 3, 0.7)]:
        v += a * np.sin(2 * np.pi * f * XS / N + ph)
    return v - v.min() + 0.1


def step_dist(x, pi, beta):
    logits = beta * (pi - COSTD * np.abs(XS - x))
    p = np.exp(logits - logits.max()); return p / p.sum()


def run(seed, alpha, beta, delete, mode):
    rng = np.random.default_rng(seed)
    v = landscape(); v_true = v.copy()
    if mode == "perfect":   pi = v_true.copy()
    elif mode == "uniform": pi = np.zeros(N)
    elif mode == "anti":    pi = (v_true.max() - v_true)          # inverted truth: bad structured prior
    else:                   pi = np.full(N, v.mean())
    x = N // 2; total = 0.0; learn = (mode == "learn")
    for t in range(T):
        x = int(rng.choice(N, p=step_dist(x, pi, beta)))
        r = v[x]; total += r
        if learn: pi[x] += alpha * (r - pi[x])
        v[x] = max(0.01, v[x] - delete)
    return total


def main():
    seeds = list(range(24))
    # Clean ceiling test: DELETE=0 only, oracle prior stays valid all run
    print("== DELETE=0 (clean ceiling: oracle pi never goes stale) ==")
    print("alpha beta | learn  meanFroz perfect anti  | L/mean L/anti L/perf")
    fps = []
    for alpha, beta in itertools.product([0.1,0.3,0.6],[1.0,3.0,6.0]):
        L = np.mean([run(s,alpha,beta,0.0,"learn") for s in seeds])
        M = np.mean([run(s,alpha,beta,0.0,"meanprior") for s in seeds])
        P = np.mean([run(s,alpha,beta,0.0,"perfect") for s in seeds])
        A = np.mean([run(s,alpha,beta,0.0,"anti") for s in seeds])
        fps.append(L/P)
        print(f"{alpha:4.1f} {beta:4.1f} | {L:6.1f} {M:7.1f} {P:7.1f} {A:6.1f} | "
              f"{L/M:5.3f} {L/A:5.3f} {L/P:5.3f}")
    fps=np.array(fps)
    print(f"  L/perfect @DEL=0: min={fps.min():.3f} med={np.median(fps):.3f} max={fps.max():.3f} "
          f"({(fps<=1.0).sum()}/{len(fps)} at/below ceiling)")

    # Shift-invariance check: uniform vs meanprior identical?
    print("\n== shift-invariance check (uniform pi=0 vs meanprior pi=mean) ==")
    diffs=[]
    for alpha,beta,d in itertools.product([0.3],[3.0],[0.0,0.02,0.05]):
        U=np.mean([run(s,alpha,beta,d,"uniform") for s in seeds])
        M=np.mean([run(s,alpha,beta,d,"meanprior") for s in seeds])
        diffs.append(abs(U-M)); print(f"  DEL={d}: uniform={U:.2f} meanprior={M:.2f} |diff|={abs(U-M):.4f}")
    print(f"  max |uniform-meanprior| = {max(diffs):.6f}  (≈0 => constant priors are policy-identical)")


if __name__ == "__main__":
    main()
