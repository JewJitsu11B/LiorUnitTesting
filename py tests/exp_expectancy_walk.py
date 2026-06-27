"""
Expectancy-theory test: a "ride" as a BIASED RANDOM WALK on a changing causal lattice, with
memory + feedback + learning (bias-update). Models the user's formulation:
  x_{t+1} ~ P(.|x_t, pi_t)        biased random walk (pi = expectancy/policy/prior)
  r_t = response (true landscape) feedback under INCOMPLETE information
  pi_{t+1} = U(pi_t, x_t, r_t)    learning = bias-update after each path-response cycle
  lattice shifts: visits deplete value (overcommitment limits future options) -> non-Markovian

Three falsifiable claims:
  A. LEARNING improves transport: bias-update (learn ON) raises cumulative success over time
     vs a frozen-prior agent (learn OFF).
  B. NON-MARKOV: the next-step distribution at a FIXED state differs by history (early vs learned
     pi) -> KL(P_early || P_late) > 0; the walk is genuinely path-dependent, not Markov.
  C. MANY-TO-ONE (right frame): different stochastic rides (different seeds) take DIVERSE paths
     but converge to SIMILAR learned biases / success -> low across-ride variance of the learned
     outcome despite high path diversity.
The emergent 'stochastics' is EPISTEMIC (incomplete info + stochastic policy), NOT chaos; the
landscape is deterministic. Scope: the randomness mechanism, not the Born measure (the algebra).
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


def run(seed, learn=True):
    rng = np.random.default_rng(seed)
    v = landscape()
    pi = np.full(N, v.mean())          # prior: knows the mean, not the structure (incomplete info)
    x = N // 2
    reward, path = [], [x]
    pi_early = None
    for t in range(T):
        x = int(rng.choice(N, p=step_dist(x, pi)))
        r = v[x]
        reward.append(r)
        if learn:
            pi[x] += ALPHA * (r - pi[x])           # bias update (expectancy learning)
        v[x] = max(0.01, v[x] - DEPLETE)           # changing lattice (overcommitment)
        path.append(x)
        if t == 50:
            pi_early = pi.copy()
    return np.array(reward), pi.copy(), np.array(path), pi_early


def kl(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())


def main():
    seeds = range(24)
    on = [run(s, True) for s in seeds]
    off = [run(s, False) for s in seeds]

    print("=" * 80)
    print("Expectancy-theory test: biased random walk + feedback + learning on a changing lattice")
    print("=" * 80)

    # A. learning improves transport (vs frozen control); within-run decline = depletion coupling
    on_early = np.mean([r[0][:150].mean() for r in on])
    on_late = np.mean([r[0][-150:].mean() for r in on])
    off_early = np.mean([r[0][:150].mean() for r in off])
    off_late = np.mean([r[0][-150:].mean() for r in off])
    on_cum = np.mean([r[0].sum() for r in on]); off_cum = np.mean([r[0].sum() for r in off])
    print("  A. LEARNING (on a depleting / changing lattice):")
    print(f"     learn-ON  early={on_early:.3f}  late={on_late:.3f}  cumulative={on_cum:.1f}")
    print(f"     learn-OFF early={off_early:.3f}  late={off_late:.3f}  cumulative={off_cum:.1f}  (frozen prior, control)")
    print(f"     => learning {'IMPROVES' if on_cum > off_cum * 1.02 else 'does NOT improve'} transport vs the")
    print(f"        frozen control; the within-run decline is the DEPLETION coupling (overcommitment")
    print(f"        erodes the exploited waves), present in both, which is the lattice changing under the ride.")

    # B. non-Markov: next-step dist at fixed states differs early vs late
    r0 = run(0, True)
    pi_e, pi_l = r0[3], r0[1]
    kls = [kl(step_dist(x, pi_e), step_dist(x, pi_l)) for x in range(0, N, 5)]
    print("  B. NON-MARKOV:")
    print(f"     mean KL(P_next early-pi || late-pi) over fixed states = {np.mean(kls):.4f}  "
          f"(>0 => transition depends on history, not just x)")

    # C. many-to-one: diverse paths, convergent learned outcome
    paths = [set(r[2].tolist()) for r in on]
    # path diversity: mean pairwise Jaccard DISTANCE (1 - overlap)
    jac = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            inter = len(paths[i] & paths[j]); uni = len(paths[i] | paths[j])
            jac.append(1 - inter / max(uni, 1))
    pis = np.array([r[1] for r in on])
    late_rewards = np.array([r[0][-150:].mean() for r in on])
    pi_var = float(pis.var(axis=0).mean())            # across-ride variance of learned bias (per cell)
    rew_cv = float(late_rewards.std() / (late_rewards.mean() + 1e-9))
    print("  C. MANY-TO-ONE:")
    print(f"     path diversity (mean pairwise Jaccard distance) = {np.mean(jac):.3f}  (high => rides differ)")
    print(f"     across-ride variance of LEARNED bias pi         = {pi_var:.4f}")
    print(f"     across-ride CV of late reward (outcome)          = {rew_cv:.4f}  (low => convergent outcome)")
    print(f"     => diverse rides, {'CONVERGENT' if rew_cv < 0.15 else 'divergent'} outcome")
    print()
    print("  All deterministic given seed (np Generator seeded). Epistemic stochasticity from the")
    print("  policy + incomplete info; landscape is deterministic. Tests the randomness mechanism only.")


if __name__ == "__main__":
    main()
