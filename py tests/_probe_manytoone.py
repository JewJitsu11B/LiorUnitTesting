"""
Probe for Claim C (many-to-one is LEARNING-driven, not trivial).

Compare learning agents vs a no-learning null (frozen prior = pure biased
random walk on the SAME depleting lattice). For BOTH populations report:
  - path diversity: mean pairwise Jaccard distance of visited cells
  - outcome CV: across-ride std/mean of late reward

C is learning-specific iff:
  learning  -> HIGH diversity AND LOW outcome-CV
  null      -> lower diversity OR higher outcome-CV
If the null ALSO shows high diversity + low CV, convergence is trivial.
"""
import numpy as np

# import the exact model
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ew", "/home/sam_leizerman/LiorFresh/testpy/exp_expectancy_walk.py")
ew = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ew)


def diversity_and_cv(runs):
    paths = [set(r[2].tolist()) for r in runs]
    jac = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            inter = len(paths[i] & paths[j])
            uni = len(paths[i] | paths[j])
            jac.append(1 - inter / max(uni, 1))
    late = np.array([r[0][-150:].mean() for r in runs])
    cv = float(late.std() / (late.mean() + 1e-9))
    return float(np.mean(jac)), cv, float(late.mean()), float(late.std())


def main():
    seeds = range(24)
    on = [ew.run(s, learn=True) for s in seeds]
    off = [ew.run(s, learn=False) for s in seeds]

    div_on, cv_on, m_on, s_on = diversity_and_cv(on)
    div_off, cv_off, m_off, s_off = diversity_and_cv(off)

    print("=" * 72)
    print("Claim C probe: learning vs no-learning null (frozen prior)")
    print("=" * 72)
    print(f"  LEARN-ON :  path-diversity(Jaccard)={div_on:.4f}   "
          f"outcome-CV={cv_on:.4f}   late mean={m_on:.4f} std={s_on:.4f}")
    print(f"  LEARN-OFF:  path-diversity(Jaccard)={div_off:.4f}   "
          f"outcome-CV={cv_off:.4f}   late mean={m_off:.4f} std={s_off:.4f}")
    print()

    # decision logic
    learn_converges = (div_on > 0.30) and (cv_on < 0.15)
    null_converges = (div_off > 0.30) and (cv_off < 0.15)
    print(f"  learning many-to-one (div>0.30 & CV<0.15)? {learn_converges}")
    print(f"  null     many-to-one (div>0.30 & CV<0.15)? {null_converges}")
    print(f"  CV ratio (null/learn) = {cv_off / max(cv_on,1e-9):.3f}")
    print(f"  diversity diff (learn-null) = {div_on - div_off:+.4f}")

    if learn_converges and not null_converges:
        print("  VERDICT: many-to-one is LEARNING-driven (C non-trivial)")
    elif learn_converges and null_converges:
        print("  VERDICT: null ALSO converges -> convergence NOT learning-driven (C trivial)")
    else:
        print("  VERDICT: learning does not show clean many-to-one")


if __name__ == "__main__":
    main()
