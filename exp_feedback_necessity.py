"""
Feedback-necessity test: is the self-referential back-coupling (gamma_t -> M_{t+1}, Newton's
third law) the SOURCE of deterministic unpredictability? Two IDENTICAL deterministic loops,
the ONLY manipulated variable being the coupling ON vs OFF.

Deterministic loop (no RNG): agent at continuous position c_t reads swell m and moves toward
the best wave; sharpness beta controls how much it AVERAGES (contractive) vs picks a single
best wave (sharp / can hop between basins):
    c_{t+1} = sum_i i*w_i / sum_i w_i ,   w_i = exp(beta*(m_i - |i-c_t|/sigma))
FEEDBACK ON : riding depletes the swell it commits to:  m -= delta*exp(-(i-c_{t+1})^2/2 sigma^2)
FEEDBACK OFF: m static.

We SWEEP beta because a contractive (low-beta, averaging) rule cannot exhibit sensitive
dependence regardless of feedback - it is structurally incapable of the phenomenon, so a
null there is uninformative. The fair test is whether, in a regime where the map CAN stretch
(higher beta), the feedback is what turns predictable into unpredictable.

Scope: ingredient ONE (randomness from feedback). NOT the Born measure (= the algebra,
verified separately). A positive result here is NOT "therefore Born."
"""
import numpy as np

N = 100
XS = np.arange(N, dtype=float)
SIGMA, DELTA = 5.0, 1.0
T = 400
EPS = 1e-9
C0S = [12.0, 27.0, 41.0, 55.0, 68.0, 83.0]


def init_field():
    m = np.zeros(N)
    for a, f, ph in [(1.0, 3, 0.0), (0.7, 7, 1.1), (0.5, 13, 2.3), (0.4, 5, 0.7)]:
        m += a * np.sin(2 * np.pi * f * XS / N + ph)
    return m - m.min() + 0.1


def move(c, m, beta):
    w = np.exp(beta * (m - np.abs(XS - c) / SIGMA))
    w = w / (w.sum() + 1e-300)
    return float((XS * w).sum())


def run(c0, feedback, beta):
    m = init_field(); c = c0; traj = [c]
    for _ in range(T):
        c = move(c, m, beta)
        if feedback:
            m = np.clip(m - DELTA * np.exp(-(XS - c) ** 2 / (2 * SIGMA ** 2)), 0.01, None)
        traj.append(c)
    return np.array(traj)


def divergence(c0, feedback, beta):
    mA, mB = init_field(), init_field()
    cA, cB = c0, c0 + EPS
    seps = []
    for _ in range(T):
        cA, cB = move(cA, mA, beta), move(cB, mB, beta)
        if feedback:
            mA = np.clip(mA - DELTA * np.exp(-(XS - cA) ** 2 / (2 * SIGMA ** 2)), 0.01, None)
            mB = np.clip(mB - DELTA * np.exp(-(XS - cB) ** 2 / (2 * SIGMA ** 2)), 0.01, None)
        seps.append(abs(cA - cB))
    return np.array(seps)


def lyap(seps):
    s = np.log(np.clip(seps, 1e-300, None))
    hi = max(5, min(int(np.argmax(seps)), len(s) - 1))
    return float(np.polyfit(np.arange(hi + 1), s[:hi + 1], 1)[0]) if hi >= 2 else 0.0


def main():
    print("=" * 86)
    print("Feedback-necessity test (sweep decision sharpness beta: contractive -> sharp)")
    print(f"N={N}, T={T}, eps={EPS}; deterministic (identical c0 -> identical traj, verified below)")
    print("=" * 86)
    # determinism
    d = max(np.abs(run(20.0, fb, 8.0) - run(20.0, fb, 8.0)).max() for fb in (False, True))
    print(f"  determinism: max|trajA-trajB| over identical-c0 reruns = {d:.1e}\n")

    print(f"  {'beta':>5} {'final-sep OFF':>14} {'final-sep ON':>13} {'lyap OFF':>10} {'lyap ON':>9}  verdict")
    for beta in [2.0, 4.0, 8.0, 16.0, 32.0]:
        so = np.array([divergence(c, False, beta) for c in C0S])
        sn = np.array([divergence(c, True, beta) for c in C0S])
        fo, fn = so[:, -1].mean(), sn[:, -1].mean()
        lo = np.mean([lyap(s) for s in so]); ln = np.mean([lyap(s) for s in sn])
        # feedback is the source iff ON diverges (grows >> eps) while OFF does not
        on_div = fn > 1.0 and ln > 0.02
        off_pred = fo < 1.0 and lo < ln
        v = "FEEDBACK = source" if (on_div and off_pred) else ("both diverge" if (fo > 1.0 and fn > 1.0) else "neither (predictable)")
        print(f"  {beta:>5.0f} {fo:>14.2e} {fn:>13.2e} {lo:>+10.4f} {ln:>+9.4f}  {v}")

    print()
    print("Read: a row where ON diverges (final-sep >> eps, lyap_ON>0) and OFF does not is the")
    print("confirmation - the back-coupling turns a predictable loop unpredictable. If no beta")
    print("shows that split, the feedback is NOT the source in this dynamics (honest negative).")
    print("Scope: ingredient ONE only; the Born measure is the algebra, tested separately.")


if __name__ == "__main__":
    main()
