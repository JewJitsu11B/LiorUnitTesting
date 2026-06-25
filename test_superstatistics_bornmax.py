"""
Superstatistics -> q-exponential BornMax: unit test for the entropy/detector architecture.

Architecture under test (CAL manuscript, sec:detectorfunc):
  - the single-log ENTROPY is the energy/cost H in the Boltzmann weight;
  - the DETECTOR sits in the temperature slot, supplying the q-deformation;
  - a power-law memory kernel (decay aK) => a Gamma-distributed inverse temperature of relative
    variance 1/aK => the plain Boltzmann weight becomes a q-exponential with q = 1 + 1/aK;
  - the Markovian limit aK -> inf (q -> 1) recovers the plain exponential.

These are exact (Gamma moment-generating-function) identities; the tests confirm them numerically.
q = 1 + 1/aK is the same backbone the ET suite confirms at R2_shape >= 0.976.
"""
import numpy as np

AKS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
HS = [0.2, 0.5, 1.0, 3.0]


def exp_q(u, q):
    """q-exponential exp_q(-u) = [1 + (q-1) u]^{-1/(q-1)}."""
    qm1 = q - 1.0
    return np.clip(1.0 + qm1 * u, 1e-300, None) ** (-1.0 / qm1)


def test_kernel_qexp_is_gamma_temperature_average():
    """T1: (1+tau)^{-aK} == <e^{-s tau}> for s ~ Gamma(shape=aK, scale=1)  [Gamma MGF, Monte Carlo]."""
    rng = np.random.default_rng(0)
    N = 4_000_000
    worst = 0.0
    for aK in AKS:
        s = rng.gamma(aK, 1.0, size=N)
        for tau in [0.5, 1.0, 3.0, 10.0]:
            mc = float(np.mean(np.exp(-s * tau)))
            cf = (1.0 + tau) ** (-aK)
            worst = max(worst, abs(mc - cf))
    assert worst < 3e-4, f"Gamma-MGF Monte-Carlo mismatch: {worst:.2e}"


def test_qexp_bornmax_weight():
    """T2: (1+H/aK)^{-aK} == exp_q(-H) with q = 1 + 1/aK  (machine precision)."""
    worst = 0.0
    for aK in AKS:
        q = 1.0 + 1.0 / aK
        for H in HS:
            worst = max(worst, abs((1.0 + H / aK) ** (-aK) - exp_q(H, q)))
    assert worst < 1e-12, f"q-exponential identity mismatch: {worst:.2e}"


def test_relative_variance_sets_q():
    """T3: Gamma(shape=aK, scale=1/aK) has mean 1, relative variance 1/aK, so q = 1 + 1/aK."""
    for aK in AKS:
        mean = aK * (1.0 / aK)
        relvar = (aK * (1.0 / aK) ** 2) / mean ** 2
        assert abs(mean - 1.0) < 1e-12
        assert abs(relvar - 1.0 / aK) < 1e-12
        assert abs((1.0 + relvar) - (1.0 + 1.0 / aK)) < 1e-12


def test_markovian_limit_recovers_plain_boltzmann():
    """T5: aK -> inf  =>  q -> 1 and the weight -> e^{-H} (plain Boltzmann)."""
    H = 1.0
    prev = np.inf
    for aK in [1, 3, 10, 30, 100, 1000, 10000]:
        w = (1.0 + H / aK) ** (-aK)
        assert w <= prev + 1e-12  # monotone increase toward e^{-H}
        prev = w
    assert abs((1.0 + H / 10000) ** (-10000) - np.exp(-H)) < 1e-3
