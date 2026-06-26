"""
Is the CAL interpolator's past term TRULY a fractional integral?

The CAL accumulation law (cal.cal_tensor.cal_tensor) is

    T_CAL = alpha * present + (1 - alpha) * transported_past,

and the past term convolves the history against the memory kernel
w(tau) = (1 + tau)^{-alpha_K}  (cal.cal_tensor.power_law_kernel).

The Riemann-Liouville fractional integral of order mu is a convolution with the
power-law kernel (t - s)^{mu - 1}:

    (I^mu f)(t) = 1/Gamma(mu) * integral_0^t (t - s)^{mu - 1} f(s) ds.

Matching the kernel exponents, (1 + tau)^{-alpha_K} <-> (t - s)^{mu - 1} gives

    mu = 1 - alpha_K.

So alpha_K < 1 is a fractional INTEGRAL, alpha_K = 1 the marginal case, and
alpha_K > 1 a fractional DERIVATIVE -- the interpolation a fractional operator must
show. We falsify three signatures of a genuine fractional integral, on test
function f(t) = t^beta, using the ACTUAL code kernel where possible:

  (1) EXPONENT : (K * f)(t) ~ t^{beta + mu}              [structural form]
  (2) GAMMA    : coefficient -> Gamma(mu) Gamma(beta+1)/Gamma(beta+1+mu)
                 (raw, un-normalized kernel = Gamma(mu) * RL)   [semigroup constant]
  (3) SEMIGROUP: (K * (K * f))(t) ~ t^{beta + 2 mu}          [orders add]

The (1 + tau) shift regularizes the s->t singularity (Hadamard style), so exact RL
equality is only ASYMPTOTIC (large t). We report the asymptotic match and flag any
exact-equality gap honestly rather than declaring a pass.
"""
import math
import numpy as np

# Tie the test to the real code kernel when the torch env is present; otherwise
# fall back to the identical raw shape so the math check still runs.
try:
    from cal.cal_tensor import power_law_kernel
    _HAVE_CAL = True
except Exception:
    _HAVE_CAL = False


def raw_kernel(tau, alpha_K):
    """The fractional-integral SHAPE used by power_law_kernel, un-normalized."""
    return (1.0 + tau) ** (-alpha_K)


def apply_kernel(f, alpha_K):
    """Discrete causal convolution g(t) = sum_{s<t} (1+(t-s))^{-alpha_K} f(s), dt=1."""
    T = len(f)
    g = np.zeros(T)
    for t in range(1, T):
        s = np.arange(0, t)
        g[t] = np.sum(raw_kernel(t - s, alpha_K) * f[s])
    return g


def fit_powerlaw(t, g, lo, hi):
    """Fit g(t) ~ C t^p on the window [lo, hi]; return (p, C)."""
    m = (t >= lo) & (t <= hi) & (g > 0)
    p, logC = np.polyfit(np.log(t[m]), np.log(g[m]), 1)
    return p, math.exp(logC)


def main():
    T = 4000
    beta = 2.0
    t = np.arange(T, dtype=float)
    f = np.power(np.clip(t, 1e-9, None), beta)
    lo, hi = T // 2, T - 1          # asymptotic window

    print("=" * 84)
    print(f"CAL memory kernel (1+tau)^(-alpha_K) vs Riemann-Liouville I^mu, mu = 1 - alpha_K")
    print(f"test fn f(t)=t^{beta:.0f}, asymptotic fit window t in [{lo},{hi}], "
          f"cal import={'yes' if _HAVE_CAL else 'NO (raw shape)'}")
    print("=" * 84)
    header = (f"  {'alpha_K':>7} {'mu':>6} {'p_fit':>8} {'p_RL':>7} {'p_err':>9} "
              f"{'C_fit/C_RL':>11} {'p2_fit':>8} {'p2_RL':>7}")
    print(header)

    for alpha_K in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]:
        mu = 1.0 - alpha_K

        # If cal is importable, confirm the code kernel equals our raw shape.
        if _HAVE_CAL:
            w_code = power_law_kernel(64, alpha_K).numpy()
            w_raw = raw_kernel(np.arange(1, 64), alpha_K)
            w_raw = w_raw / w_raw.sum()
            assert np.allclose(w_code, w_raw, atol=1e-6), "code kernel != raw shape"

        # (1) exponent + (2) coefficient
        g = apply_kernel(f, alpha_K)
        p_fit, C_fit = fit_powerlaw(t, g, lo, hi)
        p_RL = beta + mu
        # raw (un-normalized) kernel coefficient = Gamma(mu) Gamma(beta+1)/Gamma(beta+1+mu)
        try:
            C_RL = (math.gamma(mu) * math.gamma(beta + 1.0)
                    / math.gamma(beta + 1.0 + mu)) if mu > 0 else float("nan")
        except ValueError:
            C_RL = float("nan")
        cratio = (C_fit / C_RL) if (C_RL == C_RL and C_RL != 0) else float("nan")

        # (3) semigroup: apply twice, exponent must reach beta + 2 mu
        g2 = apply_kernel(g, alpha_K)
        p2_fit, _ = fit_powerlaw(t, g2, lo, hi)
        p2_RL = beta + 2.0 * mu

        print(f"  {alpha_K:>7.2f} {mu:>6.2f} {p_fit:>8.4f} {p_RL:>7.2f} "
              f"{p_fit - p_RL:>9.4f} {cratio:>11.4f} {p2_fit:>8.4f} {p2_RL:>7.2f}")

    print()
    print("Read:")
    print("  p_err -> 0          : exponent shift matches I^mu (fractional-integral form).")
    print("  C_fit/C_RL -> 1     : Gamma-ratio coefficient matches (only defined for mu>0).")
    print("  p2_fit -> beta+2mu  : semigroup holds (composing adds orders).")
    print("  alpha_K<1 integral, =1 marginal, >1 derivative: the fractional interpolation.")
    print("  Exact RL equality is only asymptotic (the 1+tau shift regularizes tau=0);")
    print("  a residual p_err or C ratio != 1 means 'asymptotically a fractional integral',")
    print("  not 'exactly' -- reported, not hidden.")


if __name__ == "__main__":
    main()
