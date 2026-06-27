"""
Why did the detector fail to track the surprisal? Hypothesis: the cal
`mixed_renyi_tsallis` form OMITS the content-surprisal factor [-ln_{nu_bar} p] that the
full functional (the infographic) carries. That factor IS the surprisal (it limits to
-ln p as nu->1). We rebuild the FULL functional

   H_full(x) = sign * ln| Integral_y  p^nu * (-ln_{nu_bar} p) * (-ln nu) * phi | / (1 - nu_bar)

and sweep nu_bar, comparing corr(., surprisal) for:
   entropy            (form="nist")
   detector_mixed     cal mixed_renyi_tsallis  (NO content surprisal)  <- what failed
   detector_full      full functional WITH content surprisal           <- the real one

If the content surprisal is the Born cost, corr(detector_full, surprisal) should be
high and STABLE across nu_bar, unlike the mixed form.
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy

torch.manual_seed(0)
N = 140
W_D, W_C = 0.1, 0.3
A_K = 1.0


def _field(seed=42):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tok = geo + spec
    E = energy_matrix(tok)
    E = E.real if torch.is_complex(E) else E
    p = E / (E.sum() + 1e-30)
    return p, p.sum(dim=1)


def _nu_bar(nu, p_bil, phi):
    p_c = p_bil.clamp(min=1e-10)
    p_phi = (p_c * phi).sum(dim=-1).clamp(min=1e-10)
    return (nu * p_c * phi).sum(dim=-1) / p_phi          # (N,)


def _neg_ln_q_p(p_bil, q_vec):
    """Content surprisal -ln_q(p) with per-row order q(x); -> -ln p as q->1."""
    p_c = p_bil.clamp(min=1e-10)
    one_m = (1.0 - q_vec).unsqueeze(1)                   # (N,1)
    near = one_m.abs() < 1e-4
    ln_q = torch.where(near, torch.log(p_c),
                       (torch.exp(one_m * torch.log(p_c)) - 1.0) / one_m)
    return -ln_q                                         # (N,N)


def detector_full(p_bil, nu, phi, clamp=True, lens=True, wrapper=True):
    log_nu = torch.log(nu.clamp(min=1e-4))
    p_c = p_bil.clamp(min=1e-10)
    p_pow = torch.exp(nu * torch.log(p_c))
    nu_bar = _nu_bar(nu, p_bil, phi)
    csurp = _neg_ln_q_p(p_c, nu_bar)                     # content surprisal -> -ln p
    lens_factor = (-log_nu) if lens else torch.ones_like(log_nu)
    integrand = (p_pow * csurp * lens_factor * phi).sum(dim=-1)
    sign = torch.sign(integrand)
    log_abs = torch.log(integrand.abs().clamp(min=1e-10))
    val = sign * log_abs                                 # the inner Renyi argument
    if not wrapper:
        return val.numpy()                               # no 1/(1-nu_bar) division
    denom = (1.0 - nu_bar)
    if clamp:
        denom = denom.clamp(min=1e-6)
    return (val / denom).numpy()


def _corr(a, b):
    a = np.nan_to_num(np.asarray(a, float), posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, float), posinf=0.0, neginf=0.0)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p_bil, p_marg = _field()
    surp = -np.log(p_marg.numpy() + 1e-30)
    phi = torch.ones(N, N, dtype=torch.float32) / N
    K = causal_kernel_matrix(N, A_K)
    nu0 = nu_field(p_marg, K, w_d=W_D, w_c=W_C)
    nubar0 = float(_nu_bar(nu0, p_bil, phi).mean())

    print("=" * 84)
    print(f"corr(., surprisal): mixed (no content surprisal) vs FULL functional, swept nu_bar")
    print(f"(N={N}, natural nu_bar={nubar0:.4f})")
    print("=" * 84)
    print(f"  {'nu_bar':>8} {'corr(full+wrapper)':>18} {'corr(inner,no-wrapper)':>23}")
    for t in [0.50, 0.70, 0.90, 0.95, 0.99, 1.01, 1.05, 1.10, 1.30, 1.50]:
        nu = (nu0 + (t - nubar0)).clamp(min=1e-4)
        H_wrap = detector_full(p_bil, nu, phi, clamp=False, lens=True, wrapper=True)
        H_inner = detector_full(p_bil, nu, phi, clamp=False, lens=True, wrapper=False)
        print(f"  {t:>8.2f} {_corr(H_wrap, surp):>18.4f} {_corr(H_inner, surp):>23.4f}")

    print()
    print("Decisive test of the wrapper hypothesis:")
    print("  if the 1/(1-nu_bar) Renyi wrapper is what destroys tracking near nu_bar=1,")
    print("  the INNER log-integrand (no wrapper) should track the surprisal where the")
    print("  full wrapped detector does not.")


if __name__ == "__main__":
    main()
