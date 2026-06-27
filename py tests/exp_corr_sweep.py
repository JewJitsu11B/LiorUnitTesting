"""
Sweep the Born-cost correlation across nu_bar.

Earlier we measured corr(detector, surprisal) only at the field's natural nu_bar (~1).
Here we SWEEP nu_bar by offsetting the natural nu-field's mean (keeping its per-position
structure, which is what drives the correlation) and report, at each nu_bar:

  corr(entropy,  surprisal)        single-log entropy  (form="nist")
  corr(detector_clamped, surprisal) cal code            (form="mixed_renyi_tsallis")
  corr(detector_raw,     surprisal) same formula, NO (1-nu_bar) clamp

so we can see whether "detector tracks the surprisal better than the entropy" holds
across nu_bar, not just at the Shannon point, and where the clamp changes the answer.
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
    return (nu * p_c * phi).sum(dim=-1) / p_phi


def detector_raw(p_bil, nu, phi):
    log_nu = torch.log(nu.clamp(min=1e-4))
    p_c = p_bil.clamp(min=1e-10)
    p_pow = torch.exp(nu * torch.log(p_c))
    integrand = (p_pow * (-log_nu) * phi).sum(dim=-1)
    nu_bar = _nu_bar(nu, p_bil, phi)
    sign = torch.sign(integrand)
    log_abs = torch.log(integrand.abs().clamp(min=1e-10))
    return (sign * log_abs / (1.0 - nu_bar)).numpy()


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

    print("=" * 76)
    print(f"corr(., surprisal) swept over nu_bar  (N={N}, natural nu_bar={nubar0:.5f})")
    print("=" * 76)
    print(f"  {'nu_bar':>8} {'corr(entropy)':>14} {'corr(det_clamped)':>18} {'corr(det_raw)':>15}")
    for t in [0.50, 0.70, 0.90, 0.95, 0.99, 1.01, 1.05, 1.10, 1.30, 1.50]:
        nu = (nu0 + (t - nubar0)).clamp(min=1e-4)
        H_ent = variable_order_entropy(p_bil, nu, form="nist").numpy()
        H_c = variable_order_entropy(p_bil, nu, form="mixed_renyi_tsallis").numpy()
        H_r = detector_raw(p_bil, nu, phi)
        print(f"  {t:>8.2f} {_corr(H_ent, surp):>14.4f} {_corr(H_c, surp):>18.4f} "
              f"{_corr(H_r, surp):>15.4f}")

    print()
    print("Read: detector should out-correlate the entropy across nu_bar if it is the")
    print("      Born cost; clamped vs raw differ only where the (1-nu_bar) clamp bites.")


if __name__ == "__main__":
    main()
