"""
Decisive check: does corr(detector, surprisal) survive WITHOUT the (1-nu_bar) clamp?

The centerpiece claim "the detector is the Born cost" rests on corr(detector, surprisal)
~ 0.984, computed with the cal code's CLAMPED detector. The two-sided limit test showed
the clamp REVERSES the detector's sign for nu_bar > 1, and the natural field sits at
nu_bar just above 1. So we recompute the correlation on the natural field with the RAW
(unclamped) detector and compare the sign:

  corr(raw, surprisal) ~ +0.98  -> claim robust; the clamp only hurt the magnitude.
  corr(raw, surprisal) ~ -0.98  -> claim was a clamp artifact (raw detector ANTI-tracks).

Also reports the fraction of positions with nu_bar > 1 (how uniformly the clamp flips).
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy

torch.manual_seed(0)
N = 140
W_D, W_C = 0.1, 0.3


def _field(seed=42):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tok = geo + spec
    E = energy_matrix(tok)
    E = E.real if torch.is_complex(E) else E
    p = E / (E.sum() + 1e-30)
    return p, p.sum(dim=1)


def detector_raw(p_bil, nu, phi):
    log_nu = torch.log(nu.clamp(min=1e-4))
    p_c = p_bil.clamp(min=1e-10)
    p_pow = torch.exp(nu * torch.log(p_c))
    integrand = (p_pow * (-log_nu) * phi).sum(dim=-1)
    p_phi = (p_c * phi).sum(dim=-1).clamp(min=1e-10)
    nu_bar = (nu * p_c * phi).sum(dim=-1) / p_phi
    sign = torch.sign(integrand)
    log_abs = torch.log(integrand.abs().clamp(min=1e-10))
    return sign * log_abs / (1.0 - nu_bar), nu_bar


def _corr(a, b):
    a = np.nan_to_num(np.asarray(a, float), posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, float), posinf=0.0, neginf=0.0)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p_bil, p_marg = _field()
    surprisal = -np.log(p_marg.numpy() + 1e-30)
    phi = torch.ones(N, N, dtype=torch.float32) / N

    print("=" * 72)
    print("Does corr(detector, surprisal) survive without the clamp?")
    print("=" * 72)
    print(f"  {'alpha_K':>8} {'corr(clamped,surp)':>20} {'corr(raw,surp)':>16} {'frac(nu_bar>1)':>16}")
    for aK in [0.5, 1.0, 1.5, 2.0]:
        K = causal_kernel_matrix(N, aK)
        nu = nu_field(p_marg, K, w_d=W_D, w_c=W_C)
        H_clamped = variable_order_entropy(p_bil, nu, form="mixed_renyi_tsallis").numpy()
        H_raw, nu_bar = detector_raw(p_bil, nu, phi)
        H_raw = H_raw.numpy()
        frac = float((nu_bar > 1.0).float().mean())
        print(f"  {aK:>8.2f} {_corr(H_clamped, surprisal):>20.4f} "
              f"{_corr(H_raw, surprisal):>16.4f} {frac:>16.3f}")

    print()
    print("Verdict rule: if corr(raw,surp) is ~+0.98 the Born-cost claim holds;")
    print("              if ~-0.98 it was a sign artifact of the clamp.")


if __name__ == "__main__":
    main()
