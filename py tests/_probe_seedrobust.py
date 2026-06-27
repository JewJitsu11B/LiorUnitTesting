"""Seed-robustness probe for renyi_H non-degeneracy and lens-detector divergence at nu=1.

Replicates the construction from exp_hermitian_norm_renyi.py and sweeps base_seed.
"""
import math
import numpy as np
import torch
from cal.biquaternion import quat_mul, hermitian, quat_norm_sq, embed_tokens_channels

torch.manual_seed(0)
N = 140


def build(seed):
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    P = geo + spec
    Px = P.unsqueeze(1).expand(N, N, 4); Py = P.unsqueeze(0).expand(N, N, 4)
    c = quat_mul(Px, Py)
    mask = (~torch.eye(N, dtype=torch.bool)).float()
    Hc = hermitian(c)
    p_H = (Hc.real ** 2 + Hc.imag ** 2).sum(-1) * mask
    p_H = p_H / p_H.sum(dim=1, keepdim=True).clamp(min=1e-30)
    return p_H


def renyi(p, nu, lens=False):
    p_c = p.clamp(min=1e-12)
    p_pow = torch.exp(nu * torch.log(p_c))
    w = (-math.log(nu)) if lens else 1.0
    inner = (p_pow * w).sum(-1)
    val = torch.sign(inner) * torch.log(inner.abs().clamp(min=1e-30))
    return (val / (1.0 - nu)).mean().item()


def shannon(p):
    p_c = p.clamp(min=1e-12)
    return (-(p_c * torch.log(p_c)).sum(-1)).mean().item()


seeds = [42, 0, 1, 7, 123, 2024, 999, 31337]
print(f"{'seed':>6} {'renyi_H@.999':>14} {'renyi_H@1.001':>14} {'Shannon':>10} "
      f"{'|rH-Shan|max':>13} {'lens@.999':>14}")
all_ok = True
for s in seeds:
    p_H = build(s)
    rH_lo = renyi(p_H, 0.999, lens=False)
    rH_hi = renyi(p_H, 1.001, lens=False)
    sh = shannon(p_H)
    lens_lo = renyi(p_H, 0.999, lens=True)
    dmax = max(abs(rH_lo - sh), abs(rH_hi - sh))
    finite = math.isfinite(rH_lo) and math.isfinite(rH_hi)
    near = dmax < 0.05
    huge = abs(lens_lo) > 1e3
    ok = finite and near and huge
    all_ok = all_ok and ok
    flag = "OK" if ok else "FAIL"
    print(f"{s:>6} {rH_lo:>14.5f} {rH_hi:>14.5f} {sh:>10.4f} "
          f"{dmax:>13.2e} {lens_lo:>14.3e}  {flag}")
print()
print("ALL_SEEDS_OK" if all_ok else "SOME_SEED_FAILED")
