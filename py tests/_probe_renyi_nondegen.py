import math
import numpy as np
import torch
from cal.biquaternion import quat_mul, hermitian, quat_norm_sq, embed_tokens_channels

torch.manual_seed(0)
N = 140
SEED = 42


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
    return (val / (1.0 - nu)).mean().item(), inner.mean().item()


def shannon(p):
    p_c = p.clamp(min=1e-12)
    return (-(p_c * torch.log(p_c)).sum(-1)).mean().item()


def main():
    p_H = build(SEED)
    S = shannon(p_H)
    print(f"N={N} seed={SEED}")
    print(f"Shannon(p_H) analytic -Sum p ln p = {S:.10f}")
    print(f"min(p_H>0) = {p_H[p_H>0].min().item():.3e}")
    print()
    print(f"{'nu':>12} {'renyi_H':>16} {'partition_H':>16} {'renyi-Shannon':>16}")
    eps = [1e-2, 1e-3, 1e-4, 1e-5]
    nus = []
    for e in eps:
        nus.append(1.0 - e)
    nus.append(1.0)  # exact (should NaN/inf via 0/0 in code, check)
    for e in reversed(eps):
        nus.append(1.0 + e)
    for nu in nus:
        if nu == 1.0:
            try:
                rH, inH = renyi(p_H, nu, lens=False)
            except Exception as ex:
                print(f"{nu:>12.6f}  EXC {ex}")
                continue
            print(f"{nu:>12.6f} {rH:>16.6f} {inH:>16.8f} {'(nu=1 exact)':>16}")
            continue
        rH, inH = renyi(p_H, nu, lens=False)
        print(f"{nu:>12.6f} {rH:>16.8f} {inH:>16.8f} {rH-S:>16.3e}")
    print()

    # L'Hopital: lim_{nu->1} log(Z(nu))/(1-nu) where Z(nu)=Sum p^nu.
    # Z(1)=1, Z'(nu)=Sum p^nu ln p, Z'(1)=Sum p ln p = -S.
    # f=log Z, g=1-nu. f'=Z'/Z, g'=-1. lim f'/g' = (Z'(1)/Z(1))/(-1) = -(-S)/1 ...
    # careful: -(Z'(1)) = -(-S)=S? Z'(1)=Sum p ln p = -S(neg). f'(1)=Z'(1)/Z(1)=-S. g'=-1.
    # lim = f'(1)/g'(1) = (-S)/(-1) = S. So L'Hopital limit = Shannon S.
    p_c = p_H.clamp(min=1e-12)
    Zprime1 = (p_c * torch.log(p_c)).sum(-1)            # per row = -S_row
    Z1 = p_c.sum(-1)                                     # ~1 per row
    lhop_per_row = (Zprime1 / Z1) / (-1.0)               # = -(Sum p ln p) = S_row
    lhop = lhop_per_row.mean().item()
    print(f"L'Hopital limit  mean[(Z'(1)/Z(1))/(-1)] = {lhop:.10f}")
    print(f"matches Shannon? diff = {lhop - S:.3e}")
    print(f"mean Z(1) (partition at nu=1) = {Z1.mean().item():.10f}")

    # scan for divergence anywhere very close to 1
    print()
    print("divergence scan (|renyi_H| max over fine grid):")
    grid = np.concatenate([1.0 - np.logspace(-1, -8, 40), 1.0 + np.logspace(-1, -8, 40)])
    mx = 0.0; arg = None
    for nu in grid:
        rH, _ = renyi(p_H, float(nu), lens=False)
        if not math.isfinite(rH):
            print(f"  NONFINITE at nu={nu}")
        if abs(rH) > mx:
            mx = abs(rH); arg = nu
    print(f"  max|renyi_H| = {mx:.6f} at nu={arg:.10f}  (Shannon={S:.6f})")


if __name__ == "__main__":
    main()
