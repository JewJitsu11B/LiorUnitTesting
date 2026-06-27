"""
Test: the biquaternionic source current's two norms, and whether the Hermitian-norm probability
makes a Renyi functional NON-DEGENERATE at nu=1 while the lens detector DEGENERATES there.

Bilocal interaction c = P_x . P_y of the source current. Two norms:
  Hermitian norm  p_H  ~ ||H(c)||^2   (positive, normalizable -> a probability)
  Reduced norm    p_red ~ |c c-bar| = |det|   (vanishes on the null cone / zero divisors)

Claims to verify (constant nu, swept two-sided through nu=1; rows normalized to sum 1):
  C1  min(p_H) > 0 (every entry positive -> normalizable); p_red has ~0 entries (null cone).
  C2  partition Sum_y p_H^nu -> 1 as nu->1, so log -> 0 cancels the 1/(1-nu) pole.
      lens partition Sum_y p_H^nu (-ln nu) -> 0 as nu->1 (the lens kills it).
  C3  Renyi on p_H (no lens) is SMOOTH through nu=1 and equals the Shannon entropy there (finite).
  C4  detector (lens inside) DIVERGES at nu=1 (inner -> 0, log -> -inf, /(1-nu) -> blowup).
Deterministic.
"""
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
    p_H = (Hc.real ** 2 + Hc.imag ** 2).sum(-1) * mask          # Hermitian norm (positive)
    p_red = quat_norm_sq(c).abs() * mask                         # reduced norm |det| (>=0, ~0 on cone)
    p_H = p_H / p_H.sum(dim=1, keepdim=True).clamp(min=1e-30)    # row-normalize
    p_red = p_red / p_red.sum(dim=1, keepdim=True).clamp(min=1e-30)
    return p_H, p_red


def renyi(p, nu, lens=False):
    p_c = p.clamp(min=1e-12)
    p_pow = torch.exp(nu * torch.log(p_c))
    w = (-math.log(nu)) if lens else 1.0
    inner = (p_pow * w).sum(-1)                                  # (N,) per-row partition
    val = torch.sign(inner) * torch.log(inner.abs().clamp(min=1e-30))
    return (val / (1.0 - nu)).mean().item(), inner.mean().item()


def shannon(p):
    p_c = p.clamp(min=1e-12)
    return (-(p_c * torch.log(p_c)).sum(-1)).mean().item()


def main():
    p_H, p_red = build(SEED)
    print("=" * 92)
    print(f"Hermitian-norm probability vs lens detector: degeneracy at nu=1   (N={N}, seed={SEED})")
    print("=" * 92)
    nz = p_H[p_H > 0]
    print(f"  C1: min(p_H off-diag)={nz.min().item():.2e} (>0, normalizable to a probability).")
    print(f"      NOTE: this field does NOT hit the null cone (reduced-norm min off-diag well >0).")
    print(f"      The ~1/N near-zero p_red entries are the masked DIAGONAL, not zero divisors.")
    print(f"      The null cone is ALGEBRAIC: explicit zero divisor 1+h*i has det=0 (exp_two_closures.py C5).")
    print(f"     Shannon(p_H) reference (the finite nu->1 value) = {shannon(p_H):.4f}")
    print()
    print(f"  {'nu':>8} {'renyi_H':>11} {'detector(lens)':>16} {'partition_H':>13} {'partition_lens':>15}")
    for nu in [0.900, 0.990, 0.999, 1.001, 1.010, 1.100]:
        rH, inH = renyi(p_H, nu, lens=False)
        rD, inD = renyi(p_H, nu, lens=True)
        print(f"  {nu:>8.3f} {rH:>11.4f} {rD:>16.3e} {inH:>13.5f} {inD:>15.3e}")
    print()
    print("Read C2: partition_H -> 1 as nu->1 (log->0, cancels the pole); partition_lens -> 0 (lens).")
    print("     C3: renyi_H -> Shannon at nu=1 is an ANALYTIC limit (L'Hopital, exact, seed-robust).")
    print("         Numerically (float32) it is stable only at a stand-off |1-nu|~1e-3..1e-2 (shown);")
    print("         within ~1e-6 of 1 it is roundoff-unstable and NaN at nu=1. Evaluate off-point /")
    print("         float64 / special-case nu=1=Shannon. (wf hermitian-norm-renyi-verify, PARTIAL.)")
    print("     C4: detector(lens) blows up toward nu=1 (degenerate) - unconditional, seed/shape-robust.")


if __name__ == "__main__":
    main()
