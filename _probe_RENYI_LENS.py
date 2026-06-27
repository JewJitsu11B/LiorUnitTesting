"""
Refutation probe for C3 and C4.

C3 (Renyi, no lens): renyi(p,nu) = (1/(1-nu)) * sign(S)*log|S|, S = Sum_y p^nu.
  Claim: for ANY row-normalized prob, this is finite (-> Shannon) at nu near 1.
  Refute by finding a distribution where it blows up.

C4 (lens detector): inner = Sum_y p^nu (-ln nu). As nu->1, (-ln nu)->0, so inner->0,
  log|inner| -> -inf, /(1-nu) blows up.
  Claim: always diverges at nu->1. Refute by finding a distribution where it stays finite.
"""
import math
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


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


def normalize(x):
    x = x.abs()
    return x / x.sum(dim=1, keepdim=True).clamp(min=1e-30)


# Build a battery of row-normalized distributions
N = 64
dists = {}
dists["uniform"] = normalize(torch.ones(N, N))
dists["random_uniform"] = normalize(torch.rand(N, N))
dists["peaked_softmax_hot"] = torch.softmax(torch.randn(N, N) * 20.0, dim=1)  # very peaked
dists["peaked_softmax_mild"] = torch.softmax(torch.randn(N, N) * 2.0, dim=1)
# sparse: one dominant entry per row plus tiny noise
sp = torch.full((N, N), 1e-9)
sp[torch.arange(N), torch.randint(0, N, (N,))] = 1.0
dists["sparse_near_delta"] = normalize(sp)
# exact one-hot (degenerate delta)
oh = torch.zeros(N, N)
oh[torch.arange(N), 0] = 1.0
dists["exact_onehot"] = normalize(oh)
# power-law / zipf
ranks = torch.arange(1, N + 1).float()
dists["zipf"] = normalize((1.0 / ranks).unsqueeze(0).repeat(N, 1))
# heavy tail with one huge and many tiny
ht = torch.full((N, N), 1e-12)
ht[:, 0] = 1.0
dists["extreme_concentration"] = normalize(ht)

# nu values bracketing 1 very tightly
nus = [0.9, 0.99, 0.999, 0.9999, 1.0001, 1.001, 1.01, 1.1]

print("=" * 100)
print("(a) C3 REFUTATION ATTEMPT: does Renyi (no lens) EVER blow up near nu=1?")
print("=" * 100)
c3_refuted = False
c3_max_dev = 0.0
for name, p in dists.items():
    sh = shannon(p)
    print(f"\n  dist={name:24s}  Shannon={sh:.5f}")
    for nu in nus:
        rH, inH = renyi(p, nu, lens=False)
        dev = abs(rH - sh)
        c3_max_dev = max(c3_max_dev, dev) if math.isfinite(dev) else float('inf')
        flag = ""
        if not math.isfinite(rH) or abs(rH) > 1e6:
            flag = "  <-- BLOWS UP (refutes C3)"
            c3_refuted = True
        print(f"    nu={nu:<8} renyi_H={rH:>14.5f}  S={inH:>12.5e}  |renyi-Shannon|={dev:.4e}{flag}")

print("\n" + "=" * 100)
print("(b) C4 REFUTATION ATTEMPT: does the LENS detector EVER stay finite near nu=1?")
print("=" * 100)
c4_refuted = False
for name, p in dists.items():
    print(f"\n  dist={name:24s}")
    blow_trend = []
    for nu in nus:
        rD, inD = renyi(p, nu, lens=True)
        blow_trend.append((nu, abs(rD)))
        finite_flag = ""
        # "stays finite" near nu=1 would mean magnitude does NOT explode as nu->1
        print(f"    nu={nu:<8} detector={rD:>16.4e}  inner(lens)={inD:>14.4e}")
    # check the two values closest to nu=1 from each side
    near = [(nu, m) for nu, m in blow_trend if abs(nu - 1.0) <= 1e-3]
    far = [(nu, m) for nu, m in blow_trend if abs(nu - 1.0) >= 0.1]
    near_mag = max(m for _, m in near) if near else 0.0
    far_mag = max(m for _, m in far) if far else 0.0
    # divergence => near magnitude >> far magnitude (grows as nu->1)
    if near_mag <= far_mag * 2 and math.isfinite(near_mag):
        # did NOT blow up approaching nu=1 -> would refute C4
        # but must confirm it is genuinely staying finite, not just clamped
        print(f"    -> near-1 max mag={near_mag:.3e} vs far max mag={far_mag:.3e}: "
              f"does NOT diverge -> would refute C4")
        c4_refuted = True
    else:
        print(f"    -> near-1 max mag={near_mag:.3e} >> far max mag={far_mag:.3e}: DIVERGES (consistent with C4)")

print("\n" + "=" * 100)
print(f"SUMMARY: C3 refuted (Renyi blew up)? {c3_refuted}   max|renyi-Shannon| near 1 = {c3_max_dev:.4e}")
print(f"         C4 refuted (lens stayed finite)? {c4_refuted}")
print("=" * 100)
