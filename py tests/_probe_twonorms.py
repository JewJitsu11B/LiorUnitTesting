import torch
from cal.biquaternion import (quat_mul, hermitian, quat_norm_sq,
                              biquat_to_matrix, embed_tokens_channels)

torch.manual_seed(0)
N = 140
SEED = 42

geo, spec = embed_tokens_channels(N, base_seed=SEED)
P = geo + spec
Px = P.unsqueeze(1).expand(N, N, 4)
Py = P.unsqueeze(0).expand(N, N, 4)
c = quat_mul(Px, Py)
mask = (~torch.eye(N, dtype=torch.bool))

# Hermitian norm ||H(c)||^2 (positive)
Hc = hermitian(c)
herm = (Hc.real ** 2 + Hc.imag ** 2).sum(-1)

# Reduced norm |c.cbar| = |N(c)| = |det|
red = quat_norm_sq(c).abs()

# det of matrix rep
M = biquat_to_matrix(c)
det = (M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0])
absdet = det.abs()

# off-diagonal selections
herm_off = herm[mask]
red_off = red[mask]
absdet_off = absdet[mask]

print("N off-diagonal entries:", herm_off.numel())
print("min Hermitian norm (off-diag): %.6e" % herm_off.min().item())
print("min reduced norm  (off-diag): %.6e" % red_off.min().item())
print("max reduced norm  (off-diag): %.6e" % red_off.max().item())

# fraction of reduced-norm below 1e-8 (on the FULL matrix, matching exp script's p_red<1e-8 after norm)
# but here use raw reduced norm off-diagonal
thr = 1e-8
below = red_off < thr
print("frac(reduced norm off-diag < 1e-8): %.4f  (count=%d)" % (below.float().mean().item(), int(below.sum())))

# Confirm reduced norm == |det| exactly (algebraic identity N(c) == det M)
print("max |reduced - |det|| over all: %.3e" % (red - absdet).abs().max().item())

# Do the near-zero reduced-norm entries coincide with det~0?
if int(below.sum()) > 0:
    coincide = (absdet_off[below] < thr).all().item()
    print("all reduced-norm<1e-8 entries also have |det|<1e-8: %s" % coincide)
    print("max |det| among reduced<1e-8 entries: %.3e" % absdet_off[below].max().item())
else:
    # check smallest entries anyway
    k = 20
    vals, idx = torch.topk(-red_off, k)
    print("no entries below 1e-8; smallest %d reduced-norm values:" % k)
    print("  reduced:", (-vals).tolist())
    print("  |det| at same:", absdet_off[idx].tolist())

# Also: does Hermitian norm EVER vanish? min on full off-diag is the test.
print("Hermitian norm strictly positive on all off-diag: %s" % (herm_off.min().item() > 0))

# scale context: median reduced norm to judge whether 1e-8 is "near zero"
print("median reduced norm (off-diag): %.3e" % red_off.median().item())
print("min reduced/median ratio: %.3e" % (red_off.min().item() / red_off.median().item()))
