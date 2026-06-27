"""
Many-to-one read test (the detector's defining claim, per "The Swell & The Surfer"):
different SUCCESSFUL / striven rides through the SAME swell collapse to MORE SIMILAR detector
reads than arbitrary rides. Tests the detector as a lossy, many-to-one PROJECTION, not a
fixed point.

User's formulation:
  ride gamma        = any admissible (causal, forward) path through the field's cells
  read   R[gamma]   = mean_t H^D_hyb(x_t)          (aggregated detector read along the path)
  success logP(g)   = mean_t log p(x_t)            (info transports better on high-prob cells)
  free energy F[g]  = R[g] - beta * logP(g)        (cost - success reward); selected ride minimizes F
  Gamma_all  = random admissible rides
  Gamma_succ = top fraction by success (high logP)
  Gamma_F    = bottom fraction by free energy (the striven / selected ride)
Claim: Var_{Gamma_succ}[R] < Var_{Gamma_all}[R]  (and Var_{Gamma_F}[R] < Var_all), vs a random
same-size control. Detector = the FIXED Hermitian-norm Renyi (non-degenerate). Deterministic.
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field

torch.manual_seed(0)
rng = np.random.default_rng(0)
N = 120
W_D, W_C, A_K = 0.1, 0.3, 1.0
M = 4000
MAXJUMP = 15
BETA = 2.0
TOPFRAC = 0.2


def detector_herm(p_bil, nu):
    pr = p_bil / p_bil.sum(1, keepdim=True).clamp(min=1e-30)
    pc = pr.clamp(min=1e-12)
    nb = (nu * pr).sum(1)
    Z = torch.exp(nu * torch.log(pc)).sum(1)
    H = torch.log(Z.clamp(min=1e-30)) / (1.0 - nb)
    sh = -(pc * torch.log(pc)).sum(1)
    return torch.where((1.0 - nb).abs() < 1e-3, sh, H).numpy()


def random_path():
    cells, x = [0], 0
    while x < N - 1:
        x = min(N - 1, x + int(rng.integers(1, MAXJUMP + 1)))
        cells.append(x)
    return np.array(cells)


def main():
    geo, spec = embed_tokens_channels(N, base_seed=42)
    E = energy_matrix(geo + spec); E = E.real if torch.is_complex(E) else E
    p = E / (E.sum() + 1e-30); p_marg = p.sum(1)
    nu = nu_field(p_marg, causal_kernel_matrix(N, A_K), w_d=W_D, w_c=W_C)
    H_cell = detector_herm(p, nu)
    logp = np.log(p_marg.numpy() + 1e-30)

    paths = [random_path() for _ in range(M)]
    R = np.array([H_cell[g].mean() for g in paths])
    logP = np.array([logp[g].mean() for g in paths])
    F = R - BETA * logP

    k = int(M * TOPFRAC)
    idx_succ = np.argsort(-logP)[:k]
    idx_F = np.argsort(F)[:k]
    idx_rand = rng.choice(M, k, replace=False)
    v_all = R.var()

    print("=" * 82)
    print(f"Many-to-one read test   (N={N}, {M} rides, populations = top {int(TOPFRAC*100)}%)")
    print("=" * 82)
    for name, idx in [("all rides", np.arange(M)), ("random subset (control)", idx_rand),
                      ("high-success  Gamma_succ", idx_succ), ("low-free-energy  Gamma_F", idx_F)]:
        r = R[idx]
        print(f"  {name:>26}: mean R={r.mean():+.4f}  Var R={r.var():.4e}  (n={len(idx)})")
    print()
    print(f"  Var ratio  succ/all = {R[idx_succ].var()/v_all:.3f}   F/all = {R[idx_F].var()/v_all:.3f}"
          f"   rand/all = {R[idx_rand].var()/v_all:.3f}")
    print("  Claim holds if succ/all and F/all are < 1 (and clearly below rand/all ~ 1):")
    print("  successful / striven rides collapse to more similar reads than arbitrary rides.")

    print()
    print("  read-quality knob (nu offset = the surfer's read): mean R over all rides")
    for label, d in [("pessimistic (-0.2)", -0.2), ("clear (0.0)", 0.0), ("overconfident (+0.2)", 0.2)]:
        Hd = detector_herm(p, (nu + d).clamp(min=1e-4))
        Rd = np.array([Hd[g].mean() for g in paths]).mean()
        print(f"    {label:>22}: mean R = {Rd:+.4f}")
    print("  (the read shifts with nu_bar -> 'the read filters the possible')")


if __name__ == "__main__":
    main()
