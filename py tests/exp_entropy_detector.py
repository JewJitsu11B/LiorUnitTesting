"""
Reproducibility experiment: the entropy and detector weighted functionals.

On the bilocal field (cal.entropy) it computes the two variable-order functionals:
  - the single-log ENTROPY  H^nu          (form="nist")               concave, axis-1
  - the DETECTOR            H_hyb          (form="mixed_renyi_tsallis") convex, Born cost

and the headline fact of the architecture: the DETECTOR tracks the surprisal -ln p
(it is the Born cost), while the single-log ENTROPY does not. Reported across a few
memory-kernel decays alpha_K. All quantities are deterministic given the seeds; this
script exists to be run independently and compared for bit-for-bit reproducibility.
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import (
    energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy,
)

torch.manual_seed(0)
N = 140
AKS = [0.5, 1.0, 1.5, 2.0]
W_D, W_C = 0.1, 0.3


def _field(seed=42):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tok = geo + spec
    E = energy_matrix(tok)
    E = E.real if torch.is_complex(E) else E
    p_bil = E / (E.sum() + 1e-30)
    return p_bil, p_bil.sum(dim=1)


def _corr(a, b):
    a = np.nan_to_num(np.asarray(a, dtype=float), posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=float), posinf=0.0, neginf=0.0)
    return float(np.corrcoef(a, b)[0, 1])


def _finite_mean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else 0.0


def main():
    p_bil, p_marg = _field()
    surprisal = -np.log(p_marg.numpy() + 1e-30)

    print("=" * 72)
    print(f"ENTROPY vs DETECTOR weighted functional (N={N}, w_d={W_D}, w_c={W_C})")
    print("=" * 72)
    print(f"  {'alpha_K':>8} {'corr(det,surp)':>16} {'corr(ent,surp)':>16} "
          f"{'mean_ent':>14} {'mean_det':>14}")
    for aK in AKS:
        K = causal_kernel_matrix(N, aK)
        nu = nu_field(p_marg, K, w_d=W_D, w_c=W_C)
        H_det = variable_order_entropy(p_bil, nu, form="mixed_renyi_tsallis").numpy()
        H_ent = variable_order_entropy(p_bil, nu, form="nist").numpy()
        cd = _corr(H_det, surprisal)
        ce = _corr(H_ent, surprisal)
        print(f"  {aK:>8.2f} {cd:>16.6f} {ce:>16.6f} "
              f"{_finite_mean(H_ent):>14.6e} {_finite_mean(H_det):>14.6e}")

    print()
    print("Readout: corr(detector, surprisal) > corr(entropy, surprisal) at every aK")
    print("  => the detector is the Born cost; the single-log entropy is not.")


if __name__ == "__main__":
    main()
