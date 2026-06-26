"""
Two-sided limit test of the entropy and detector functionals as nu_bar -> 1.

Sitting at the field's natural nu_bar (~1, the Shannon point) hides the structure.
Here we DIAL nu_bar directly by using a uniform nu-field (nu_bar = the uniform value),
and sweep it on BOTH sides of 1 to test the limits:

  - ENTROPY  H^nu  (form="nist"):  expected sign-flipping ZERO at nu_bar = 1
                                   (H > 0 for nu_bar < 1, H = 0 at 1, H < 0 for nu_bar > 1).
  - DETECTOR H_hyb (form="mixed_renyi_tsallis"): expected POLE at nu_bar = 1, via the
                                   1/(1 - nu_bar) Renyi prefactor.

We report three things per nu_bar:
  entropy            : the single-log entropy mean
  detector (code)    : the cal implementation, with (1 - nu_bar).clamp(min=1e-6)
  detector (raw)     : the same formula WITHOUT the clamp, to expose the true two-sided limit

so the clamp's effect near nu_bar = 1 is visible rather than hidden.
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import energy_matrix, variable_order_entropy

torch.manual_seed(0)
N = 140


def _field(seed=42):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tok = geo + spec
    E = energy_matrix(tok)
    E = E.real if torch.is_complex(E) else E
    return E / (E.sum() + 1e-30)            # (N,N) bilocal density


def detector_raw(p_bil, nu, phi):
    """The mixed_renyi_tsallis detector WITHOUT the (1-nu_bar) clamp."""
    log_nu = torch.log(nu.clamp(min=1e-4))
    p_c = p_bil.clamp(min=1e-10)
    p_pow = torch.exp(nu * torch.log(p_c))
    integrand = (p_pow * (-log_nu) * phi).sum(dim=-1)
    p_phi = (p_c * phi).sum(dim=-1).clamp(min=1e-10)
    nu_bar = (nu * p_c * phi).sum(dim=-1) / p_phi
    sign = torch.sign(integrand)
    log_abs = torch.log(integrand.abs().clamp(min=1e-10))
    return sign * log_abs / (1.0 - nu_bar)          # no clamp


def main():
    p_bil = _field()
    phi = torch.ones(N, N, dtype=torch.float32) / N

    offsets = [0.5, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-5]
    nubars = sorted([1.0 - o for o in offsets] + [1.0 + o for o in offsets])

    print("=" * 84)
    print(f"Two-sided limit nu_bar -> 1   (N={N}, uniform nu-field = nu_bar)")
    print("=" * 84)
    print(f"  {'nu_bar':>9} {'1-nu_bar':>11} {'entropy':>14} {'detector(code)':>16} {'detector(raw)':>16}")
    for nb in nubars:
        nu = torch.full((N, N), float(nb), dtype=torch.float32)
        H_ent = variable_order_entropy(p_bil, nu, form="nist").numpy()
        H_det_c = variable_order_entropy(p_bil, nu, form="mixed_renyi_tsallis").numpy()
        H_det_r = detector_raw(p_bil, nu, phi).numpy()
        print(f"  {nb:>9.5f} {1.0 - nb:>11.1e} {np.mean(H_ent):>14.4e} "
              f"{np.mean(H_det_c):>16.4e} {np.mean(H_det_r):>16.4e}")

    print()
    print("Two-sided readout:")
    print("  entropy : sign-flipping ZERO at nu_bar=1 (>0 below, <0 above) if signs flip.")
    print("  detector(raw): even/odd POLE at nu_bar=1 (magnitude grows both sides);")
    print("                 compare sign on the two sides to classify odd vs even.")
    print("  detector(code): clamp at |1-nu_bar|<1e-6 caps the pole (not reached in this grid).")


if __name__ == "__main__":
    main()
