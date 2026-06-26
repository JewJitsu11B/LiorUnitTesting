"""
Does feeding the detector the BORN-point field change the surprisal tracking?

At alpha=2 (where Born closure happens) theta = alpha*pi/2 = pi, so the spectral phase
factor is e^{i pi} = -1: the spectral channel is REAL and ANTIPARALLEL, W = geo - |spec|.
Every prior experiment used the off-Born generic field geo + spec (random complex phase).
Here we compare corr(detector/entropy, surprisal) for:

   generic : geo + spec            (random complex spectral phase)   <- what I used
   born    : geo - |spec|          (alpha=2, spectral phase = -1)     <- the Born condition
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy

torch.manual_seed(0)
N = 140
W_D, W_C = 0.1, 0.3


def fields(seed=42):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tok_generic = geo + spec
    tok_born = geo - spec.abs().to(geo.dtype)          # spectral phase e^{i pi} = -1
    return tok_generic, tok_born


def _p_surp(tok):
    E = energy_matrix(tok)
    E = E.real if torch.is_complex(E) else E
    p = E / (E.sum() + 1e-30)
    p_marg = p.sum(dim=1)
    surp = -np.log(p_marg.numpy() + 1e-30)
    return p, p_marg, surp


def _nu_bar_mean(nu, p, phi):
    p_c = p.clamp(min=1e-10)
    p_phi = (p_c * phi).sum(dim=-1).clamp(min=1e-10)
    return float(((nu * p_c * phi).sum(dim=-1) / p_phi).mean())


def _corr(a, b):
    a = np.nan_to_num(np.asarray(a, float), posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, float), posinf=0.0, neginf=0.0)
    return float(np.corrcoef(a, b)[0, 1])


def run(tok, label):
    p, p_marg, surp = _p_surp(tok)
    phi = torch.ones(N, N, dtype=torch.float32) / N
    print(f"  {label} field:")
    print(f"    {'alpha_K':>8} {'nu_bar':>9} {'corr(detector)':>16} {'corr(entropy)':>16}")
    for aK in [0.5, 1.0, 1.5, 2.0]:
        K = causal_kernel_matrix(N, aK)
        nu = nu_field(p_marg, K, w_d=W_D, w_c=W_C)
        H_det = variable_order_entropy(p, nu, form="mixed_renyi_tsallis").numpy()
        H_ent = variable_order_entropy(p, nu, form="nist").numpy()
        print(f"    {aK:>8.2f} {_nu_bar_mean(nu, p, phi):>9.4f} "
              f"{_corr(H_det, surp):>16.4f} {_corr(H_ent, surp):>16.4f}")


def main():
    tok_g, tok_b = fields()
    print("=" * 64)
    print("Detector/entropy correlation with surprisal: generic vs Born input")
    print("=" * 64)
    run(tok_g, "GENERIC (geo + spec, random complex)")
    run(tok_b, "BORN (geo - |spec|, alpha=2 spec=-1)")


if __name__ == "__main__":
    main()
