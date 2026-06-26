"""
Test B: closed-loop CAL field test WITH TIME (worldline), Born-cost 5-way at the fixed point.
Built on the verified holonomy foundation (exp_holonomy_detlift.py) and the fixed detector
(Hermitian-norm Renyi, exp_hermitian_norm_renyi.py).

Loop at alpha=2; positions = causal worldline (x'<x is past):
  Psi -> p_H (Hermitian-norm density = energy_matrix) -> H(detector) -> J -> T=2J+past -> Psi'
  J_x   = exp(-Hz_x/tau)*Hermitian(Psi_x) + exp(i S_x)*AntiHermitian(Psi_x)   [user's J choice]
  S_x   = holonomy winding angle = arg det(U_x), U_x = path-ordered propagator product (the TIME axis)
  past_x= sum_{x'<x} k(x-x') * holonomy-transport(J_{x'} -> x)   [relative holonomy R=U_x U_x'^-1]
  T = 2J + past; Psi' = T / ||T||.  Iterate to fixed point; record det (off the null cone).

At the fixed point: corr(functional, surprisal=-ln p_marg) for the 5 conditions
  1 shannon(row)  2 fixed-order(nist const nu)  3 variable-order entropy(nist)
  4a lens detector (mixed_renyi_tsallis, the OLD/degenerate form)
  4b Hermitian-norm Renyi detector (the FIX; standard Renyi on p_H, special-cased at nu_bar~1)
  5 both = multiple-R of (3) and (4b).

SIMPLIFICATIONS (flagged): the causal-past integral is the holonomy-transported k(tau)-weighted
sum; the log-space Bayesian rank-2-forcing refinement is a further layer not added here. The
arrow / integration-order non-commutator is a SEPARATE measurement, not in this script.
"""
import math
import numpy as np
import torch
from cal.biquaternion import (embed_tokens, hermitian, anti_hermitian, biquat_to_matrix,
                              matrix_to_biquat, biquat_norm, CDTYPE)
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy
from cal.cal_tensor import power_law_kernel
from cal.propagator import causal_propagator

torch.manual_seed(0)
N = 60
EPS, TAU = 0.2, 1.0
W_D, W_C, A_K = 0.1, 0.3, 1.0
MAX_ITER, TOL = 120, 1e-4
LAM = 0.3   # under-relaxation (damping) to find a fixed point of the oscillating map


def det2(M):
    return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]


def _corr(a, b):
    a = np.nan_to_num(np.asarray(a, float), posinf=0, neginf=0)
    b = np.nan_to_num(np.asarray(b, float), posinf=0, neginf=0)
    s = a.std() * b.std()
    return float(np.corrcoef(a, b)[0, 1]) if s > 1e-30 else 0.0


def _mR(r1, r2, r12):
    d = 1 - r12 ** 2
    return float(np.sqrt(max((r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * r12) / d, 0.0))) if abs(d) > 1e-9 else float("nan")


def nu_bar_of(nu, p):
    pr = p / p.sum(1, keepdim=True).clamp(min=1e-30)
    return (nu * pr).sum(1)


def detector_herm(p, nu):
    """FIX: standard Renyi on the Hermitian-norm probability, special-cased at nu_bar~1 = Shannon."""
    pr = p / p.sum(1, keepdim=True).clamp(min=1e-30)
    pc = pr.clamp(min=1e-12)
    nb = (nu * pr).sum(1)
    Z = torch.exp(nu * torch.log(pc)).sum(1)
    H = torch.log(Z.clamp(min=1e-30)) / (1.0 - nb)
    sh = -(pc * torch.log(pc)).sum(1)
    return torch.where((1.0 - nb).abs() < 1e-3, sh, H).numpy()


def step(Psi, K, k_tau, phi):
    E = energy_matrix(Psi); E = E.real if torch.is_complex(E) else E
    p = E / (E.sum() + 1e-30); p_marg = p.sum(1)
    nu = nu_field(p_marg, K, w_d=W_D, w_c=W_C)
    H = variable_order_entropy(p, nu, phi, form="mixed_renyi_tsallis")
    Hz = (H - H.mean()) / (H.std() + 1e-9)
    P = causal_propagator(Psi, EPS)
    U = torch.zeros(N, 2, 2, dtype=CDTYPE); U[0] = torch.eye(2, dtype=CDTYPE)
    for x in range(1, N):
        U[x] = P[x - 1] @ U[x - 1]
    Uinv = torch.linalg.inv(U)
    S = torch.angle(det2(U))                                   # holonomy winding angle
    Gp, Ap = hermitian(Psi), anti_hermitian(Psi)
    J = (torch.exp(-Hz / TAU).to(CDTYPE).unsqueeze(-1) * Gp
         + torch.exp(1j * S.to(CDTYPE)).unsqueeze(-1) * Ap)
    M_J = biquat_to_matrix(J)
    past_M = torch.zeros(N, 2, 2, dtype=CDTYPE)
    for x in range(1, N):
        acc = torch.zeros(2, 2, dtype=CDTYPE)
        for xp in range(x):
            R = U[x] @ Uinv[xp]; Rinv = U[xp] @ Uinv[x]
            acc = acc + k_tau[x - xp - 1] * (R @ M_J[xp] @ Rinv)
        past_M[x] = acc
    T = 2.0 * J + matrix_to_biquat(past_M)
    Tn = T / biquat_norm(T).clamp(min=1e-9).unsqueeze(-1).to(CDTYPE)
    return Tn, p, p_marg, nu, past_M


def main():
    Psi = embed_tokens(N)
    Psi = Psi / biquat_norm(Psi).clamp(min=1e-9).unsqueeze(-1).to(CDTYPE)
    K = causal_kernel_matrix(N, A_K); k_tau = power_law_kernel(N); phi = torch.ones(N, N) / N

    print("=" * 94)
    print(f"Test B: closed-loop field (alpha=2, with worldline/time)  N={N}")
    print("=" * 94)
    conv, min_det_past_all = False, float("inf")
    it = 0
    for it in range(MAX_ITER):
        Tn, p, p_marg, nu, past_M = step(Psi, K, k_tau, phi)
        Psi_new = (1.0 - LAM) * Psi + LAM * Tn
        Psi_new = Psi_new / biquat_norm(Psi_new).clamp(min=1e-9).unsqueeze(-1).to(CDTYPE)
        diff = biquat_norm(Psi_new - Psi).mean().item()
        mdpast = det2(past_M[1:]).abs().min().item()
        min_det_past_all = min(min_det_past_all, mdpast)
        if it % 20 == 0 or diff < TOL:
            print(f"  iter {it:3d}: ||dPsi||={diff:.2e}  min|det(past)|={mdpast:.2e}")
        Psi = Psi_new
        if diff < TOL:
            conv = True; break
    print(f"  -> {'CONVERGED' if conv else 'NO fixed point (map did not contract)'} after {it+1} iters; "
          f"min|det(past)| over run = {min_det_past_all:.2e} (det>0 => off the null cone)")

    # 5-way at the fixed point
    surp = -np.log(p_marg.numpy() + 1e-30)
    pc = p.clamp(min=1e-12)
    H_sh = (-(pc * torch.log(pc)).sum(1)).numpy()
    nb_mean = float(nu_bar_of(nu, p).mean())
    H_fix = variable_order_entropy(p, torch.full((N, N), nb_mean), phi, form="nist").numpy()
    H_var = variable_order_entropy(p, nu, phi, form="nist").numpy()
    H_lens = variable_order_entropy(p, nu, phi, form="mixed_renyi_tsallis").numpy()
    H_herm = detector_herm(p, nu)
    r = {"1 shannon": _corr(H_sh, surp), "2 fixed-order": _corr(H_fix, surp),
         "3 var-entropy": _corr(H_var, surp), "4a lens detector(OLD)": _corr(H_lens, surp),
         "4b Herm-Renyi detector(FIX)": _corr(H_herm, surp)}
    R_both = _mR(r["3 var-entropy"], r["4b Herm-Renyi detector(FIX)"], _corr(H_var, H_herm))

    print()
    print(f"  fixed-point nu_bar(mean) = {nb_mean:.4f}   (detector evaluated away from the nu=1 degeneracy)")
    print(f"  {'condition':>30}  corr(.,surprisal)")
    for k, v in r.items():
        print(f"  {k:>30}  {v:+.4f}")
    print(f"  {'5 both (multiple-R of 3 & 4b)':>30}  {R_both:.4f}")


if __name__ == "__main__":
    main()
