"""
CAL stress-energy tensor construction.

present_tensor   — rank-4 tensor from pairwise interactions (Eq. 5-6)
cal_tensor       — full CAL: present + transported past (Eq. 14)
power_law_kernel — normalized memory weight w(τ) = (1+τ)^{-α_K}

The geometric channel (Ae^φ, real) sources the metric.
The spectral channel (Be^{iφ}, complex) sources the connection.
Both present_tensor and cal_tensor accept tokens from either channel.
"""

import math
import torch
from torch import Tensor
from .biquaternion import (
    quat_mul, quat_inverse, biquat_norm,
    CDTYPE, FDTYPE,
)


def power_law_kernel(N: int, alpha_K: float = 0.5) -> Tensor:
    """Normalized power-law memory weight.

    w(τ) = (1 + τ)^{-α_K},  τ = 1, ..., N-1,  normalized.
    Returns shape (N-1,) float32.
    """
    tau = torch.arange(1, N, dtype=FDTYPE)
    w = (1.0 + tau).pow(-alpha_K)
    return w / w.sum()


def _interaction_components(tokens: Tensor) -> Tensor:
    """Pairwise interaction biquaternions c_{xj} = P_x · P_j.

    Args:
        tokens: (N, 4) complex64.
    Returns:
        c: (N, N, 4) complex64.
    """
    N = tokens.shape[0]
    Px = tokens.unsqueeze(1).expand(N, N, 4)
    Pj = tokens.unsqueeze(0).expand(N, N, 4)
    return quat_mul(Px, Pj)


def _outer4(c: Tensor) -> Tensor:
    """Rank-4 outer product T[μ,ν,ρ,σ] = c[μ] c̄[ν] c[ρ] c̄[σ]."""
    cbar = c.conj()
    return (c[..., :, None, None, None]
            * cbar[..., None, :, None, None]
            * c[..., None, None, :, None]
            * cbar[..., None, None, None, :])


def present_tensor(tokens: Tensor, kT: float = 1.0) -> Tensor:
    """Build rank-4 stress-energy tensor from pairwise interactions (Eq. 5-6).

    Pass geometric-channel tokens for the metric tensor.
    Pass spectral-channel tokens for the connection tensor.
    Pass full (geo+spec) tokens for the combined tensor.

    Args:
        tokens: (N, 4) complex64.
        kT: informational temperature.
    Returns:
        T: (N, 4, 4, 4, 4) complex64.
    """
    N = tokens.shape[0]
    c = _interaction_components(tokens)
    T_all = _outer4(c)

    mask = ~torch.eye(N, dtype=torch.bool, device=tokens.device)
    T_all = T_all * mask[:, :, None, None, None, None]

    prefactor = kT * math.log(2) / max(N - 1, 1)
    return T_all.sum(dim=1) * prefactor


def cal_tensor(tokens: Tensor, alpha: float, kernel_fn=None,
               kT: float = 1.0) -> Tensor:
    """Full CAL stress-energy tensor (Eq. 14).

    T_CAL(x) = α · present + (1−α) · transported_past

    Pass geometric-channel tokens for metric, spectral for connection.

    Args:
        tokens: (N, 4) complex64.
        alpha: phase-angle interpolator in [0, 1].
        kernel_fn: callable(N) → (N-1,) weights.
        kT: informational temperature.
    Returns:
        T: (N, 4, 4, 4, 4) complex64.
    """
    N = tokens.shape[0]
    if kernel_fn is None:
        kernel_fn = power_law_kernel

    c = _interaction_components(tokens)

    # --- Present term ---
    T_present = _outer4(c)
    mask_self = ~torch.eye(N, dtype=torch.bool, device=tokens.device)
    T_present = T_present * mask_self[:, :, None, None, None, None]
    T_present = T_present.sum(dim=1)

    # --- Past (transported) term ---
    w = kernel_fn(N).to(tokens.device)
    Pinv = quat_inverse(tokens)

    T_past = torch.zeros(N, 4, 4, 4, 4, dtype=CDTYPE, device=tokens.device)

    for x in range(1, N):
        for s in range(x):
            tau = x - s
            if tau > len(w):
                continue
            wt = w[tau - 1]

            U = quat_mul(tokens[x:x+1], Pinv[s:s+1])
            U_norm = biquat_norm(U).clamp(min=1e-7)
            U = U / U_norm.unsqueeze(-1).to(CDTYPE)
            U_inv = quat_inverse(U)

            c_s = c[s]
            Uc = quat_mul(U.expand_as(c_s), c_s)
            c_transported = quat_mul(Uc, U_inv.expand_as(c_s))

            T_sj = _outer4(c_transported)
            T_sj[s] = 0
            T_past[x] += wt * T_sj.sum(dim=0)

    prefactor = kT * math.log(2) / max(N - 1, 1)
    return (alpha * T_present + (1 - alpha) * T_past) * prefactor
