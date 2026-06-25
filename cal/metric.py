"""
Metric extraction from rank-4 stress-energy tensor.

extract_metric               — double contraction g_μν = Σ_ρ T[μ,ρ,ν,ρ]
hermitian_project_metric     — basis change to Hermitian subspace (negates spatial block)
anti_hermitian_project_metric — basis change to anti-Hermitian subspace (negates temporal)
signature                    — (n_pos, n_neg) from eigenvalues
scalar_curvature             — Eq. 8
"""

import torch
from torch import Tensor
from .biquaternion import FDTYPE


def extract_metric(T: Tensor) -> Tensor:
    """Double contraction g_μν = Σ_ρ T[μ,ρ,ν,ρ], symmetrize.

    Args:
        T: (..., 4, 4, 4, 4) complex.
    Returns:
        g: (..., 4, 4) float32 — real symmetric metric.
    """
    g = torch.einsum('...ibjb->...ij', T)
    g = g.real.to(FDTYPE)
    g = 0.5 * (g + g.transpose(-2, -1))
    return g


def hermitian_project_metric(g: Tensor) -> Tensor:
    """Project metric to Hermitian subspace via D = diag(1, h, h, h).

    The Hermitian biquaternion basis is {1, hi, hj, hk}. Changing from
    the component basis {1, i, j, k} to this basis multiplies spatial
    indices by h = √−1, giving h² = −1 on the spatial block:

        g_H[0,0] = g[0,0]       (timelike, unchanged)
        g_H[k,l] = −g[k,l]      (spatial block negated, h² = −1)
        g_H[0,k] = g_H[k,0] = 0 (cross terms vanish under Re)

    Args:
        g: (..., 4, 4) float32.
    Returns:
        g_H: (..., 4, 4) float32 with Lorentzian signature.
    """
    g_H = torch.zeros_like(g)
    g_H[..., 0, 0] = g[..., 0, 0]
    g_H[..., 1:, 1:] = -g[..., 1:, 1:]
    return g_H


def anti_hermitian_project_metric(g: Tensor) -> Tensor:
    """Project metric to anti-Hermitian subspace.

    Anti-Hermitian basis is {h, i, j, k} (imaginary scalar, real vector).
    Changing basis: temporal index gets h, spatial unchanged:

        g_A[0,0] = −g[0,0]      (temporal negated)
        g_A[k,l] = g[k,l]       (spatial unchanged)
        g_A[0,k] = g_A[k,0] = 0

    Args:
        g: (..., 4, 4) float32.
    Returns:
        g_A: (..., 4, 4) float32.
    """
    g_A = torch.zeros_like(g)
    g_A[..., 0, 0] = -g[..., 0, 0]
    g_A[..., 1:, 1:] = g[..., 1:, 1:]
    return g_A


def signature(g: Tensor, rtol: float = 1e-6) -> tuple[int, int]:
    """Compute metric signature (n_pos, n_neg) from eigenvalues."""
    evals = torch.linalg.eigvalsh(g.to(torch.float64))
    threshold = rtol * evals.abs().max()
    n_pos = int((evals > threshold).sum().item())
    n_neg = int((evals < -threshold).sum().item())
    return (n_pos, n_neg)


def signature_batch(g: Tensor, rtol: float = 1e-6) -> list[tuple[int, int]]:
    """Compute signature for each metric in a batch."""
    return [signature(g[i], rtol) for i in range(g.shape[0])]


def scalar_curvature(g: Tensor, T: Tensor) -> Tensor:
    """Scalar curvature per Eq. 8.

    R_sc = sqrt(|g^{μρ} g^{νσ} T_{μνρσ} / n²|) + ε
    """
    eps = 1e-12
    n = 4
    g64 = g.to(torch.float64)
    g_inv = torch.linalg.inv(g64 + eps * torch.eye(4, dtype=torch.float64,
                                                     device=g.device))
    T_real = T.real.to(torch.float64)
    contracted = torch.einsum('...mp,...ns,...mnrs->...', g_inv, g_inv, T_real)
    R_sc = torch.sqrt(torch.abs(contracted) / (n * n) + eps)
    return R_sc.to(FDTYPE)
