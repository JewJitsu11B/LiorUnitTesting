"""
Causal propagator and transport for biquaternions in M₂(C).

causal_propagator — P_X(ε) = exp(ε · ad(X)_ah)
transport         — Y^X = P Y P⁻¹
biquat_to_matrix  — re-exported from biquaternion module
matrix_to_biquat  — re-exported from biquaternion module
"""

import torch
from torch import Tensor
from .biquaternion import (
    biquat_to_matrix, matrix_to_biquat, quat_mul, quat_inverse,
    biquat_norm, CDTYPE,
)


def _matrix_log(M: Tensor) -> Tensor:
    """Matrix logarithm for 2×2 complex matrices via eigendecomposition."""
    evals, evecs = torch.linalg.eig(M)
    log_evals = torch.log(evals)
    return evecs @ torch.diag_embed(log_evals) @ torch.linalg.inv(evecs)


def _anti_hermitian_part(M: Tensor) -> Tensor:
    """Anti-Hermitian part: (M − M†)/2."""
    return (M - M.conj().transpose(-2, -1)) / 2


def causal_propagator(X: Tensor, epsilon: float) -> Tensor:
    """Causal propagator P_X(ε) = exp(ε · ad(X)_ah).

    X is a biquaternion (shape (..., 4)).
    Returns the propagator as a 2×2 complex matrix (..., 2, 2).

    ad(X)_ah is the anti-Hermitian part of log(X / sqrt(det X)).
    """
    M = biquat_to_matrix(X)
    det_M = torch.linalg.det(M)
    sqrt_det = torch.sqrt(det_M)
    # Normalize: M / sqrt(det) has unit determinant
    # Avoid division by zero for singular matrices
    sqrt_det_safe = torch.where(sqrt_det.abs() < 1e-10,
                                 torch.ones_like(sqrt_det), sqrt_det)
    M_norm = M / sqrt_det_safe.unsqueeze(-1).unsqueeze(-1)
    log_M = _matrix_log(M_norm)
    Y_ah = _anti_hermitian_part(log_M)
    P = torch.linalg.matrix_exp(epsilon * Y_ah)
    return P


def transport(Y_matrix: Tensor, P: Tensor) -> Tensor:
    """Conjugation transport: Y^X = P Y P⁻¹.

    Args:
        Y_matrix: (..., 2, 2) matrix to transport.
        P: (..., 2, 2) propagator.
    Returns:
        (..., 2, 2) transported matrix.
    """
    P_inv = torch.linalg.inv(P)
    return P @ Y_matrix @ P_inv


def transport_biquat(Y: Tensor, X: Tensor, epsilon: float) -> Tensor:
    """Transport biquaternion Y through the causal field of X at coupling ε.

    Args:
        Y: (..., 4) biquaternion to transport.
        X: (..., 4) connection biquaternion.
        epsilon: coupling strength.
    Returns:
        (..., 4) transported biquaternion.
    """
    P = causal_propagator(X, epsilon)
    Y_mat = biquat_to_matrix(Y)
    transported = transport(Y_mat, P)
    return matrix_to_biquat(transported)
