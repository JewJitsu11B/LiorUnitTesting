"""
Biquaternion arithmetic on H_C = C ⊗ H ≅ M₂(C).

A biquaternion q = q₀ + q₁i + q₂j + q₃k with each qₙ ∈ C
is stored as a torch.complex64 tensor of shape (..., 4).

Equation-of-state embedding (paper Eq. 1):
    qₙ = Aₙ exp(aₙ) + Bₙ exp(i bₙ),  Aₙ, Bₙ, aₙ, bₙ ∈ R.

Multiplication follows the standard Hamilton rules with (+,−,−,−)
scalar structure (paper Eq. 3).
"""

import torch
from torch import Tensor

CDTYPE = torch.complex64   # biquaternion / matrix dtype
FDTYPE = torch.float32     # real-valued outputs (metrics, norms)


# ── Multiplication ─────────────────────────────────────────────────

def quat_mul(P: Tensor, Q: Tensor) -> Tensor:
    """Biquaternion product PQ.  Paper Eq. 3.

    (PQ)₀ = p₀q₀ − p₁q₁ − p₂q₂ − p₃q₃
    (PQ)₁ = p₀q₁ + p₁q₀ + p₂q₃ − p₃q₂
    (PQ)₂ = p₀q₂ − p₁q₃ + p₂q₀ + p₃q₁
    (PQ)₃ = p₀q₃ + p₁q₂ − p₂q₁ + p₃q₀
    """
    p0, p1, p2, p3 = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    q0, q1, q2, q3 = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]

    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0

    return torch.stack([r0, r1, r2, r3], dim=-1)


# ── Conjugation / projection ──────────────────────────────────────

def quat_conj(q: Tensor) -> Tensor:
    """Quaternion conjugate q̃ = q₀ − q₁i − q₂j − q₃k.
    Negates vector part; does NOT complex-conjugate components."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def hermitian_conj(q: Tensor) -> Tensor:
    """Hermitian conjugate q† = q̄₀ − q̄₁i − q̄₂j − q̄₃k.
    Complex-conjugates each component AND negates vector part.
    Paper Eq. 9."""
    qbar = q.conj()
    return torch.stack([qbar[..., 0], -qbar[..., 1],
                        -qbar[..., 2], -qbar[..., 3]], dim=-1)


def hermitian(q: Tensor) -> Tensor:
    """Hermitian projection H(q) = (q + q†)/2.  Paper Eq. 10.
    Real scalar component, purely imaginary vector components."""
    return (q + hermitian_conj(q)) / 2


def anti_hermitian(q: Tensor) -> Tensor:
    """Anti-Hermitian projection A(q) = (q − q†)/2.  Paper Eq. 10.
    Purely imaginary scalar, real vector components."""
    return (q - hermitian_conj(q)) / 2


# ── Norm / inverse ────────────────────────────────────────────────

def quat_norm_sq(q: Tensor) -> Tensor:
    """Quaternion norm squared N(q) = q₀² + q₁² + q₂² + q₃².
    Complex-valued (may vanish for zero-divisors)."""
    return q[..., 0]**2 + q[..., 1]**2 + q[..., 2]**2 + q[..., 3]**2


def biquat_norm(q: Tensor) -> Tensor:
    """Frobenius norm ||q|| = sqrt(Σ|qₙ|²).  Real-valued."""
    return torch.sqrt((q.real**2 + q.imag**2).sum(dim=-1)).to(FDTYPE)


def quat_inverse(q: Tensor) -> Tensor:
    """Multiplicative inverse q⁻¹ = q̃ / N(q).
    Undefined for zero-divisors (returns inf)."""
    N = quat_norm_sq(q).unsqueeze(-1)
    return quat_conj(q) / N


# ── M₂(C) representation ──────────────────────────────────────────

def biquat_to_matrix(q: Tensor) -> Tensor:
    """Convert biquaternion → 2×2 complex matrix.

    M(q) = [[q₀ + h q₃,   q₁ + h q₂],
            [−q₁ + h q₂,  q₀ − h q₃]]

    where h = √−1 ∈ C.  Shape (...,4) → (...,2,2).
    """
    h = 1j
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    row0 = torch.stack([q0 + h*q3,  q1 + h*q2], dim=-1)
    row1 = torch.stack([-q1 + h*q2, q0 - h*q3], dim=-1)
    return torch.stack([row0, row1], dim=-2).to(CDTYPE)


def matrix_to_biquat(M: Tensor) -> Tensor:
    """Convert 2×2 complex matrix → biquaternion.  Shape (...,2,2) → (...,4)."""
    h = 1j
    m00, m01, m10, m11 = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    q0 = (m00 + m11) / 2
    q3 = (m00 - m11) / (2 * h)
    q1 = (m01 - m10) / 2
    q2 = (m01 + m10) / (2 * h)
    return torch.stack([q0, q1, q2, q3], dim=-1).to(CDTYPE)


# ── Embedding ──────────────────────────────────────────────────────

def embed_token_channels(seed: int, n_components: int = 4) -> tuple[Tensor, Tensor]:
    """Embed one token, returning geometric and spectral channels separately.

    qₙ = Aₙ exp(aₙ) + Bₙ exp(i bₙ)
    geometric: Aₙ exp(aₙ)     — real, amplitude/energy, sources the metric
    spectral:  Bₙ exp(i bₙ)   — complex, phase/coherence, sources the connection

    Returns:
        geo:  (4,) complex64 — real-valued (imag ≈ 0)
        spec: (4,) complex64 — complex oscillatory
    """
    rng = torch.Generator().manual_seed(seed)
    params = torch.randn(n_components, 4, generator=rng, dtype=torch.float32)
    A, B, a, b = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    geo = (A * torch.exp(a)).to(CDTYPE)
    spec = (B * torch.exp(1j * b.to(torch.complex64))).to(CDTYPE)
    return geo, spec


def embed_token(seed: int, n_components: int = 4) -> Tensor:
    """Embed one token via equation-of-state form (paper Eq. 1).

    qₙ = Aₙ exp(aₙ) + Bₙ exp(i bₙ),  deterministic from seed.
    Returns shape (4,) complex64  (geometric + spectral combined).
    """
    geo, spec = embed_token_channels(seed, n_components)
    return geo + spec


def embed_tokens_channels(n_tokens: int, base_seed: int = 42) -> tuple[Tensor, Tensor]:
    """Embed n_tokens, returning both channels.

    Returns:
        geo:  (n_tokens, 4) complex64 — geometric channel
        spec: (n_tokens, 4) complex64 — spectral channel
    """
    geos, specs = [], []
    for i in range(n_tokens):
        g, s = embed_token_channels(base_seed + i)
        geos.append(g)
        specs.append(s)
    return torch.stack(geos), torch.stack(specs)


def embed_tokens(n_tokens: int, base_seed: int = 42) -> Tensor:
    """Embed n_tokens as biquaternions.  Shape (n_tokens, 4) complex64."""
    geo, spec = embed_tokens_channels(n_tokens, base_seed)
    return geo + spec


def embed_sentence_channels(words: list, base_seed: int = 1000) -> tuple[Tensor, Tensor]:
    """Embed word list, returning both channels."""
    geos, specs = [], []
    for w in words:
        seed = (hash(w) % (2**31)) ^ base_seed
        g, s = embed_token_channels(seed)
        geos.append(g)
        specs.append(s)
    return torch.stack(geos), torch.stack(specs)


def embed_sentence(words: list, base_seed: int = 1000) -> Tensor:
    """Embed word list with deterministic per-word seeds."""
    geo, spec = embed_sentence_channels(words, base_seed)
    return geo + spec
