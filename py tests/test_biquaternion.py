"""
Phase 1 unit tests for biquaternion arithmetic.

U1 — Noncommutativity
U2 — Exact Hermitian decomposition
U3 — Hermitian structure (real scalar, imaginary vector)
Plus algebraic sanity checks and M₂(C) round-trip.
"""

import torch
import pytest
from cal.biquaternion import (
    quat_mul, hermitian, anti_hermitian, hermitian_conj, quat_conj,
    quat_inverse, biquat_norm, quat_norm_sq,
    embed_token, embed_tokens, embed_sentence,
    biquat_to_matrix, matrix_to_biquat,
    CDTYPE, FDTYPE,
)

SEED = 42
N = 1000
TOL = 1e-6


def _random_biquaternions(n: int, seed: int = SEED) -> torch.Tensor:
    torch.manual_seed(seed)
    real = torch.randn(n, 4, dtype=torch.float32)
    imag = torch.randn(n, 4, dtype=torch.float32)
    return (real + 1j * imag).to(CDTYPE)


# ── U1: Noncommutativity ──────────────────────────────────────────

class TestU1:
    def test_noncommutative_fraction(self):
        """At least 99% of random pairs satisfy ||PQ − QP|| > 0.1."""
        torch.manual_seed(SEED)
        P = _random_biquaternions(N, seed=SEED)
        Q = _random_biquaternions(N, seed=SEED + 1)

        PQ = quat_mul(P, Q)
        QP = quat_mul(Q, P)
        comm = biquat_norm(PQ - QP)

        frac = (comm > 0.1).float().mean().item()
        assert frac >= 0.99, f"{frac:.1%} non-commutative (need ≥99%)"


# ── U2: Hermitian decomposition ───────────────────────────────────

class TestU2:
    def test_decomposition_exact(self):
        """||q − H(q) − A(q)|| < 1e-6 for 1000 random q."""
        q = _random_biquaternions(N)
        residual = biquat_norm(q - hermitian(q) - anti_hermitian(q))
        assert residual.max().item() < TOL, (
            f"Residual {residual.max().item():.2e}")


# ── U3: Hermitian structure ───────────────────────────────────────

class TestU3:
    def test_scalar_is_real(self):
        """Im(H(q)[0]) < 1e-6 for 1000 random q."""
        q = _random_biquaternions(N)
        H = hermitian(q)
        assert H[..., 0].imag.abs().max().item() < TOL

    def test_vector_is_purely_imaginary(self):
        """Re(H(q)[1:]) < 1e-6 for 1000 random q."""
        q = _random_biquaternions(N)
        H = hermitian(q)
        assert H[..., 1:].real.abs().max().item() < TOL


# ── Algebraic sanity ──────────────────────────────────────────────

class TestAlgebra:
    def test_associativity(self):
        """(PQ)R = P(QR) for H_C."""
        P = _random_biquaternions(200, seed=SEED)
        Q = _random_biquaternions(200, seed=SEED + 1)
        R = _random_biquaternions(200, seed=SEED + 2)
        left = quat_mul(quat_mul(P, Q), R)
        right = quat_mul(P, quat_mul(Q, R))
        # Triple product accumulates float32 error; relax to 1e-4
        assert biquat_norm(left - right).max().item() < 1e-4

    def test_basis_squares(self):
        """i² = j² = k² = −1."""
        neg1 = torch.tensor([-1, 0, 0, 0], dtype=CDTYPE)
        for idx in range(1, 4):
            e = torch.zeros(4, dtype=CDTYPE)
            e[idx] = 1.0
            assert biquat_norm(quat_mul(e, e) - neg1).item() < 1e-7

    def test_ijk_neg_one(self):
        """ijk = −1."""
        i = torch.tensor([0, 1, 0, 0], dtype=CDTYPE)
        j = torch.tensor([0, 0, 1, 0], dtype=CDTYPE)
        k = torch.tensor([0, 0, 0, 1], dtype=CDTYPE)
        neg1 = torch.tensor([-1, 0, 0, 0], dtype=CDTYPE)
        ijk = quat_mul(quat_mul(i, j), k)
        assert biquat_norm(ijk - neg1).item() < 1e-7

    def test_inverse(self):
        """q · q⁻¹ ≈ 1 for generic elements."""
        q = _random_biquaternions(200)
        one = torch.zeros(200, 4, dtype=CDTYPE)
        one[:, 0] = 1.0
        product = quat_mul(q, quat_inverse(q))
        assert biquat_norm(product - one).max().item() < 1e-4


# ── M₂(C) round-trip ─────────────────────────────────────────────

class TestMatrix:
    def test_roundtrip(self):
        q = _random_biquaternions(200)
        q2 = matrix_to_biquat(biquat_to_matrix(q))
        assert biquat_norm(q - q2).max().item() < TOL

    def test_mul_consistency(self):
        """quat_mul matches matrix multiplication in M₂(C)."""
        P = _random_biquaternions(200, seed=SEED)
        Q = _random_biquaternions(200, seed=SEED + 1)
        pq_quat = quat_mul(P, Q)
        Mp, Mq = biquat_to_matrix(P), biquat_to_matrix(Q)
        pq_mat = matrix_to_biquat(torch.bmm(Mp, Mq))
        assert biquat_norm(pq_quat - pq_mat).max().item() < 1e-4


# ── Embedding ─────────────────────────────────────────────────────

class TestEmbed:
    def test_deterministic(self):
        assert torch.allclose(embed_token(42), embed_token(42))

    def test_different_seeds(self):
        assert biquat_norm(embed_token(42) - embed_token(43)).item() > 0.01

    def test_shape(self):
        t = embed_tokens(6)
        assert t.shape == (6, 4)
        assert t.dtype == CDTYPE

    def test_sentence(self):
        s = embed_sentence(["the", "cat", "sat"])
        assert s.shape == (3, 4)
