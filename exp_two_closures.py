"""
Test the two-closure distinction (the proposed V.B fix) BEFORE writing it into the paper.

Claims to verify against the actual cal conjugations:
  reduced norm   N(q) = q*qbar   (qbar = quat_conj: negate vector, keep complex coeffs)
      C1: q*qbar is a PURE SCALAR equal to quat_norm_sq(q) = det(X_q).
      C2: MULTIPLICATIVE: N(q r) = N(q) N(r) to machine precision.
      C5a: VANISHES on zero divisors (null cone).
  Hermitian closure  <q qdagger>_0   (qdagger = hermitian_conj: conjugate coeffs AND vector)
      C3: <q qdagger>_0 = |z0|^2+...+|z3|^2 is REAL, >= 0, and = 0 only for q = 0.
      C4: NOT multiplicative (exhibit a counterexample).
      C5b: STAYS POSITIVE on the same zero divisor (complementarity: metric degenerates on the
           null cone, Born weight does not).
  Dagger vs bar on the privileged complex imaginary h:
      C6: h_dagger = -h  (dagger conjugates h)  while  h_bar = +h  (bar leaves it).
      C7: the dagger closure <W Wdagger>_0 is REAL >= 0 (interference gone), while the bar
          closure <W Wbar>_0 = N(W) is COMPLEX (interference survives) -- this is why Born
          uses the dagger, not the reduced norm.
"""
import torch
from cal.biquaternion import (quat_mul, quat_conj, hermitian_conj, quat_norm_sq,
                              biquat_to_matrix, quat_inverse, CDTYPE)

torch.manual_seed(0)


def rand_biquat(n):
    return (torch.randn(n, 4) + 1j * torch.randn(n, 4)).to(CDTYPE)


def reduced_norm_full(q):
    return quat_mul(q, quat_conj(q))            # q * qbar (full biquaternion)


def born_full(q):
    return quat_mul(q, hermitian_conj(q))       # q * qdagger (full biquaternion)


def det_mat(q):
    X = biquat_to_matrix(q)
    return X[..., 0, 0] * X[..., 1, 1] - X[..., 0, 1] * X[..., 1, 0]


def ok(b):
    return "PASS" if b else "**FAIL**"


def main():
    q = rand_biquat(200)
    r = rand_biquat(200)

    # C1: q*qbar is pure scalar = quat_norm_sq = det
    qqbar = reduced_norm_full(q)
    N = quat_norm_sq(q)
    vec_zero = torch.allclose(qqbar[..., 1:], torch.zeros_like(qqbar[..., 1:]), atol=1e-4)
    scal_eq = torch.allclose(qqbar[..., 0], N, atol=1e-4)
    det_eq = torch.allclose(det_mat(q), N, atol=1e-4)
    print("=" * 78)
    print("Two-closure distinction (proposed V.B fix) -- empirical test")
    print("=" * 78)
    print(f"  C1  q*qbar pure scalar, = quat_norm_sq, = det(X_q):  "
          f"vec0={ok(vec_zero)}  scalar={ok(scal_eq)}  det={ok(det_eq)}")

    # C2: reduced norm multiplicative
    N_qr = quat_norm_sq(quat_mul(q, r))
    mult = torch.allclose(N_qr, N * quat_norm_sq(r), atol=1e-3, rtol=1e-4)
    print(f"  C2  N(q r) = N(q) N(r)  (multiplicative):           {ok(mult)}  "
          f"max|diff|={(N_qr - N*quat_norm_sq(r)).abs().max():.2e}")

    # C3: Born scalar real, >= 0, sum|z|^2, zero only at q=0
    b = born_full(q)[..., 0]
    sumsq = (q.real**2 + q.imag**2).sum(-1)
    real_pos = torch.allclose(b.imag, torch.zeros_like(b.imag), atol=1e-4) and bool((b.real >= -1e-5).all())
    eq_sumsq = torch.allclose(b.real, sumsq, atol=1e-4)
    print(f"  C3  <q qdag>_0 real & >=0 & = sum|z|^2:              "
          f"real_pos={ok(real_pos)}  =sum|z|^2={ok(eq_sumsq)}")

    # C4: Born NOT multiplicative
    b_qr = born_full(quat_mul(q, r))[..., 0].real
    b_q_b_r = born_full(q)[..., 0].real * born_full(r)[..., 0].real
    not_mult = not torch.allclose(b_qr, b_q_b_r, atol=1e-2, rtol=1e-2)
    print(f"  C4  <q qdag>_0 NOT multiplicative:                   {ok(not_mult)}  "
          f"mean|ratio-1|={((b_qr/b_q_b_r) - 1).abs().mean():.3f}")

    # C5: zero divisor  q0 = 1 + h*i  (h = 1j),  N = 1 + h^2 = 0
    z = torch.tensor([[1.0 + 0j, 1j, 0, 0]], dtype=CDTYPE)
    Nz = quat_norm_sq(z).item()
    bz = born_full(z)[..., 0].real.item()
    detz = det_mat(z).item()
    inv = quat_inverse(z)
    noninv = bool(torch.isinf(inv.abs()).any() or torch.isnan(inv.abs()).any())
    print(f"  C5  zero divisor z=(1, h, 0, 0):")
    print(f"        reduced norm N(z) = {Nz:.3e}   (VANISHES on null cone: {ok(abs(Nz) < 1e-5)})")
    print(f"        Born weight <z zdag>_0 = {bz:.3f}   (STAYS POSITIVE: {ok(bz > 0.5)})")
    print(f"        det(X_z) = {detz:.3e}   non-invertible: {ok(noninv)}")

    # C6: dagger vs bar on the privileged complex imaginary h = (1j,0,0,0)
    h = torch.tensor([[1j, 0, 0, 0]], dtype=CDTYPE)
    h_bar = quat_conj(h)[..., 0].item()
    h_dag = hermitian_conj(h)[..., 0].item()
    print(f"  C6  privileged imaginary h: h_bar = {h_bar:+.0f}  (=+h),  "
          f"h_dag = {h_dag:+.0f}  (=-h):  {ok(h_bar == 1j and h_dag == -1j)}")

    # C7: dagger closure real (interference gone); bar closure complex (interference survives)
    W = rand_biquat(200)
    bar0 = reduced_norm_full(W)[..., 0]      # <W Wbar>_0 = N(W), complex
    dag0 = born_full(W)[..., 0]              # <W Wdag>_0, real
    bar_complex = bool(bar0.imag.abs().max() > 1e-2)
    dag_real = bool(dag0.imag.abs().max() < 1e-4)
    print(f"  C7  <W Wbar>_0 COMPLEX (interference survives): {ok(bar_complex)} "
          f"(max|imag|={bar0.imag.abs().max():.2f});  "
          f"<W Wdag>_0 REAL (interference gone): {ok(dag_real)}")

    print()
    print("All PASS => the two-closure paragraph is correct as stated and safe to write into V.B.")


if __name__ == "__main__":
    main()
