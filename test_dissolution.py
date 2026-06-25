"""
Phase 4: Group F — Zero-divisor dissolution tests.

F1  Zero divisors exist in H_C
F2  Order-of-operations dissolution (central test)
F3  KL divergence vs control
"""

import os, csv
import torch
import numpy as np
import pytest

from cal.biquaternion import (
    quat_mul, quat_inverse, biquat_norm, CDTYPE,
)
from cal.propagator import causal_propagator, transport

SEED = 42
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)


def _make_zero_divisor_pair(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct idempotent pair p_k, q_k for family k ∈ {1,2,3}.

    p_k = (1 + h·e_k)/2,  q_k = (1 − h·e_k)/2
    where h = √−1 ∈ C and e_k ∈ {i, j, k}.
    """
    h = 1j
    p = torch.zeros(4, dtype=CDTYPE)
    q = torch.zeros(4, dtype=CDTYPE)
    p[0] = 0.5
    q[0] = 0.5
    p[k] = h * 0.5    # +h·e_k / 2
    q[k] = -h * 0.5   # −h·e_k / 2
    return p, q


def _random_biquaternions(n, seed):
    torch.manual_seed(seed)
    real = torch.randn(n, 4, dtype=torch.float32)
    imag = torch.randn(n, 4, dtype=torch.float32)
    return (real + 1j * imag).to(CDTYPE)


class TestF1:
    """Zero divisors exist in H_C."""

    def test_zero_divisor_pairs(self):
        for k in [1, 2, 3]:
            p, q = _make_zero_divisor_pair(k)
            pq = quat_mul(p.unsqueeze(0), q.unsqueeze(0))[0]
            assert biquat_norm(pq).item() < 1e-6, (
                f"Family {k}: ||p·q|| = {biquat_norm(pq).item():.2e}")

    def test_idempotent(self):
        for k in [1, 2, 3]:
            p, q = _make_zero_divisor_pair(k)
            pp = quat_mul(p.unsqueeze(0), p.unsqueeze(0))[0]
            qq = quat_mul(q.unsqueeze(0), q.unsqueeze(0))[0]
            assert biquat_norm(pp - p).item() < 1e-6, (
                f"Family {k}: ||p²−p|| = {biquat_norm(pp-p).item():.2e}")
            assert biquat_norm(qq - q).item() < 1e-6, (
                f"Family {k}: ||q²−q|| = {biquat_norm(qq-q).item():.2e}")


class TestF2:
    """Order-of-operations dissolution test."""

    def test_dissolution(self):
        torch.manual_seed(SEED)
        from cal.biquaternion import biquat_to_matrix, matrix_to_biquat

        p, q = _make_zero_divisor_pair(1)  # Use family 1
        n_trials = 10000
        epsilons = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
        n_steps = 20

        # Step 1: Control — R · (p·q) · R⁻¹ should stay zero
        R = _random_biquaternions(n_trials, SEED)
        pq = quat_mul(p.unsqueeze(0).expand(n_trials, 4),
                       q.unsqueeze(0).expand(n_trials, 4))
        R_inv = quat_inverse(R)
        control = quat_mul(quat_mul(R, pq), R_inv)
        control_norms = biquat_norm(control)
        mu_control = control_norms.mean().item()
        sigma_control = control_norms.std().item()
        threshold = mu_control + 6 * sigma_control

        assert mu_control < 1e-4, f"Control mean = {mu_control:.2e}"

        rows = []
        for eps in epsilons:
            # Step 2: Path A — multiply first, then transport
            r0 = pq.clone()  # p·q = 0
            r_mat = biquat_to_matrix(r0)
            for step in range(n_steps):
                X = _random_biquaternions(n_trials, SEED + step * 1000)
                P = causal_propagator(X, eps)
                r_mat = transport(r_mat, P)
            path_a_norms = biquat_norm(matrix_to_biquat(r_mat))
            path_a_max = path_a_norms.max().item()

            # Step 3: Path B — transport first, then multiply
            p_mat = biquat_to_matrix(p.unsqueeze(0).expand(n_trials, 4))
            q_mat = biquat_to_matrix(q.unsqueeze(0).expand(n_trials, 4))
            for step in range(n_steps):
                X_p = _random_biquaternions(n_trials, SEED + step * 2000)
                X_q = _random_biquaternions(n_trials, SEED + step * 3000)
                P_p = causal_propagator(X_p, eps)
                P_q = causal_propagator(X_q, eps)
                p_mat = transport(p_mat, P_p)
                q_mat = transport(q_mat, P_q)

            p_transported = matrix_to_biquat(p_mat)
            q_transported = matrix_to_biquat(q_mat)
            product = quat_mul(p_transported, q_transported)
            path_b_norms = biquat_norm(product)
            path_b_mean = path_b_norms.mean().item()
            path_b_std = path_b_norms.std().item()

            effective_threshold = max(threshold, 1e-4)
            frac_dissolved = (path_b_norms > effective_threshold).float().mean().item()

            rows.append([eps, n_trials, path_a_max, path_b_mean,
                         path_b_std, frac_dissolved, effective_threshold])

            assert path_a_max < 1e-4, (
                f"eps={eps}: Path A max = {path_a_max:.2e}")
            assert frac_dissolved > 0.99, (
                f"eps={eps}: fraction dissolved = {frac_dissolved:.4f}")

        path = os.path.join(RESULTS, 'F2_dissolution.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epsilon', 'n_trials', 'path_A_max', 'path_B_mean',
                         'path_B_std', 'fraction_dissolved', 'threshold'])
            w.writerows(rows)


class TestF3:
    """KL divergence between dissolution and control distributions."""

    def test_kl_monotonic(self):
        torch.manual_seed(SEED)
        from cal.biquaternion import biquat_to_matrix, matrix_to_biquat

        p, q = _make_zero_divisor_pair(1)
        n_trials = 5000
        n_steps = 20
        epsilons = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

        # Control distribution
        R = _random_biquaternions(n_trials, SEED)
        pq = quat_mul(p.unsqueeze(0).expand(n_trials, 4),
                       q.unsqueeze(0).expand(n_trials, 4))
        R_inv = quat_inverse(R)
        control = quat_mul(quat_mul(R, pq), R_inv)
        control_norms = biquat_norm(control).clamp(min=1e-15)
        log_control = torch.log10(control_norms)

        kl_values = []
        for eps in epsilons:
            p_mat = biquat_to_matrix(p.unsqueeze(0).expand(n_trials, 4))
            q_mat = biquat_to_matrix(q.unsqueeze(0).expand(n_trials, 4))
            for step in range(n_steps):
                X_p = _random_biquaternions(n_trials, SEED + step * 2000 + int(eps * 10000))
                X_q = _random_biquaternions(n_trials, SEED + step * 3000 + int(eps * 10000))
                P_p = causal_propagator(X_p, eps)
                P_q = causal_propagator(X_q, eps)
                p_mat = transport(p_mat, P_p)
                q_mat = transport(q_mat, P_q)

            product = quat_mul(matrix_to_biquat(p_mat), matrix_to_biquat(q_mat))
            path_b_norms = biquat_norm(product).clamp(min=1e-15)
            log_pathb = torch.log10(path_b_norms)

            # KL via histogram binning (50 bins)
            lo = min(log_control.min().item(), log_pathb.min().item()) - 1
            hi = max(log_control.max().item(), log_pathb.max().item()) + 1
            bins = torch.linspace(lo, hi, 51)
            hist_c = torch.histogram(log_control, bins=bins).hist.float() + 1
            hist_b = torch.histogram(log_pathb, bins=bins).hist.float() + 1
            p_dist = hist_b / hist_b.sum()
            q_dist = hist_c / hist_c.sum()
            kl = (p_dist * torch.log(p_dist / q_dist)).sum().item()
            kl_values.append(kl)

        # Save results
        path = os.path.join(RESULTS, 'F3_kl_divergence.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epsilon', 'kl_divergence'])
            for eps, kl in zip(epsilons, kl_values):
                w.writerow([eps, kl])

        # Assert generally monotonic (allow 5% tolerance)
        for i in range(1, len(kl_values)):
            assert kl_values[i] >= kl_values[i-1] * 0.95, (
                f"KL not monotonic: eps={epsilons[i]}: {kl_values[i]:.4f} < "
                f"0.95 * {kl_values[i-1]:.4f}")
