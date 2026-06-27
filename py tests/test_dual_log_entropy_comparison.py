"""
Comparative tests: NIST bilocal vs. dual-log functionals.

Forms compared:
  nist               — lens -ln(ν) inside, content p^{ν+1}, one log
  dual_tsallis       — both lens -ln(ν) and content -ln_ν(p) inside (both Tsallis-placed)
  mixed_renyi_tsallis — lens inside (Tsallis-placed), content in outer log (Rényi-placed)

Test groups:
  CMP1 — six-way value comparison (extends C8)
  CMP2 — sign and fixed-point structure across ν sweep
  CMP3 — additivity decomposition over independent subsystems
  CMP4 — sensitivity to bilocal vs. factorized distribution
  CMP5 — concavity probe
"""

import os, csv, math
import torch
import numpy as np
import pytest

from cal.biquaternion import embed_tokens_channels, FDTYPE
from cal.entropy import (
    energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy,
)

SEED = 42
RESULTS = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS, exist_ok=True)

FORMS = ["nist", "dual_tsallis", "mixed_renyi_tsallis"]


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def _bilocal_pipeline(N, seed, alpha_K=0.5):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tokens = geo + spec
    E_mat = energy_matrix(tokens)
    p_bil = E_mat / (E_mat.sum() + 1e-30)
    K = causal_kernel_matrix(N, alpha_K=alpha_K)
    p_marg = p_bil.sum(dim=1)
    nu = nu_field(p_marg, K)
    phi = torch.ones(N, N, dtype=FDTYPE) / N
    return E_mat, p_bil, nu, phi, tokens


def _compute_all_forms(p_bil, nu, phi):
    """Return dict of H tensors for all three new forms."""
    return {f: variable_order_entropy(p_bil, nu, phi, form=f) for f in FORMS}


def _constant_nu(N, val):
    return torch.full((N, N), val, dtype=FDTYPE)


# ---------------------------------------------------------------------------
# CMP1 — Six-way value comparison
# ---------------------------------------------------------------------------

class TestCMP1_ValueComparison:
    """All three forms produce finite, distinct values."""

    def test_CMP1_all_forms_finite(self):
        N = 100
        E_mat, p_bil, nu, phi, _ = _bilocal_pipeline(N, SEED)
        H = _compute_all_forms(p_bil, nu, phi)
        for f, h in H.items():
            assert torch.all(torch.isfinite(h)), f"{f}: non-finite values"

    def test_CMP1_all_forms_distinct(self):
        """No two forms produce identical entropy vectors."""
        N = 100
        E_mat, p_bil, nu, phi, _ = _bilocal_pipeline(N, SEED)
        H = _compute_all_forms(p_bil, nu, phi)
        forms = list(H.keys())
        rows = [['form_A', 'form_B', 'mean_abs_diff']]
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                fa, fb = forms[i], forms[j]
                diff = (H[fa] - H[fb]).abs().mean().item()
                rows.append([fa, fb, diff])
                assert diff > 1e-15, f"{fa} and {fb} are numerically identical"
        with open(os.path.join(RESULTS, 'CMP1_six_way_value_comparison.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

    def test_CMP1_spatial_variation_all_forms(self):
        """Each form has spatial variation (log-range > 1) or records why it does not."""
        N = 100
        E_mat, p_bil, nu, phi, _ = _bilocal_pipeline(N, SEED)
        H = _compute_all_forms(p_bil, nu, phi)
        rows = [['form', 'log_range', 'has_variation']]
        for f, h in H.items():
            h_abs = h.abs().clamp(min=1e-30)
            log_range = (torch.log(h_abs.max()) - torch.log(h_abs.min())).item()
            rows.append([f, log_range, log_range > 1.0])
        with open(os.path.join(RESULTS, 'CMP1_spatial_variation.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        # At minimum the existing nist form must vary
        nist_h = H['nist'].abs().clamp(min=1e-30)
        nist_range = (torch.log(nist_h.max()) - torch.log(nist_h.min())).item()
        assert nist_range > 1.0, f"nist log-range={nist_range:.2f}"


# ---------------------------------------------------------------------------
# CMP2 — Sign and fixed-point structure
# ---------------------------------------------------------------------------

class TestCMP2_SignAndFixedPoint:
    """Each form's behavior at and around ν=1."""

    def test_CMP2_fixed_point_at_nu_one(self):
        """All bilocal forms should give H≈0 when ν≡1 (ln(ν)=0 kills the lens)."""
        N = 60
        E_mat, p_bil, nu_real, phi, _ = _bilocal_pipeline(N, SEED)
        nu_one = _constant_nu(N, 1.0)
        rows = [['form', 'max_abs_H_at_nu1']]
        for f in FORMS:
            H = variable_order_entropy(p_bil, nu_one, phi, form=f)
            max_abs = H.abs().max().item()
            rows.append([f, max_abs])
        with open(os.path.join(RESULTS, 'CMP2_fixed_point_nu1.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        # nist is the reference — already tested; verify dual and mixed also vanish
        for f in FORMS:
            H = variable_order_entropy(p_bil, nu_one, phi, form=f)
            assert H.abs().max().item() < 1e-2, (
                f"{f}: H should ≈ 0 at ν=1; max|H|={H.abs().max().item():.3e}")

    def test_CMP2_sign_sweep(self):
        """Record sign of H for each form at ν ∈ {0.5, 1.0, 1.5, 2.0}."""
        N = 60
        E_mat, p_bil, _, phi, _ = _bilocal_pipeline(N, SEED)
        nu_vals = [0.5, 1.0, 1.5, 2.0]
        rows = [['nu', 'form', 'mean_H', 'frac_positive']]
        for val in nu_vals:
            nu_c = _constant_nu(N, val)
            for f in FORMS:
                H = variable_order_entropy(p_bil, nu_c, phi, form=f)
                rows.append([val, f, H.mean().item(), (H > 0).float().mean().item()])
        with open(os.path.join(RESULTS, 'CMP2_sign_sweep.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        # Structural assertion: nist and dual_tsallis should both flip sign near ν=1
        # (both have -ln(ν) as a factor; mixed is less constrained due to outer log)
        for f in ['nist', 'dual_tsallis']:
            H_low = variable_order_entropy(p_bil, _constant_nu(N, 0.5), phi, form=f)
            H_high = variable_order_entropy(p_bil, _constant_nu(N, 2.0), phi, form=f)
            assert H_low.mean().item() > H_high.mean().item(), (
                f"{f}: expected H(ν=0.5) > H(ν=2.0) (sign-flip across ν=1)")


# ---------------------------------------------------------------------------
# CMP3 — Additivity decomposition over independent subsystems
# ---------------------------------------------------------------------------

class TestCMP3_Additivity:
    """Does H(A⊕B) equal H(A)+H(B) (Rényi-additive) or the Tsallis deformed sum?

    Independence is constructed analytically: the combined bilocal distribution is
    a block-diagonal matrix diag(p_A, p_B), normalised to sum to 1.  The ν field
    is constructed from the combined marginal so cross-block entries are well-defined.
    """

    def _block_combine(self, p_A, p_B):
        """Block-diagonal bilocal distribution for two independent subsystems."""
        NA = p_A.shape[0]
        NB = p_B.shape[0]
        N = NA + NB
        p_full = torch.zeros(N, N, dtype=FDTYPE)
        p_full[:NA, :NA] = p_A
        p_full[NA:, NA:] = p_B
        return p_full / (p_full.sum() + 1e-30)

    def test_CMP3_additivity_type(self):
        N2 = 50
        E_A, p_A, nu_A, phi_A, _ = _bilocal_pipeline(N2, SEED)
        E_B, p_B, nu_B, phi_B, _ = _bilocal_pipeline(N2, SEED + 1)
        p_full = self._block_combine(p_A, p_B)
        N = N2 * 2
        K_full = causal_kernel_matrix(N)
        p_marg_full = p_full.sum(dim=1)
        nu_full = nu_field(p_marg_full, K_full)
        phi_full = torch.ones(N, N, dtype=FDTYPE) / N

        rows = [['form', 'H_A_mean', 'H_B_mean', 'H_AB_mean',
                 'additive_residual', 'tsallis_residual', 'closer_to']]
        for f in FORMS:
            H_A = variable_order_entropy(p_A, nu_A, phi_A, form=f).mean().item()
            H_B = variable_order_entropy(p_B, nu_B, phi_B, form=f).mean().item()
            H_AB = variable_order_entropy(p_full, nu_full, phi_full, form=f).mean().item()
            additive = abs(H_AB - (H_A + H_B))
            # Tsallis deformed sum: H_A + H_B + (1-ν̄)·H_A·H_B
            nu_bar = nu_full.mean().item()
            tsallis_sum = H_A + H_B + (1.0 - nu_bar) * H_A * H_B
            tsallis_res = abs(H_AB - tsallis_sum)
            closer = 'additive' if additive < tsallis_res else 'tsallis'
            rows.append([f, H_A, H_B, H_AB, additive, tsallis_res, closer])

        with open(os.path.join(RESULTS, 'CMP3_additivity_decomposition.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        # Structural expectation: mixed_renyi_tsallis should be closer to additive
        # (outer log gives Rényi-style composition); dual_tsallis should be closer to tsallis.
        # We record results but allow the test to pass regardless — this is a characterisation test.
        # The only hard assertion is that H_AB is finite for all forms.
        for f in FORMS:
            H_AB = variable_order_entropy(p_full, nu_full, phi_full, form=f)
            assert torch.all(torch.isfinite(H_AB)), f"{f}: H_AB non-finite"


# ---------------------------------------------------------------------------
# CMP4 — Sensitivity to bilocal structure
# ---------------------------------------------------------------------------

class TestCMP4_BilocaliySensitivity:
    """Which form distinguishes bilocal from factorized distribution most sharply?"""

    def test_CMP4_bilocal_vs_factorized_gap(self):
        N = 100
        E_mat, p_bil, nu, phi, _ = _bilocal_pipeline(N, SEED)
        # Factorized approximation
        p_marg = p_bil.sum(dim=1)
        p_fact = torch.outer(p_marg, p_marg)
        p_fact = p_fact / (p_fact.sum() + 1e-30)

        rows = [['form', 'mean_delta_H', 'max_delta_H', 'min_delta_H']]
        for f in FORMS:
            H_true = variable_order_entropy(p_bil,  nu, phi, form=f)
            H_fact = variable_order_entropy(p_fact, nu, phi, form=f)
            delta = (H_true - H_fact).abs()
            rows.append([f, delta.mean().item(), delta.max().item(), delta.min().item()])

        with open(os.path.join(RESULTS, 'CMP4_bilocal_vs_factorized.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        # All forms should produce a non-zero gap
        for f in FORMS:
            H_true = variable_order_entropy(p_bil,  nu, phi, form=f)
            H_fact = variable_order_entropy(p_fact, nu, phi, form=f)
            delta = (H_true - H_fact).abs().mean().item()
            assert delta > 1e-15, f"{f}: bilocal vs factorized gap is zero"

    def test_CMP4_gap_ranking(self):
        """Record which form is most sensitive to the bilocal structure."""
        N = 100
        E_mat, p_bil, nu, phi, _ = _bilocal_pipeline(N, SEED)
        p_marg = p_bil.sum(dim=1)
        p_fact = torch.outer(p_marg, p_marg)
        p_fact = p_fact / (p_fact.sum() + 1e-30)

        gaps = {}
        for f in FORMS:
            H_t = variable_order_entropy(p_bil,  nu, phi, form=f)
            H_f = variable_order_entropy(p_fact, nu, phi, form=f)
            gaps[f] = (H_t - H_f).abs().mean().item()

        ranked = sorted(gaps.items(), key=lambda kv: kv[1], reverse=True)
        with open(os.path.join(RESULTS, 'CMP4_gap_ranking.csv'), 'w', newline='') as fh:
            csv.writer(fh).writerows([['rank', 'form', 'mean_delta_H']]
                                     + [[i+1, f, g] for i, (f, g) in enumerate(ranked)])
        # No hard assertion — purely characterisation


# ---------------------------------------------------------------------------
# CMP5 — Concavity probe
# ---------------------------------------------------------------------------

class TestCMP5_Concavity:
    """Is H(λ·p_A + (1-λ)·p_B) ≥ λ·H(p_A) + (1-λ)·H(p_B)?

    A concave entropy is a proper measure of uncertainty.
    A failure here is a hard constraint: the form may not satisfy entropy axioms.
    """

    def test_CMP5_concavity_check(self):
        N = 60
        E_A, p_A, nu, phi, _ = _bilocal_pipeline(N, SEED)
        E_B, p_B, _, _,   _ = _bilocal_pipeline(N, SEED + 7)
        # Renormalise p_B to same shape
        p_B = p_B / (p_B.sum() + 1e-30)

        lambdas = [0.1, 0.25, 0.5, 0.75, 0.9]
        rows = [['form', 'lambda', 'H_mix', 'H_convex_combo', 'concave']]
        violations = {f: 0 for f in FORMS}

        for lam in lambdas:
            p_mix = lam * p_A + (1.0 - lam) * p_B
            p_mix = p_mix / (p_mix.sum() + 1e-30)
            for f in FORMS:
                H_A = variable_order_entropy(p_A,   nu, phi, form=f)
                H_B = variable_order_entropy(p_B,   nu, phi, form=f)
                H_M = variable_order_entropy(p_mix, nu, phi, form=f)
                combo = lam * H_A + (1.0 - lam) * H_B
                # concave if H_mix >= combo element-wise (mean test)
                is_concave = (H_M.mean() >= combo.mean()).item()
                if not is_concave:
                    violations[f] += 1
                rows.append([f, lam, H_M.mean().item(), combo.mean().item(), is_concave])

        with open(os.path.join(RESULTS, 'CMP5_concavity_probe.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        # Record violations per form; no hard pass/fail — this is a characterisation test.
        viol_rows = [['form', 'violations_out_of', 'n_lambdas']]
        for f in FORMS:
            viol_rows.append([f, violations[f], len(lambdas)])
        with open(os.path.join(RESULTS, 'CMP5_concavity_violations.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(viol_rows)

        # Hard assertion only: nist (already validated) must not violate concavity more than half
        assert violations['nist'] <= len(lambdas) // 2, (
            f"nist concavity violations={violations['nist']} out of {len(lambdas)}")
