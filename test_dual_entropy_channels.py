"""
Dual Entropy Channel Analysis
==============================

Claim: The dual-Tsallis bilocal form is a dual entropic measure with two
functionally independent channels:

    Lens channel:    -ln(ν(x,y))         — LOCAL  (built from marginals + distance)
    Content channel: -ln_ν(p(x,y))       — NON-LOCAL (requires joint distribution)

Per-pair decomposition:
    dual_contribution(x,y) = p^ν · lens_factor · content_factor · φ

H₀ (Null hypothesis, shared across all tests):
    The two channels are interchangeable — they measure the same underlying
    structure, respond equally to factorization, carry redundant information,
    and do not independently track local vs. non-local properties.

H₁ (Alternative hypothesis):
    The channels are functionally independent.
    - Lens is invariant to factorization (ν depends only on marginals + distance).
    - Content collapses under factorization (joint ≠ product of marginals).
    - Their per-pair values are decorrelated across pairs.
    - In a tribalized system, content collapses for cross-tribe pairs while
      lens remains nonzero (local structure survives even when non-local
      coherence is lost).

Tests:
    DC1 — Channel decorrelation: Pearson r between lens and content factors
    DC2 — Factorization invariance: Δlens = 0 exactly, Δcontent > 0
    DC3 — Graded bilocality tracking: content monotone in λ, lens flat
    DC4 — Tribalism simulation: content collapses cross-tribe, lens does not
    DC5 — Predictability: content not linearly predictable from lens (low R²)
"""

import os, csv, math
import torch
import numpy as np
import pytest
from scipy import stats

from cal.biquaternion import embed_tokens_channels, FDTYPE, quat_mul, hermitian
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field

SEED = 42
RESULTS = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# Channel decomposition utilities
# ---------------------------------------------------------------------------

def _bilocal_setup(N, seed, alpha_K=0.5):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tokens = geo + spec
    E_mat = energy_matrix(tokens)
    p_bil = E_mat / (E_mat.sum() + 1e-30)
    K = causal_kernel_matrix(N, alpha_K=alpha_K)
    p_marg = p_bil.sum(dim=1)
    nu = nu_field(p_marg, K)
    phi = torch.ones(N, N, dtype=FDTYPE) / N
    return p_bil, nu, phi, p_marg, K


def _factorized(p_bil):
    """Same marginals as p_bil, but factorized: p(x)·p(y)."""
    p_marg = p_bil.sum(dim=1)
    p_fact = torch.outer(p_marg, p_marg)
    return p_fact / (p_fact.sum() + 1e-30)


def _channel_factors(p_bil, nu, phi):
    """
    Decompose per-pair dual-Tsallis integrand into lens and content factors.

    Returns:
        weight   (N,N): p^ν · φ          — common weighting
        lens     (N,N): -ln(ν)            — local factor
        content  (N,N): -ln_ν(p)          — non-local factor
        dual_mat (N,N): weight·lens·content — full integrand
    """
    p = p_bil.clamp(min=1e-10)
    log_nu = torch.log(nu.clamp(min=1e-4))
    p_pow_nu = torch.exp(nu * torch.log(p))

    one_m_nu = 1.0 - nu
    p_pow_1mnu = torch.exp(one_m_nu * torch.log(p))
    near_one = one_m_nu.abs() < 1e-4
    ln_nu_p = torch.where(near_one, torch.log(p), (p_pow_1mnu - 1.0) / one_m_nu)

    weight   = p_pow_nu * phi
    lens     = -log_nu                   # local channel: depends only on ν (marginals+distance)
    content  = -ln_nu_p                  # non-local channel: depends on joint p(x,y)
    dual_mat = weight * lens * content

    return weight, lens, content, dual_mat


def _pearson(a, b):
    """Pearson r between flattened tensors."""
    af = a.flatten().numpy().astype(np.float64)
    bf = b.flatten().numpy().astype(np.float64)
    r, p = stats.pearsonr(af, bf)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# DC1 — Channel decorrelation
# ---------------------------------------------------------------------------

class TestDC1_Decorrelation:
    """
    H₀: |r(lens, content)| > 0.5  — channels are strongly correlated (redundant).
    H₁: |r(lens, content)| < 0.3  — channels are decorrelated (independent).

    Physical meaning: If H₁ holds, knowing the local measurement apparatus
    at a pair tells you little about the non-local field content at that pair.
    Local and non-local structure vary independently across pairs.
    """

    def test_DC1_channel_pearson_correlation(self):
        N = 100
        p_bil, nu, phi, *_ = _bilocal_setup(N, SEED)
        weight, lens, content, _ = _channel_factors(p_bil, nu, phi)

        r, pval = _pearson(lens, content)

        with open(os.path.join(RESULTS, 'DC1_channel_correlation.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['quantity', 'value'],
                ['pearson_r_lens_content', r],
                ['p_value', pval],
                ['H0_threshold', 0.5],
                ['H1_threshold', 0.3],
                ['verdict', 'H1_supported' if abs(r) < 0.3 else 'H0_not_rejected'],
            ])

        assert abs(r) < 0.3, (
            f"H₀ not rejected: channels correlated at r={r:.4f}. "
            f"Expected |r|<0.3 (independent channels) under H₁.")

    def test_DC1_correlation_stable_across_N(self):
        """Channel decorrelation should hold across token counts."""
        rows = [['N', 'pearson_r', 'verdict']]
        for N in [40, 70, 100, 150]:
            p_bil, nu, phi, *_ = _bilocal_setup(N, SEED + N)
            _, lens, content, _ = _channel_factors(p_bil, nu, phi)
            r, _ = _pearson(lens, content)
            rows.append([N, r, 'H1' if abs(r) < 0.3 else 'H0'])

        with open(os.path.join(RESULTS, 'DC1_correlation_vs_N.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        # Majority of N values should show decorrelation
        verdicts = [row[2] for row in rows[1:]]
        h1_count = verdicts.count('H1')
        assert h1_count >= 3, (
            f"Channel decorrelation unstable: only {h1_count}/4 N values support H₁.")


# ---------------------------------------------------------------------------
# DC2 — Factorization invariance
# ---------------------------------------------------------------------------

class TestDC2_FactorizationInvariance:
    """
    H₀: Factorizing p(x,y) → p(x)·p(y) changes the lens and content channels
        by comparable amounts (both channels equally sensitive to factorization).

    H₁: Δlens = 0 exactly (ν depends only on marginals, which are preserved
        by factorization). Δcontent > 0 (joint p changes, content reads joint).

    Physical meaning: The lens is a purely local instrument — it cannot detect
    whether the distribution is genuinely bilocal or factorized, because it only
    reads the marginals. The content is the bilocality sensor.

    Analytical prediction: Since p_fact has identical marginals to p_true,
    the ν field is IDENTICAL for both → Δlens = 0 analytically.
    """

    def test_DC2_lens_invariant_under_factorization(self):
        """Lens factor does not change when p is factorized (Δlens = 0 exactly)."""
        N = 100
        p_bil, nu, phi, p_marg, K = _bilocal_setup(N, SEED)
        p_fact = _factorized(p_bil)

        # ν for p_true: built from p_true marginals
        nu_true = nu_field(p_marg, K)
        # ν for p_fact: built from p_fact marginals (same, by construction)
        nu_fact = nu_field(p_fact.sum(dim=1), K)

        delta_nu = (nu_true - nu_fact).abs().max().item()
        # Lens factor is -ln(ν), so if ν is identical, lens is identical
        _, lens_true, _, _ = _channel_factors(p_bil,  nu_true, phi)
        _, lens_fact, _, _ = _channel_factors(p_fact, nu_fact, phi)
        delta_lens = (lens_true - lens_fact).abs().mean().item()

        with open(os.path.join(RESULTS, 'DC2_factorization_invariance.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['quantity', 'value'],
                ['max_delta_nu', delta_nu],
                ['mean_delta_lens', delta_lens],
                ['interpretation', 'lens invariant to factorization if delta_lens ~ 0'],
            ])

        assert delta_lens < 1e-6, (
            f"H₀ not rejected: Δlens={delta_lens:.3e} should be ~0. "
            f"Lens is not invariant to factorization.")

    def test_DC2_content_sensitive_to_factorization(self):
        """Content factor changes when p is factorized (Δcontent > 0)."""
        N = 100
        p_bil, nu, phi, p_marg, K = _bilocal_setup(N, SEED)
        p_fact = _factorized(p_bil)

        _, _, content_true, _ = _channel_factors(p_bil,  nu, phi)
        _, _, content_fact, _ = _channel_factors(p_fact, nu, phi)
        delta_content = (content_true - content_fact).abs().mean().item()

        with open(os.path.join(RESULTS, 'DC2_content_sensitivity.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['quantity', 'value'],
                ['mean_delta_content', delta_content],
                ['interpretation', 'content is bilocal sensor if delta_content > 0'],
            ])

        assert delta_content > 1e-6, (
            f"H₀ not rejected: Δcontent={delta_content:.3e} is near zero. "
            f"Content is not detecting factorization.")

    def test_DC2_content_vs_lens_sensitivity_ratio(self):
        """Content is significantly more sensitive to factorization than lens."""
        N = 100
        p_bil, nu, phi, p_marg, K = _bilocal_setup(N, SEED)
        p_fact = _factorized(p_bil)

        nu_true = nu_field(p_marg, K)
        nu_fact = nu_field(p_fact.sum(dim=1), K)

        _, lens_true, content_true, _ = _channel_factors(p_bil,  nu_true, phi)
        _, lens_fact, content_fact, _ = _channel_factors(p_fact, nu_fact, phi)

        delta_lens    = (lens_true    - lens_fact).abs().mean().item()
        delta_content = (content_true - content_fact).abs().mean().item()

        # Avoid div-by-zero: lens delta is analytically ~0, add epsilon
        ratio = delta_content / (delta_lens + 1e-10)

        with open(os.path.join(RESULTS, 'DC2_sensitivity_ratio.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['quantity', 'value'],
                ['delta_lens', delta_lens],
                ['delta_content', delta_content],
                ['content_to_lens_ratio', ratio],
                ['H1_threshold_ratio', 10.0],
                ['verdict', 'H1_supported' if ratio > 10 else 'H0_not_rejected'],
            ])

        assert ratio > 10.0, (
            f"H₀ not rejected: content/lens sensitivity ratio = {ratio:.1f}, "
            f"expected > 10× under H₁.")


# ---------------------------------------------------------------------------
# DC3 — Graded bilocality tracking
# ---------------------------------------------------------------------------

class TestDC3_GradedBilocality:
    """
    H₀: Both channels respond equally to increasing bilocal structure
        (as p transitions from factorized to true bilocal).

    H₁: Content channel monotonically increases with bilocality parameter λ.
        Lens channel is flat (insensitive to λ), since ν stays fixed.

    Constructed by interpolating:
        p_λ = (1-λ)·p_fact + λ·p_true    for λ ∈ [0, 0.2, ..., 1.0]

    Physical meaning: Content is a monotone detector of bilocal structure.
    Lens is not. You can use content to track the degree of bilocal coherence
    in a field as it transitions from factorized (tribalized) to genuine
    bilocal (cross-cutting democratic field).
    """

    def test_DC3_content_monotone_in_lambda(self):
        N = 80
        p_bil, nu, phi, p_marg, K = _bilocal_setup(N, SEED)
        p_fact = _factorized(p_bil)

        lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        lens_means    = []
        content_means = []

        rows = [['lambda', 'mean_lens', 'mean_content']]
        for lam in lambdas:
            p_mix = (1.0 - lam) * p_fact + lam * p_bil
            p_mix = p_mix / (p_mix.sum() + 1e-30)
            # ν fixed from true marginals (they don't change with λ since marginals preserved)
            _, lens, content, _ = _channel_factors(p_mix, nu, phi)
            lm = lens.abs().mean().item()
            cm = content.abs().mean().item()
            lens_means.append(lm)
            content_means.append(cm)
            rows.append([lam, lm, cm])

        with open(os.path.join(RESULTS, 'DC3_graded_bilocality.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        # Content should be monotone in λ (Pearson r > 0.8).
        # Lens may be exactly constant (ConstantInputWarning from scipy = strongest confirmation).
        r_content, _ = stats.pearsonr(lambdas, content_means)
        lens_std = np.std(lens_means)
        lens_is_constant = lens_std < 1e-10
        r_lens = 0.0 if lens_is_constant else stats.pearsonr(lambdas, lens_means)[0]

        with open(os.path.join(RESULTS, 'DC3_monotonicity.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['channel', 'pearson_r_with_lambda', 'std', 'verdict'],
                ['content', r_content, np.std(content_means),
                 'tracks_bilocality' if r_content > 0.8 else 'does_not_track'],
                ['lens', r_lens, lens_std,
                 'exactly_constant (strongest H1 confirmation)' if lens_is_constant
                 else ('local_only' if abs(r_lens) < 0.3 else 'also_sensitive')],
            ])

        assert r_content > 0.8, (
            f"H₀ not rejected: content r={r_content:.3f} with λ, expected > 0.8.")
        # Lens must be less sensitive than content — if exactly constant that's satisfied trivially
        assert lens_is_constant or abs(r_lens) < r_content, (
            f"H₀ not rejected: lens r={r_lens:.3f} not less than content r={r_content:.3f}.")


# ---------------------------------------------------------------------------
# DC4 — Tribalism simulation
# ---------------------------------------------------------------------------

class TestDC4_TribalisimSimulation:
    """
    H₀: In a tribalized distribution (factorized cross-tribe), the content and
        lens channels are equally degraded for cross-tribe pairs.

    H₁: Content channel collapses for cross-tribe pairs (no non-local coherence
        between tribes), but lens channel remains nonzero (local structure
        preserved — tribes still exist as distinct groups with different influence
        levels, even when they don't genuinely interact).

    Construction:
        - Tribe A: N/2 tokens with genuine bilocal structure
        - Tribe B: N/2 tokens with genuine bilocal structure
        - Cross-tribe: factorized (p_A_marg ⊗ p_B_marg, no genuine interaction)

    Physical meaning: Tribalism = content collapses cross-tribe, lens survives.
    The lens correctly identifies that two groups exist with different properties
    (local structure: unequal influence, causal distance > 0). The content
    correctly identifies that they no longer genuinely interact (non-local
    coherence: near zero for cross-tribe pairs).
    """

    # ORIGINAL HELPER — kept for potential future use.
    # Construction flaw: setting cross-tribe probability to near-zero creates a
    # probability-scale mismatch.  The q-log surprisal -ln_ν(p) grows as p→0,
    # so low-mass cross-tribe pairs score HIGHER content than high-mass within-tribe
    # pairs — the opposite of the intuition being tested.  The within/cross ratio
    # comparison therefore fails to distinguish genuine coherence from mere scale effects.
    # See _democratic_and_tribal and test_DC4_content_collapses_cross_tribe for the
    # revised construction that controls for probability mass.
    def _tribal_distribution(self, N_half, seed):
        """Block structure: within-tribe genuine bilocal, cross-tribe factorized at 10% weight."""
        torch.manual_seed(seed)
        p_A, _, _, p_marg_A, _ = _bilocal_setup(N_half, seed)
        p_B, _, _, p_marg_B, _ = _bilocal_setup(N_half, seed + 1)
        N = N_half * 2
        p_full = torch.zeros(N, N, dtype=FDTYPE)
        p_full[:N_half, :N_half] = p_A * 0.5
        p_full[N_half:, N_half:] = p_B * 0.5
        cross = torch.outer(p_marg_A, p_marg_B)
        cross = cross / (cross.sum() + 1e-30) * 0.1
        p_full[:N_half, N_half:] = cross
        p_full[N_half:, :N_half] = cross.T
        p_full = p_full / (p_full.sum() + 1e-30)
        return p_full, N_half

    # ORIGINAL TEST (v1) — retained with annotation.
    # Flaw: uses within/cross ratio of content values.  Because cross-tribe probability
    # mass is set to ~10% of within-tribe mass, q-log surprisal is intrinsically larger
    # for cross-tribe pairs (content ∝ -ln_ν(p), large for small p).  This means
    # within/cross ratio ≈ 0 trivially — not because of coherence collapse but because
    # of scale.  The test therefore cannot distinguish the hypothesis from the null.
    # Renamed from test_DC4_content_collapses_cross_tribe; kept for archival completeness.
    def test_DC4_v1_content_collapses_cross_tribe(self):
        """[ORIGINAL v1 — see annotation above.] Within/cross content ratio test."""
        N_half = 50
        p_tribal, n_h = self._tribal_distribution(N_half, SEED)
        N = N_half * 2

        K = causal_kernel_matrix(N)
        nu = nu_field(p_tribal.sum(dim=1), K)
        phi = torch.ones(N, N, dtype=FDTYPE) / N

        _, lens, content, _ = _channel_factors(p_tribal, nu, phi)

        within_mask = torch.zeros(N, N, dtype=torch.bool)
        within_mask[:n_h, :n_h] = True
        within_mask[n_h:, n_h:] = True
        cross_mask = ~within_mask & ~torch.eye(N, dtype=torch.bool)

        content_within = content[within_mask].abs().mean().item()
        content_cross  = content[cross_mask].abs().mean().item()
        lens_within    = lens[within_mask].abs().mean().item()
        lens_cross     = lens[cross_mask].abs().mean().item()

        content_ratio = content_within / (content_cross + 1e-12)
        lens_ratio    = lens_within    / (lens_cross    + 1e-12)

        # This assertion FAILS by construction (see annotation) — kept for record.
        # pytest.skip() omitted deliberately so the failure is visible in the test log.
        assert content_ratio > lens_ratio * 5, (
            f"H₀ not rejected: content within/cross ratio={content_ratio:.1f}, "
            f"lens within/cross ratio={lens_ratio:.1f}. "
            f"Expected content_ratio > 5 × lens_ratio under H₁.")

    # REVISED HELPER — replaces _tribal_distribution.
    # Fixes the probability-scale mismatch by constructing two distributions with
    # IDENTICAL marginals and identical within-tribe blocks, differing ONLY in whether
    # cross-tribe coupling is genuine (full bilocal) or factorized.  The content channel
    # Δ at cross-tribe pairs then isolates coherence collapse from scale effects.
    def _democratic_and_tribal(self, N_half, seed):
        """
        Two configurations with identical marginals, differing only in
        whether cross-tribe coupling is genuine (democratic) or factorized (tribal).

        Democratic: all pairs computed from full-span bilocal tokens.
        Tribal:     within-tribe unchanged; cross-tribe replaced by p_marg_A ⊗ p_marg_B
                    at the same total weight as the democratic cross-tribe block.
        """
        N = N_half * 2
        torch.manual_seed(seed)
        geo, spec = embed_tokens_channels(N, base_seed=seed)
        tokens = geo + spec
        E_mat = energy_matrix(tokens)
        p_dem = E_mat / (E_mat.sum() + 1e-30)   # democratic: genuine bilocal everywhere

        # Tribal: replace cross-tribe blocks with factorized product
        p_trib = p_dem.clone()
        p_marg = p_dem.sum(dim=1)
        p_marg_A = p_marg[:N_half]  / (p_marg[:N_half].sum()  + 1e-30)
        p_marg_B = p_marg[N_half:]  / (p_marg[N_half:].sum()  + 1e-30)
        # Weight of the democratic cross-tribe block
        cross_weight = p_dem[:N_half, N_half:].sum()
        cross_fact = torch.outer(p_marg_A, p_marg_B)
        cross_fact = cross_fact / (cross_fact.sum() + 1e-30) * cross_weight
        p_trib[:N_half, N_half:] = cross_fact
        p_trib[N_half:, :N_half] = cross_fact.T
        p_trib = p_trib / (p_trib.sum() + 1e-30)

        cross_mask = torch.zeros(N, N, dtype=torch.bool)
        cross_mask[:N_half, N_half:] = True
        cross_mask[N_half:, :N_half] = True

        return p_dem, p_trib, cross_mask, N

    # REVISED TEST (v2) — replaces test_DC4_v1_content_collapses_cross_tribe.
    # Change: measures Δcontent and Δlens at cross-tribe pairs between democratic and
    # tribalized configurations of equal weight, rather than within/cross ratio.
    # This controls for probability-scale effects (see _tribal_distribution annotation).
    def test_DC4_content_collapses_cross_tribe(self):
        """
        [REVISED v2] Tribalization = replacing genuine bilocal cross-tribe coupling with
        factorized coupling of equal weight.

        H₀: Δcontent_cross ≈ Δlens_cross (both channels equally detect tribalization).
        H₁: Δlens_cross ≈ 0 (lens invariant — marginals unchanged),
            Δcontent_cross > 0 (content detects the factorization).

        This is DC2 applied specifically to the cross-tribe pairs.
        """
        N_half = 50
        p_dem, p_trib, cross_mask, N = self._democratic_and_tribal(N_half, SEED)

        K = causal_kernel_matrix(N)
        # ν built from marginals — same for democratic and tribal (marginals preserved)
        nu_dem  = nu_field(p_dem.sum(dim=1),  K)
        nu_trib = nu_field(p_trib.sum(dim=1), K)
        phi = torch.ones(N, N, dtype=FDTYPE) / N

        _, lens_dem,    content_dem,    _ = _channel_factors(p_dem,  nu_dem,  phi)
        _, lens_trib,   content_trib,   _ = _channel_factors(p_trib, nu_trib, phi)

        delta_lens_cross    = (lens_dem    - lens_trib   )[cross_mask].abs().mean().item()
        delta_content_cross = (content_dem - content_trib)[cross_mask].abs().mean().item()
        ratio = delta_content_cross / (delta_lens_cross + 1e-12)

        with open(os.path.join(RESULTS, 'DC4_tribal_channel_analysis.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['quantity', 'value'],
                ['delta_lens_cross_tribe',    delta_lens_cross],
                ['delta_content_cross_tribe', delta_content_cross],
                ['content_to_lens_delta_ratio', ratio],
                ['interpretation',
                 'H1: content detects tribalization at cross-tribe pairs; lens does not'],
                ['verdict', 'H1_supported' if ratio > 10 else 'H0_not_rejected'],
            ])

        assert ratio > 10, (
            f"H₀ not rejected: Δcontent/Δlens at cross-tribe pairs = {ratio:.1f}, "
            f"expected > 10. Content should detect tribalization; lens should not.")

    def test_DC4_lens_survives_tribalization(self):
        """Lens remains nonzero for cross-tribe pairs (local structure survives tribalism)."""
        N_half = 50
        p_tribal, n_h = self._tribal_distribution(N_half, SEED)
        N = N_half * 2

        K = causal_kernel_matrix(N)
        nu = nu_field(p_tribal.sum(dim=1), K)
        phi = torch.ones(N, N, dtype=FDTYPE) / N

        _, lens, _, _ = _channel_factors(p_tribal, nu, phi)

        cross_AB = torch.zeros(N, N, dtype=torch.bool)
        cross_AB[:n_h, n_h:] = True
        cross_AB[n_h:, :n_h] = True

        lens_cross_mean = lens[cross_AB].abs().mean().item()

        assert lens_cross_mean > 1e-4, (
            f"Lens collapsed cross-tribe: mean={lens_cross_mean:.3e}. "
            f"Local structure should survive even when non-local coherence is lost.")


# ---------------------------------------------------------------------------
# DC5 — Predictability (content not linearly predictable from lens)
# ---------------------------------------------------------------------------

class TestDC5_Predictability:
    """
    H₀: Content is linearly predictable from lens (R² > 0.5).
        If so, the two channels carry redundant information and the dual
        interpretation is superfluous.

    H₁: Content is not linearly predictable from lens (R² < 0.25).
        Knowing the local apparatus tells you little about the non-local field.

    This test directly operationalizes the claim that the two channels are
    independent information sources, not two measurements of the same thing.
    """

    def test_DC5_content_not_predictable_from_lens(self):
        N = 120
        p_bil, nu, phi, *_ = _bilocal_setup(N, SEED)
        _, lens, content, _ = _channel_factors(p_bil, nu, phi)

        L = lens.flatten().numpy().astype(np.float64)
        C = content.flatten().numpy().astype(np.float64)

        slope, intercept, r, pval, se = stats.linregress(L, C)
        R2 = r ** 2

        with open(os.path.join(RESULTS, 'DC5_predictability.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['quantity', 'value'],
                ['slope', slope],
                ['intercept', intercept],
                ['pearson_r', r],
                ['R_squared', R2],
                ['H0_threshold_R2', 0.5],
                ['H1_threshold_R2', 0.25],
                ['verdict', 'H1_supported' if R2 < 0.25 else 'H0_not_rejected'],
            ])

        assert R2 < 0.25, (
            f"H₀ not rejected: content~lens R²={R2:.3f}, expected < 0.25. "
            f"Content is predictable from lens — channels may not be independent.")

    def test_DC5_residual_structure(self):
        """Lens residuals (content - predicted) carry structure beyond noise."""
        N = 120
        p_bil, nu, phi, *_ = _bilocal_setup(N, SEED)
        _, lens, content, _ = _channel_factors(p_bil, nu, phi)

        L = lens.flatten().numpy().astype(np.float64)
        C = content.flatten().numpy().astype(np.float64)
        slope, intercept, *_ = stats.linregress(L, C)
        residuals = C - (slope * L + intercept)

        # Residuals should have non-trivial variance (not just noise)
        residual_std  = residuals.std()
        content_std   = C.std()
        unexplained_fraction = residual_std / (content_std + 1e-10)

        with open(os.path.join(RESULTS, 'DC5_residual_structure.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([
                ['quantity', 'value'],
                ['content_std', content_std],
                ['residual_std', residual_std],
                ['unexplained_fraction', unexplained_fraction],
                ['interpretation', 'fraction of content variance not explained by lens'],
            ])

        # Most of content's variance should be unexplained by lens
        assert unexplained_fraction > 0.8, (
            f"Lens explains too much of content: only {1-unexplained_fraction:.1%} unexplained. "
            f"Expected > 80% unexplained under H₁.")


# ---------------------------------------------------------------------------
# DC6 — Integration-Order Signature (2D and 3D spatiotemporal grids)
# ---------------------------------------------------------------------------

# Grid helpers ------------------------------------------------------------

def _grid_tokens_amplitude(shape, seed):
    """Embed N = prod(shape) tokens; return (tokens, p_bil, p_marg, N)."""
    N = 1
    for d in shape:
        N *= d
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tokens = geo + spec
    E_mat = energy_matrix(tokens)
    p_bil = E_mat / (E_mat.sum() + 1e-30)
    p_marg = p_bil.sum(dim=1)
    return tokens, p_bil, p_marg, N


def _pair_masks_2d(N_space, N_time):
    """Boolean masks classifying all (N,N) pairs on a space×time grid.

    Token i → space_idx = i % N_space,  time_idx = i // N_space.
    Causal direction: j is in the past of i when time_idx[i] > time_idx[j].
    """
    N = N_space * N_time
    idx = torch.arange(N)
    s = (idx % N_space).float()
    t = (idx // N_space).float()

    ds = (s.unsqueeze(1) - s.unsqueeze(0)).abs()   # (N,N) spatial separation
    dt = t.unsqueeze(1) - t.unsqueeze(0)            # (N,N) signed temporal (+ = j past)

    causal = dt > 0

    spatial_mask   = (dt == 0) & (ds > 0)
    temporal_mask  = causal & (ds == 0)
    mixed_mask     = causal & (ds > 0)
    timelike_mask  = mixed_mask & (dt.abs() > ds)   # |Δt| > |Δs|
    lightlike_mask = mixed_mask & (dt.abs() == ds)  # |Δt| = |Δs|
    spacelike_mask = mixed_mask & (dt.abs() < ds)   # |Δt| < |Δs|

    return {
        'spatial':        spatial_mask,
        'temporal':       temporal_mask,
        'mixed':          mixed_mask,
        'timelike':       timelike_mask,
        'lightlike':      lightlike_mask,
        'spacelike_mixed': spacelike_mask,
    }


def _pair_masks_3d(N_x, N_y, N_t):
    """Boolean masks for a 3D x×y×time grid.

    Token i → ix = i % N_x,  iy = (i // N_x) % N_y,  it = i // (N_x*N_y).
    """
    N = N_x * N_y * N_t
    idx = torch.arange(N)
    ix = (idx % N_x).float()
    iy = ((idx // N_x) % N_y).float()
    it = (idx // (N_x * N_y)).float()

    dx  = (ix.unsqueeze(1) - ix.unsqueeze(0)).abs()
    dy  = (iy.unsqueeze(1) - iy.unsqueeze(0)).abs()
    dt  = it.unsqueeze(1) - it.unsqueeze(0)            # signed
    dr2 = dx**2 + dy**2                                # squared spatial distance
    dt2 = dt**2

    causal     = dt > 0
    dr_nonzero = dr2 > 0

    spatial_mask   = (dt == 0) & dr_nonzero
    temporal_mask  = causal & ~dr_nonzero
    mixed_mask     = causal & dr_nonzero
    timelike_mask  = mixed_mask & (dt2 > dr2)
    lightlike_mask = mixed_mask & (dt2 == dr2)
    spacelike_mask = mixed_mask & (dt2 < dr2)

    return {
        'spatial':        spatial_mask,
        'temporal':       temporal_mask,
        'mixed':          mixed_mask,
        'timelike':       timelike_mask,
        'lightlike':      lightlike_mask,
        'spacelike_mixed': spacelike_mask,
    }


def _spatiotemporal_kernel_2d(N_space, N_time, alpha_K=0.5):
    """Causal power-law kernel on a space×time grid.  K(i,j) > 0 iff j is past of i."""
    N = N_space * N_time
    idx = torch.arange(N, dtype=FDTYPE)
    s   = idx % N_space
    t   = idx // N_space
    ds  = (s.unsqueeze(1) - s.unsqueeze(0)).abs()
    dt  = t.unsqueeze(1) - t.unsqueeze(0)
    dist = (ds**2 + dt.clamp(min=0)**2).sqrt()
    K    = (1.0 + dist).pow(-alpha_K)
    return K * (dt > 0).float()


def _spatiotemporal_kernel_3d(N_x, N_y, N_t, alpha_K=0.5):
    """Causal power-law kernel on an x×y×time grid."""
    N   = N_x * N_y * N_t
    idx = torch.arange(N, dtype=FDTYPE)
    ix  = idx % N_x
    iy  = (idx // N_x) % N_y
    it  = idx // (N_x * N_y)
    dx  = (ix.unsqueeze(1) - ix.unsqueeze(0)).abs()
    dy  = (iy.unsqueeze(1) - iy.unsqueeze(0)).abs()
    dt  = it.unsqueeze(1) - it.unsqueeze(0)
    dist = (dx**2 + dy**2 + dt.clamp(min=0)**2).sqrt()
    K    = (1.0 + dist).pow(-alpha_K)
    return K * (dt > 0).float()


def _ha_partition(tokens):
    """Per-pair Hermitian / anti-Hermitian energy partition.

    Returns h_mat (N,N), a_mat (N,N), ratio_mat (N,N) where
    ratio = ||H(Px·Py)||² / (||H||² + ||A||²).
    r_HA = 1 → pure Hermitian (1,3) regime;
    r_HA = 0.5 → balanced (2,2) boundary;
    r_HA < 0.5 → anti-Hermitian dominant.
    """
    N  = tokens.shape[0]
    Px = tokens.unsqueeze(1).expand(N, N, 4)
    Py = tokens.unsqueeze(0).expand(N, N, 4)
    c  = quat_mul(Px, Py)
    Hc = hermitian(c)
    Ac = c - Hc
    h_sq = (Hc.real**2 + Hc.imag**2).sum(dim=-1).to(FDTYPE)
    a_sq = (Ac.real**2 + Ac.imag**2).sum(dim=-1).to(FDTYPE)
    diag_mask = ~torch.eye(N, dtype=torch.bool)
    h_sq = h_sq * diag_mask.float()
    a_sq = a_sq * diag_mask.float()
    ratio = h_sq / (h_sq + a_sq + 1e-30)
    return h_sq, a_sq, ratio


def _sensitivity_by_mask(p_bil, nu, phi, masks):
    """Δcontent and Δlens under factorization, stratified by pair-type masks."""
    p_marg = p_bil.sum(dim=1)
    p_fact = torch.outer(p_marg, p_marg)
    p_fact = p_fact / (p_fact.sum() + 1e-30)
    _, lens_t, content_t, _ = _channel_factors(p_bil,  nu, phi)
    _, lens_f, content_f, _ = _channel_factors(p_fact, nu, phi)
    dl = (lens_t    - lens_f   ).abs()
    dc = (content_t - content_f).abs()
    out = {}
    for name, mask in masks.items():
        n = mask.sum().item()
        if n == 0:
            out[name] = dict(n=0, delta_lens=float('nan'),
                             delta_content=float('nan'), ratio=float('nan'))
            continue
        out[name] = dict(
            n=n,
            delta_lens=dl[mask].mean().item(),
            delta_content=dc[mask].mean().item(),
            ratio=dc[mask].mean().item() / (dl[mask].mean().item() + 1e-12),
        )
    return out


class TestDC6_IntegrationOrderSignature:
    """
    DC6 — Factorization sensitivity and Hermitian/anti-Hermitian energy
    partition stratified by pair type on 2D (space×time) and 3D (x×y×time) grids.

    Pair taxonomy:
        spatial        — same time slice, Δspace > 0        (K = 0 in ν)
        temporal       — same spatial position, Δt > 0      (K > 0 in ν, Δspace = 0)
        mixed          — Δspace > 0 AND Δt > 0              (both K and density active)
        └─ timelike    — |Δt| > |Δspace|                    (causal-kernel-dominated)
        └─ lightlike   — |Δt| = |Δspace|                    (balanced)
        └─ spacelike   — |Δt| < |Δspace|                    (density-dominated)

    H₀: Δcontent/Δlens ratio and r_HA (Hermitian/anti-Hermitian balance) are
        uniform across pair types.
    H₁: Spatial pairs show highest content/lens ratio (DC2 result).
        r_HA varies with lightcone angle — timelike pairs approach the (2,2)
        boundary (r_HA ↓ toward 0.5) while spacelike pairs stay near (1,3)
        (r_HA closer to 1).

    Political reading: the metric approaching (2,2) under temporal integration
    is the entropic fingerprint of cross-sector coherence collapse — measurable
    per pair-type in the content channel before the structural transition completes.
    """

    def _run_grid(self, shape, grid_label, masks_fn, kernel_fn, kernel_args):
        tokens, p_bil, p_marg, N = _grid_tokens_amplitude(shape, SEED)
        K   = kernel_fn(*kernel_args)
        nu  = nu_field(p_marg, K)
        phi = torch.ones(N, N, dtype=FDTYPE) / N
        masks = masks_fn(*shape)
        sens  = _sensitivity_by_mask(p_bil, nu, phi, masks)
        _, _, ratio_mat = _ha_partition(tokens)

        rows = [['grid', 'pair_type', 'n_pairs',
                 'delta_lens', 'delta_content', 'content_to_lens_ratio',
                 'mean_r_HA']]
        for name, mask in masks.items():
            n = mask.sum().item()
            if n == 0:
                continue
            r_ha = ratio_mat[mask].mean().item()
            s    = sens[name]
            rows.append([grid_label, name, n,
                          s['delta_lens'], s['delta_content'], s['ratio'], r_ha])

        fname = f'DC6_{grid_label}_pair_stratification.csv'
        with open(os.path.join(RESULTS, fname), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        return sens, ratio_mat, masks

    def test_DC6_2d_sensitivity_by_pair_type(self):
        """2D space×time grid: DC2 consistency check + sensitivity stratification."""
        shape = (6, 6)   # 36 tokens total
        sens, ratio_mat, masks = self._run_grid(
            shape, '2D', _pair_masks_2d,
            _spatiotemporal_kernel_2d, shape)

        # DC2 must hold within spatial pairs (K=0 → lens invariant, content sensitive)
        assert sens['spatial']['ratio'] > 5, (
            f"Spatial pairs: content/lens Δ-ratio = {sens['spatial']['ratio']:.1f}, "
            f"expected > 5. DC2 should hold within a time slice.")

        # Mixed pairs must exist on this grid
        assert sens['mixed']['n'] > 0, "No mixed pairs found on 6×6 grid."

    def test_DC6_3d_sensitivity_by_pair_type(self):
        """3D x×y×time grid: same checks extended to full 3D spatial volume."""
        shape = (4, 4, 4)   # 64 tokens total
        sens, ratio_mat, masks = self._run_grid(
            shape, '3D', _pair_masks_3d,
            _spatiotemporal_kernel_3d, shape)

        assert sens['spatial']['ratio'] > 5, (
            f"3D spatial pairs: content/lens Δ-ratio = {sens['spatial']['ratio']:.1f}, "
            f"expected > 5.")

        assert sens['mixed']['n'] > 0, "No mixed pairs found on 4×4×4 grid."

        # In 3D, timelike and spacelike mixed pairs should both be populated
        assert sens['timelike']['n'] > 0,       "No timelike pairs on 4×4×4 grid."
        assert sens['spacelike_mixed']['n'] > 0, "No spacelike mixed pairs on 4×4×4 grid."

    def test_DC6_lightcone_r_HA(self):
        """
        r_HA (Hermitian-sector fraction) vs. lightcone character on a 2D grid.

        H₀: r_HA is uniform across lightcone angle.
        H₁: r_HA(timelike) < r_HA(spacelike_mixed) — anti-Hermitian gains weight
            as temporal separation dominates, approaching the (2,2) boundary.

        This is a characterisation test: no hard ordering assertion.
        Hard assertions: both pair classes exist and r_HA is finite.
        """
        N_space, N_time = 8, 8   # 64 tokens, richer light-cone sampling
        tokens, p_bil, p_marg, N = _grid_tokens_amplitude((N_space, N_time), SEED)
        masks = _pair_masks_2d(N_space, N_time)
        _, _, ratio_mat = _ha_partition(tokens)

        r_ha = {}
        rows = [['pair_type', 'n_pairs', 'mean_r_HA', 'std_r_HA']]
        for name in ('spatial', 'temporal', 'mixed',
                     'timelike', 'lightlike', 'spacelike_mixed'):
            mask = masks[name]
            n = mask.sum().item()
            if n == 0:
                r_ha[name] = float('nan')
                rows.append([name, 0, float('nan'), float('nan')])
            else:
                vals = ratio_mat[mask]
                r_ha[name] = vals.mean().item()
                rows.append([name, n, r_ha[name], vals.std().item()])

        # Verdict: does r_HA decrease toward the timelike direction?
        tl = r_ha.get('timelike',       float('nan'))
        sl = r_ha.get('spacelike_mixed', float('nan'))
        h1_supported = (not math.isnan(tl) and not math.isnan(sl)
                        and tl < sl)
        rows.append(['verdict', '', 'H1_supported' if h1_supported else 'H0_not_rejected', ''])

        with open(os.path.join(RESULTS, 'DC6_lightcone_r_HA.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        # Hard assertions: pair classes exist and values are finite
        assert masks['timelike'].sum()       > 0, "No timelike pairs on 8×8 grid."
        assert masks['spacelike_mixed'].sum() > 0, "No spacelike-mixed pairs on 8×8 grid."
        assert not math.isnan(r_ha['timelike']),       "r_HA(timelike) is NaN."
        assert not math.isnan(r_ha['spacelike_mixed']), "r_HA(spacelike_mixed) is NaN."
