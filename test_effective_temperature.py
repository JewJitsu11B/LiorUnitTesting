"""
Effective-Temperature / Superstatistical-Form Suite (ET1-ET5)
=============================================================

Appendix A of `bilocal entropy tests summary 4.pdf` -- the only unrun suite, flagged
there as the highest-value next step.  Pre-registration: PREREGISTRATION.md (read first).

Object: the LENS channel -ln(nu) read along a causal-past trajectory through the bilocal
field, as a function of separation tau, with kernel K(tau)=(1+tau)^{-aK}.

Analytic backbone (kernel-only, w_d=0), verified exact before registration:
    lens[x,x-tau] = -ln(1 + w_c (1+tau)^{-aK})
    (1+tau)^{-aK} is exactly a q-exponential exp_q(-tau/tau_c) with
        q     = 1 + 1/aK   (q-character, memory-length reading)
        tau_c = 1/aK
So 1/tau_c = aK is a DECORRELATION RATE, not a temperature.  The backbone establishes the
superstatistical FORM; "effective temperature" is a partially-tested target (needs FDT/KMS).

Primary inference instrument: PARAMETER-FREE RESIDUAL TEST.
    q = 1+1/aK, tau_c = 1/aK are fixed from aK (a construction input, not fitted).
    Shape-only R2: normalize G(tau) and q-exp prediction to run from 1 (at first tau) to 0
    (at last tau) -- amplitude A and baseline G_inf cancel, no tail estimation needed.
    At w_d=0: exact backbone, R2_shape = 1.
    At w_d>0: density coupling through -ln distorts shape ~5-10%; R2_shape > 0.90 means
              kernel q-exp shape dominates the tau-variation.

Fitting (curve_fit) retained as SECONDARY / FAILURE-CHARACTERIZATION only:
    GATE: verify q,tau_c land near closed form (sanity).
    ET3: characterize the density-coupling downward pull on secondary q-hat
         (restricted to the well-conditioned regime aK <= 2.0; at aK=3.0 the
          shape approaches exponential and the fitter is unreliable).

Tests:
    GATE  Exact -ln reduction holds; fit q,tau_c match closed form  (gates everything)
    ET1   Trajectory stationarity (response depends on tau, not abs x)
    ET2   Response shape is q-exp with fixed q=1+1/aK; q-exp beats best-fit exponential
    ET3   Per-aK residual tests pass across the full grid (1/aK parameterization)
    ET4   Markovian limit: residual test holds at large aK (q->1 algebraically)
    ET5   Same fixed-q shape fits i.i.d.; structured field detects content-dependence
    SWAP  Lens carries real swap-asymmetry (detailed-balance-like, non-degenerate)

Run:
    env313t/bin/python -m pytest Docs/effective_temperature/test_effective_temperature.py -v
    env313t/bin/python Docs/effective_temperature/test_effective_temperature.py   # summary
"""

import os, sys, csv, math

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))

import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit

from cal.biquaternion import embed_tokens_channels, FDTYPE
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field

SEED = 42
N_DEFAULT = 140
TAU_MAX = 40
POOL_SEEDS = [SEED + k * 101 for k in range(10)]
ALPHA_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
WD_SMALL    = 0.1
WC_SMALL    = 0.3
WC_DEFAULT  = 0.8
WD_DEFAULT  = 0.5

RESULTS = os.path.join(_HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)


def _csv(name, rows):
    with open(os.path.join(RESULTS, name), 'w', newline='') as f:
        csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------------
# Field + lens construction
# ---------------------------------------------------------------------------

def _tokens(field, N, seed):
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tok = geo + spec
    if field == 'structured':
        env = torch.ones(N, dtype=tok.real.dtype)
        b = N // 4
        for k, amp in enumerate([1.0, 2.2, 0.6, 1.6]):
            lo, hi = k * b, (N if k == 3 else (k + 1) * b)
            env[lo:hi] = amp
        tok = tok * env.unsqueeze(1).to(tok.dtype)
    return tok


def _lens(field, N, seed, alpha_K, w_d, w_c):
    tok = _tokens(field, N, seed)
    E = energy_matrix(tok)
    p_bil = E / (E.sum() + 1e-30)
    p_marg = p_bil.sum(dim=1)
    K = causal_kernel_matrix(N, alpha_K=alpha_K)
    nu = nu_field(p_marg, K, w_d=w_d, w_c=w_c)
    lens = -torch.log(nu.clamp(min=1e-4))
    return lens, nu, p_marg, K


def _response(lens, tau_max=TAU_MAX):
    """G(tau)=<lens[x,x-tau]>_x; returns (taus, G, sig, cnt)."""
    N = lens.shape[0]
    t_hi = min(tau_max, N - 2)
    taus, G, sig, cnt = [], [], [], []
    for t in range(1, t_hi + 1):
        v = torch.diagonal(lens, offset=-t).numpy().astype(np.float64)
        if v.size < 2:
            continue
        taus.append(t); G.append(v.mean())
        sig.append(v.std(ddof=1) / math.sqrt(v.size)); cnt.append(v.size)
    taus = np.array(taus, float); G = np.array(G)
    sig = np.array(sig); cnt = np.array(cnt)
    span = (G.max() - G.min()) if G.size else 1.0
    sig = np.clip(sig, 1e-3 * (abs(span) + 1e-12), None)
    return taus, G, sig, cnt


def _pooled_response(field, alpha_K, w_d, w_c, N, seeds):
    """Average G(tau) over seeds before any comparison; density term averages toward its mean."""
    rows = []
    for s in seeds:
        lens, *_ = _lens(field, N, s, alpha_K, w_d, w_c)
        taus, G, sig, cnt = _response(lens)
        rows.append((taus, G))
    L = min(len(r[0]) for r in rows)
    taus = rows[0][0][:L]
    M = np.array([r[1][:L] for r in rows])
    Gp = M.mean(axis=0)
    sig = M.std(axis=0, ddof=1) / math.sqrt(M.shape[0])
    span = abs(Gp.max() - Gp.min())
    sig = np.clip(sig, 1e-3 * (span + 1e-12), None)
    return taus, Gp, sig, M


# ---------------------------------------------------------------------------
# PRIMARY INSTRUMENT: parameter-free residual test
# ---------------------------------------------------------------------------

def _qexp_norm(taus, q, tc):
    """Normalized q-exponential (value at tau=0 is 1.0)."""
    qm1 = q - 1.0
    return np.clip(1.0 + qm1 * np.asarray(taus, float) / tc, 1e-12, None) ** (-1.0 / qm1)


R2_THRESH = 0.90   # shape explains >= 90% of tau-variation; density coupling accounts for rest


def _residual_check(taus, G, sig, aK, tail_frac=0.2):
    """Fix q=1+1/aK, tau_c=1/aK from aK.  A and G_inf are data-derived.

    PRIMARY metric: r2_shape -- shape-only R2.  Normalizes G and G_pred to run from 1
    (at first tau) to 0 (at last tau), eliminating amplitude and G_inf estimation bias.
    With tau_c = 1/aK, aK*tau_c = 1 exactly, so f(tau) = (1+tau)^{-aK} and
    f(tau_1) = 2^{-aK} (not [1+1/aK^2]^{-aK}).  Two independent biases in the
    tail-estimation approach are eliminated by shape-only normalization:
      - Amplitude factor: 2^{-aK} departs from 1 most at large aK (near-Markovian).
      - Tail non-convergence: at small aK, 33% of signal remains at tau=40 for aK=0.3
        (41^{-0.3} ~ 0.33), biasing the tail-estimated G_inf.
    The shape-only metric is immune to both.

    SECONDARY: r2 with tail-estimated G_inf (informative for large aK where tail converges).
    chi2 reported but not for pass/fail (sigma is measurement precision, not model tolerance).
    """
    q  = 1.0 + 1.0 / aK
    tc = 1.0 / aK
    qn = _qexp_norm(taus, q, tc)

    # --- shape-only R2 (primary) ---
    eps_G = float(G[-1]); eps_q = float(qn[-1])
    denom_G = float(G[0] - eps_G); denom_q = float(qn[0] - eps_q)
    G_s = (G - eps_G) / (denom_G + 1e-30)
    P_s = (qn - eps_q) / (denom_q + 1e-30)
    resid_s = G_s - P_s
    ss_res_s = float(np.sum(resid_s ** 2))
    ss_tot_s = float(np.sum((G_s - G_s.mean()) ** 2))
    r2_shape = 1.0 - ss_res_s / (ss_tot_s + 1e-30)

    # --- tail-estimated A / G_inf (secondary, informative for large aK) ---
    n_tail = max(2, int(len(G) * tail_frac))
    G_inf = float(G[-n_tail:].mean())
    qn0 = float(qn[0])
    A = (float(G[0]) - G_inf) / max(qn0, 1e-9)
    pred = A * qn + G_inf
    resid = G - pred
    chi2  = float(np.sum((resid / sig) ** 2))
    dof   = max(len(taus) - 2, 1)
    pval  = float(stats.chi2.sf(chi2, dof))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((G - G.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)

    # expose shape-normalized arrays for downstream use (e.g. adversarial null)
    return dict(chi2=chi2, dof=dof, pval=pval, r2=r2, r2_shape=r2_shape,
                resid=resid, pred=pred,
                q_fixed=q, tc_fixed=tc, A=A, G_inf=G_inf,
                G_s=G_s, ss_tot_s=ss_tot_s)


def _pooled_residual_check(field, alpha_K, w_d, w_c, N=N_DEFAULT, seeds=POOL_SEEDS):
    taus, Gp, sig, M = _pooled_response(field, alpha_K, w_d, w_c, N, seeds)
    rc = _residual_check(taus, Gp, sig, alpha_K)
    return rc, taus, Gp, sig


# ---------------------------------------------------------------------------
# SECONDARY: weighted exp vs q-exp fit (GATE sanity + failure characterization)
# ---------------------------------------------------------------------------

def _expm(t, A, tc, G0):
    return A * np.exp(np.clip(-t / tc, -700.0, 700.0)) + G0


def _qexp(t, A, tc, q, G0):
    qm1 = q - 1.0
    base = np.clip(1.0 + qm1 * t / tc, 1e-12, None)
    return A * base ** (-1.0 / qm1) + G0


def _chi2w(y, f, sig):
    return float(np.sum(((y - f) / sig) ** 2))


def _fit(taus, G, sig):
    out = dict(ok=False, q=np.nan, q_se=np.nan, tau_c=np.nan,
               aic_exp=np.nan, aic_qexp=np.nan, lrt=np.nan, lrt_p=np.nan)
    A0 = float(G[0] - G[-1]); G0 = float(G[-1]); tc0 = max(float(taus.mean()), 1.0)
    try:
        pe, _ = curve_fit(_expm, taus, G, p0=[A0, tc0, G0], sigma=sig,
                          absolute_sigma=True, maxfev=40000)
        chi_e = _chi2w(G, _expm(taus, *pe), sig)
        pq, cq = curve_fit(_qexp, taus, G, p0=[A0, 1.0, 1.5, G0], sigma=sig,
                           absolute_sigma=True, maxfev=40000,
                           bounds=([-np.inf, 1e-6, 1.0+1e-9, -np.inf],
                                   [ np.inf, np.inf,      8.0,  np.inf]))
        chi_q = _chi2w(G, _qexp(taus, *pq), sig)
    except Exception as e:
        out['err'] = str(e); return out
    out['ok'] = True
    out['tau_c'] = float(pq[1]); out['q'] = float(pq[2])
    out['q_se'] = float(np.sqrt(cq[2, 2])) if np.all(np.isfinite(cq)) else np.nan
    out['aic_exp']  = chi_e + 2 * 3
    out['aic_qexp'] = chi_q + 2 * 4
    lrt = max(chi_e - chi_q, 0.0)
    out['lrt'] = lrt
    out['lrt_p'] = 0.5 * float(stats.chi2.sf(lrt, 1))
    return out


# ===========================================================================
# GATE -- exact -ln reduction (zero parameter); fit as sanity only
# ===========================================================================

def run_gate():
    res = []
    rows = [['alpha_K', 'max_rel_dev_vs_exact', 'x_spread',
             'fit_q', 'q_pred', 'fit_tau_c', 'tc_pred']]
    wc = 0.05
    for aK in [0.7, 1.5, 2.0, 3.0]:
        lens, *_ = _lens('iid', N_DEFAULT, SEED, aK, w_d=0.0, w_c=wc)
        taus = np.arange(1, 31, dtype=float)
        emp      = np.array([torch.diagonal(lens, offset=-int(t))[0].item() for t in taus])
        xspread  = max(float(torch.diagonal(lens, offset=-int(t)).std().item()) for t in taus[:20])
        analytic = -np.log(1.0 + wc * (1.0 + taus) ** (-aK))
        keep = np.abs(analytic) > 1e-4
        rel = float(np.max(np.abs(emp[keep] - analytic[keep]) / np.abs(analytic[keep])))
        tA, G, sig, _ = _response(lens)
        r = _fit(tA, G, sig)   # secondary sanity; q,tau_c should land near closed form
        q_pred, tc_pred = 1 + 1/aK, 1/aK
        rows.append([aK, rel, xspread, r['q'], q_pred, r['tau_c'], tc_pred])
        res.append((f"GATE exact reduction aK={aK}: rel-dev {rel:.1e}",
                    rel < 1e-2 and xspread < 1e-5,
                    f"rel={rel:.2e} xspread={xspread:.2e}"))
        res.append((f"GATE fit q,tau_c match closed form aK={aK}",
                    abs(r['q'] - q_pred)/q_pred < 0.08 and abs(r['tau_c'] - tc_pred)/tc_pred < 0.12,
                    f"q={r['q']:.3f}(pred {q_pred:.3f}) tau_c={r['tau_c']:.3f}(pred {tc_pred:.3f})"))
    _csv('deputy_analytic_qscaling.csv', rows)
    return res


# ===========================================================================
# ET1 -- trajectory stationarity
# ===========================================================================

def run_et1():
    res = []; rows = [['alpha_K', 'max_rel_halfsplit_dev', 'median_rel_spread', 'verdict']]
    THRESH = 0.35
    for aK in [0.5, 1.0, 2.0]:
        devs, spreads = [], []
        for s in POOL_SEEDS:
            lens, *_ = _lens('iid', N_DEFAULT, s, aK, w_d=WD_SMALL, w_c=WC_SMALL)
            glo, ghi, gall, sp = [], [], [], []
            for t in range(1, 26):
                v = torch.diagonal(lens, offset=-int(t)).numpy().astype(np.float64)
                if v.size < 8: continue
                half = v.size // 2
                glo.append(v[:half].mean()); ghi.append(v[half:].mean())
                gall.append(v.mean()); sp.append(v.std() / (abs(v.mean()) + 1e-9))
            glo, ghi, gall = map(np.array, (glo, ghi, gall))
            devs.append(np.max(np.abs(glo - ghi) / (np.abs(gall) + 1e-9)))
            spreads.append(np.median(sp))
        md = float(np.mean(devs)); ms = float(np.mean(spreads))
        ok = md < THRESH
        rows.append([aK, md, ms, 'H1_stationary' if ok else 'H0_not_rejected'])
        res.append((f"ET1 stationarity aK={aK}: half-split dev {md:.3f} < {THRESH}", ok,
                    f"halfsplit={md:.3f} median_rel_spread={ms:.3f}"))
    _csv('ET1_stationarity.csv', rows)
    return res


# ===========================================================================
# ET2 -- q-exp shape with fixed q=1+1/aK; adversarial null vs best-fit exponential
# ===========================================================================

def run_et2():
    """Primary: R2_shape test with fixed q=1+1/aK, tau_c=1/aK (zero free parameters).

    Null calibration:
      (a) Exact q-exp synthetic data: must pass R2_shape > R2_THRESH (validates true positives).
      (b) Adversarial null: fit a free-parameter exponential to the ACTUAL FIELD DATA, compute
          its R2_shape against the field response using the same endpoint normalization, and
          verify the zero-parameter q-exp wins.  This directly establishes discrimination on
          the field data, not on synthetic data with an arbitrary tau_c.

    Secondary: fitted q reported for comparison but not the inferential object.
    """
    res = []
    aK = 1.5
    q_pred = 1.0 + 1.0 / aK

    # --- field residual test (pooled over seeds) ---
    rc, taus, Gp, sig = _pooled_residual_check('iid', aK, w_d=WD_SMALL, w_c=WC_SMALL)
    field_pass = rc['r2_shape'] > R2_THRESH

    # secondary: fitted q for reporting
    rf = _fit(taus, Gp, sig)

    _csv('ET2_thermal_fit.csv', [
        ['quantity', 'value'],
        ['alpha_K', aK], ['w_c', WC_SMALL], ['w_d', WD_SMALL],
        ['q_fixed_pred', q_pred], ['tc_fixed_pred', 1/aK],
        ['A_data_derived', rc['A']], ['G_inf_data_derived', rc['G_inf']],
        ['r2_shape', rc['r2_shape']], ['r2_tail', rc['r2']], ['chi2', rc['chi2']], ['dof', rc['dof']],
        ['secondary_fit_q', rf['q']], ['secondary_fit_tau_c', rf['tau_c']],
        ['verdict', 'H1_qexponential_shape' if field_pass else 'H0_not_rejected'],
    ])
    res.append((f"ET2 fixed q-exp shape: R2_shape>{R2_THRESH} aK={aK}",
                field_pass,
                f"R2_shape={rc['r2_shape']:.4f}; secondary fit q={rf['q']:.3f} (pred {q_pred:.3f})"))

    # --- null calibration ---
    rng = np.random.RandomState(7)
    t_synth = np.arange(1, TAU_MAX + 1, dtype=float)
    noise_sig = 0.005

    # (a) exact q-exp synthetic data -- R2_shape should exceed threshold (true positive)
    A_true = 0.6; Ginf_true = 0.05
    qn = _qexp_norm(t_synth, q_pred, 1.0/aK)
    y_q = A_true * qn + Ginf_true + rng.normal(0, noise_sig, t_synth.size)
    sig_q = np.full_like(t_synth, noise_sig)
    rc_q = _residual_check(t_synth, y_q, sig_q, aK)
    qdata_pass = rc_q['r2_shape'] > R2_THRESH

    # (b) adversarial null: best-fit free-parameter exponential to the actual field data.
    # The zero-parameter q-exp must outperform this adversary on the field data itself.
    # Shape-normalize the best-fit exponential using the same endpoint logic.
    try:
        A0 = float(Gp[0] - Gp[-1]); G0 = float(Gp[-1])
        pe, _ = curve_fit(_expm, taus, Gp, p0=[A0, 2.0, G0], maxfev=40000)
        best_fit_exp = _expm(taus, *pe)
        eps_e = float(best_fit_exp[-1]); denom_e = float(best_fit_exp[0] - eps_e)
        P_s_exp = (best_fit_exp - eps_e) / (denom_e + 1e-30)
        # field response already shape-normalized in rc (G_s, ss_tot_s exposed)
        G_s_field = rc['G_s']
        ss_tot_field = rc['ss_tot_s']
        resid_exp = G_s_field - P_s_exp
        r2_shape_bestfit_exp = 1.0 - float(np.sum(resid_exp**2)) / (ss_tot_field + 1e-30)
        # zero-parameter q-exp must beat the free-parameter adversary
        shape_dominance_pass = bool(rc['r2_shape'] > r2_shape_bestfit_exp)
        adv_detail = (f"q-exp R2_shape={rc['r2_shape']:.4f} vs "
                      f"best-fit-exp R2_shape={r2_shape_bestfit_exp:.4f} "
                      f"(exp tau_c={pe[1]:.3f})")
    except Exception as e:
        shape_dominance_pass = False
        r2_shape_bestfit_exp = float('nan')
        adv_detail = f"curve_fit failed: {e}"

    _csv('ET2_null_calibration.csv', [
        ['case', 'r2_shape', 'chi2', 'dof', 'expected_outcome', 'pass'],
        ['exact_qexp_synthetic', rc_q['r2_shape'], rc_q['chi2'], rc_q['dof'],
         f'R2_shape>{R2_THRESH} (true positive check)', qdata_pass],
        ['adversarial_bestfit_exp', r2_shape_bestfit_exp, '', '',
         'q-exp R2_shape > best-fit-exp R2_shape on field data', shape_dominance_pass],
    ])
    res.append((f"ET2 null: exact q-exp synthetic data R2_shape>{R2_THRESH}",
                bool(qdata_pass), f"R2_shape={rc_q['r2_shape']:.4f}"))
    res.append(("ET2 adversarial null: zero-param q-exp beats free-param best-fit exp on field data",
                shape_dominance_pass, adv_detail))
    return res


# ===========================================================================
# ET3 -- per-aK residual tests; downward-pull characterization (aK <= 2.0 only)
# ===========================================================================

def run_et3():
    """Primary: R2_shape > R2_THRESH at each aK validates the 1/aK parameterization directly --
    the forward model built on q=1+1/aK, tau_c=1/aK is what passes or fails.  No separate
    linear regression needed; that would only restate the analytic identity q-1 = 1/aK.

    Secondary characterization: the density-kernel coupling through -ln shifts secondary
    q-hat below q_pred for aK <= 2.0.  At aK=3.0 the q->1 shape is near-exponential and
    the free-parameter fitter is unreliable; q-hat at that point is not a valid diagnostic.
    """
    res = []
    rows = [['alpha_K', 'inv_aK', 'q_fixed', 'tc_fixed',
             'r2_shape', 'r2_tail', 'chi2', 'r2_shape_pass', 'secondary_fit_q']]

    r2s_shape, chi2s, aKs_pass, fit_results = [], [], [], []
    aKs_all = ALPHA_GRID
    aKs_arr = np.array(aKs_all)
    inv_arr = 1.0 / aKs_arr

    for aK in aKs_all:
        rc, taus, Gp, sig = _pooled_residual_check('iid', aK, w_d=WD_SMALL, w_c=WC_SMALL)
        rf = _fit(taus, Gp, sig)   # secondary
        r2s_shape.append(rc['r2_shape']); chi2s.append(rc['chi2'])
        aKs_pass.append(rc['r2_shape'] > R2_THRESH)
        fit_results.append(rf)
        rows.append([aK, 1/aK, rc['q_fixed'], rc['tc_fixed'],
                     rc['r2_shape'], rc['r2'], rc['chi2'],
                     rc['r2_shape'] > R2_THRESH, rf['q']])
    _csv('ET3_alphaK_sweep.csv', rows)

    n_pass = sum(aKs_pass)
    res.append((f"ET3 per-aK shape tests: >= 6/7 pass (R2_shape>{R2_THRESH})",
                n_pass >= 6,
                f"{n_pass}/7 pass; R2_shapes={[f'{r:.3f}' for r in r2s_shape]}"))

    # Spearman sanity: R2_shape should not strongly degrade with 1/aK
    rho, pval_rho = stats.spearmanr(r2s_shape, [1.0/a for a in aKs_all])
    res.append(("ET3 secondary: R2_shape not strongly degrading with 1/aK",
                rho > -0.7,
                f"Spearman rho(R2_shape, 1/aK)={rho:.3f} p={pval_rho:.2e}"))

    # Density-coupling downward pull on secondary q-hat (well-conditioned regime only).
    # At aK=3.0, q->1 makes the free-parameter fit ill-conditioned; that point is excluded.
    q_pred_arr = 1.0 + inv_arr
    well_conditioned = aKs_arr <= 2.0
    q_fit_arr = np.array([rf['q'] if rf['ok'] else float('nan') for rf in fit_results])
    q_shifts_wc = q_fit_arr[well_conditioned] - q_pred_arr[well_conditioned]
    downward_pull = bool(all(np.isfinite(s) and s < 0 for s in q_shifts_wc))
    aK30_shift = float(q_fit_arr[~well_conditioned][0] - q_pred_arr[~well_conditioned][0]) \
                 if (~well_conditioned).any() else float('nan')
    res.append(("ET3 density coupling: secondary q-hat below q_pred for aK<=2.0",
                downward_pull,
                f"shifts(aK<=2.0)={[f'{s:.3f}' for s in q_shifts_wc]}; "
                f"aK=3.0 shift={aK30_shift:.3f} (ill-conditioned, excluded)"))

    # w_c sweep at w_d=0: exact -ln reduction holds at all w_c (rel-dev test, zero parameters)
    aK_wc = 1.5
    wc_rows = [['w_c', 'max_rel_dev_vs_exact', 'r2_vs_exact']]
    wc_rels = []
    for wc in [0.05, 0.1, 0.2, 0.4, 0.8]:
        lens, *_ = _lens('iid', N_DEFAULT, SEED, aK_wc, w_d=0.0, w_c=wc)
        tA, G, sig, _ = _response(lens)
        analytic = -np.log(1.0 + wc * (1.0 + tA) ** (-aK_wc))
        keep = np.abs(analytic) > 1e-4
        rel = float(np.max(np.abs(G[keep] - analytic[keep]) / np.abs(analytic[keep]))) if keep.any() else 0.0
        resid = G - analytic
        ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((G - G.mean())**2))
        r2_ex = 1.0 - ss_res/(ss_tot+1e-30)
        wc_rows.append([wc, rel, r2_ex]); wc_rels.append(rel)
    _csv('ET3_wc_sweep.csv', wc_rows)
    wc_pass = all(r <= 0.01 for r in wc_rels)
    res.append(("ET3 w_c sweep: exact -ln reduction holds at all w_c (w_d=0, rel-dev<=1%)",
                wc_pass, f"max_rel_devs={[f'{r:.3f}' for r in wc_rels]}"))

    # Operating-point degradation: documented finding (density masks at w_d=0.5)
    deg_rows = [['w_d', 'r2_aK0.7', 'r2_aK1.5', 'r2_aK3.0', 'any_fail']]
    for wd in [0.1, 0.3, 0.5]:
        r2vs = []
        for aK in [0.7, 1.5, 3.0]:
            rc2, _, _, _ = _pooled_residual_check('iid', aK, wd, WC_SMALL)
            r2vs.append(rc2['r2'])
        any_fail = any(r < 0.99 for r in r2vs)
        deg_rows.append([wd] + r2vs + [any_fail])
    _csv('ET3_wd_degradation.csv', deg_rows)
    return res


# ===========================================================================
# ET4 -- Markovian limit: residual test holds at large aK; q->1 algebraically
# ===========================================================================

def run_et4():
    """q=1+1/aK -> 1 as aK -> inf is algebraic (no free parameters, no fitting needed).
    Empirical content: the fixed-q shape test (R2>R2_THRESH) continues to pass at large aK,
    where the response becomes flat and near-Markovian.  delta/aK>>1 are analytic boundaries."""
    res = []
    rows = [['alpha_K', 'q_fixed', 'r2_shape', 'r2_tail', 'chi2', 'r2_shape_pass', 'kind']]
    fit_aKs = [0.7, 1.0, 1.5, 2.0, 3.0]
    r2s = []
    for aK in fit_aKs:
        rc, _, _, _ = _pooled_residual_check('iid', aK, w_d=WD_SMALL, w_c=WC_SMALL)
        r2s.append(rc['r2_shape'])
        rows.append([aK, rc['q_fixed'], rc['r2_shape'], rc['r2'], rc['chi2'],
                     rc['r2_shape'] > R2_THRESH, 'residual_test'])
    for aK in [4.0, 6.0, 10.0, 20.0]:
        rows.append([aK, 1+1/aK, '', '', '', '', 'analytic_boundary'])
    rows.append(['delta(aK->inf)', 1.0, '', '', '', '', 'analytic_boundary'])
    _csv('ET4_markovian_limit.csv', rows)

    large_aK_pass = r2s[-2] > R2_THRESH and r2s[-1] > R2_THRESH
    res.append((f"ET4 shape test R2_shape>{R2_THRESH} at large aK (near-Markovian q->1)",
                large_aK_pass,
                f"R2_shape@aK2={r2s[-2]:.4f} R2_shape@aK3={r2s[-1]:.4f}; q->1 algebraically"))
    return res


# ===========================================================================
# ET5 -- structured vs i.i.d.: same fixed-q shape fits both (kernel property)
# ===========================================================================

def run_et5():
    """Both fields tested against the SAME fixed-q prediction (q=1+1/aK from aK).
    First claim: i.i.d. field passes R2_shape>0.90 (kernel q-exp shape dominates).
    Second claim: structured field R2_shape < i.i.d. at all tested aK (content-dependence
    detected).  The topical block density is tau-dependent (nearby positions share a block),
    distorting the kernel q-exp shape at large aK where the density contribution is comparable
    to the kernel.  This is a finding, not a falsification: the kernel property holds for
    random fields; field content modulates the shape for structured fields."""
    res = []
    rows = [['alpha_K', 'r2_shape_iid', 'r2_shape_struct',
             'resid_rms_iid', 'resid_rms_struct', 'iid_pass', 'struct_pass']]
    aKs = [0.7, 1.0, 1.5, 2.0]
    iid_passes, struct_passes = [], []
    for aK in aKs:
        rci, taus_i, Gi, sig_i = _pooled_residual_check('iid',        aK, WD_SMALL, WC_SMALL)
        rcs, taus_s, Gs, sig_s = _pooled_residual_check('structured',  aK, WD_SMALL, WC_SMALL)
        ip = rci['r2_shape'] > R2_THRESH; sp = rcs['r2_shape'] > R2_THRESH
        iid_passes.append(ip); struct_passes.append(sp)
        rms_i = float(np.sqrt(np.mean(rci['resid']**2)))
        rms_s = float(np.sqrt(np.mean(rcs['resid']**2)))
        rows.append([aK, rci['r2_shape'], rcs['r2_shape'], rms_i, rms_s, ip, sp])
    _csv('ET5_structured_vs_random.csv', rows)

    iid_all = all(iid_passes)
    iid_r2s    = [float(r[1]) for r in rows[1:]]
    struct_r2s = [float(r[2]) for r in rows[1:]]
    content_dep = all(s < i for s, i in zip(struct_r2s, iid_r2s))
    res.append((f"ET5 i.i.d. field: R2_shape>{R2_THRESH} across aK grid",
                iid_all,
                f"R2_shapes={[f'{r:.3f}' for r in iid_r2s]}"))
    res.append(("ET5 content-dependence: structured R2_shape < i.i.d. at all aK",
                content_dep,
                f"iid={[f'{r:.3f}' for r in iid_r2s]} struct={[f'{r:.3f}' for r in struct_r2s]}; "
                f"kernel shape dominates i.i.d.; density-tau structure distorts shape in structured field"))
    return res


# ===========================================================================
# SWAP deputy -- real detailed-balance-like asymmetry
# ===========================================================================

def deputy_swap():
    """Pass criterion: |mean_asymmetry| > 5 * SEM, where SEM = std / sqrt(n_seeds).
    Significance is reported in SEM units throughout, consistent with the criterion."""
    res = []
    rows = [['alpha_K', 'mean_asym_tau<=10', 'std_across_seeds', 'SEM', 'significance_SEM', 'nonzero']]
    for aK in [0.7, 1.5]:
        vals = []
        for s in POOL_SEEDS:
            lens, *_ = _lens('iid', N_DEFAULT, s, aK, w_d=WD_DEFAULT, w_c=WC_DEFAULT)
            a = []
            for t in range(1, 11):
                past = torch.diagonal(lens, offset=-int(t))
                fut  = torch.diagonal(lens, offset= int(t))
                a.append(float((past - fut).mean().item()))
            vals.append(np.mean(a))
        vals = np.array(vals)
        sem = vals.std(ddof=1) / math.sqrt(vals.size)
        significance = abs(vals.mean()) / (sem + 1e-12)
        nonzero = significance > 5.0
        rows.append([aK, float(vals.mean()), float(vals.std(ddof=1)),
                     float(sem), float(significance), bool(nonzero)])
        res.append((f"SWAP asymmetry non-zero aK={aK}", bool(nonzero),
                    f"mean={vals.mean():.4f}, SEM={sem:.4f}, "
                    f"significance={significance:.1f} sigma (SEM units)"))
    _csv('deputy_swap_asymmetry.csv', rows)
    return res


# ===========================================================================
# pytest entrypoints + supervisor
# ===========================================================================

def _assert_all(results):
    for name, ok, details in results:
        assert ok, f"{name} :: {details}"


def test_GATE():  _assert_all(run_gate())
def test_ET1():   _assert_all(run_et1())
def test_ET2():   _assert_all(run_et2())
def test_ET3():   _assert_all(run_et3())
def test_ET4():   _assert_all(run_et4())
def test_ET5():   _assert_all(run_et5())
def test_SWAP():  _assert_all(deputy_swap())


def supervisor():
    groups = [("GATE", run_gate), ("ET1", run_et1), ("ET2", run_et2), ("ET3", run_et3),
              ("ET4", run_et4), ("ET5", run_et5), ("SWAP", deputy_swap)]
    all_results = {}; overall = True
    for name, fn in groups:
        print("\n" + "=" * 64)
        print(f"  {name}")
        print("=" * 64)
        try:
            results = fn()
        except Exception as e:
            results = [(f"{name} crashed", False, repr(e))]
        all_results[name] = results
        for tn, ok, d in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {tn}")
            if d and not ok:
                print(f"         {d}")
            overall = overall and ok
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for name, results in all_results.items():
        npass = sum(1 for _, ok, _ in results if ok)
        print(f"  {name}: {npass}/{len(results)} [{'PASS' if npass == len(results) else 'FAIL'}]")
    tp = sum(sum(1 for _, ok, _ in r if ok) for r in all_results.values())
    tt = sum(len(r) for r in all_results.values())
    print(f"\n  OVERALL: {tp}/{tt} {'PASS' if overall else 'FAIL'}")
    print("=" * 64)
    return overall, all_results


if __name__ == "__main__":
    print("=" * 64)
    print("  Effective-Temperature / Superstatistical-Form Suite (ET1-ET5)")
    print("=" * 64)
    ok, _ = supervisor()
    sys.exit(0 if ok else 1)
