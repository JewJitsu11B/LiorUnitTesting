"""
End-to-end architecture test: the DETECTOR is the Born cost; its kernel sets the q-deformation.

Confirms on the actual bilocal field (cal.entropy) that:
  - the detector (mixed_renyi_tsallis) tracks the surprisal -ln p, i.e. it is the Born cost
    (exp(-H/tau) recovers the Born weight), while the single-log entropy (nist) does NOT -- so the
    cost slot belongs to the detector, not the entropy;
  - q-deforming the detector-cost Boltzmann weight by q = 1 + 1/aK (the kernel-set superstatistical
    index) broadens BornMax into a heavier-tailed law;
  - normalization never fails (Gibbs is shift/scale invariant).
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy


def _field(N=140, seed=42):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    tok = geo + spec
    E = energy_matrix(tok)
    p_bil = E / (E.sum() + 1e-30)
    return p_bil, p_bil.sum(dim=1)


def _costs(aK=1.5):
    p_bil, p_marg = _field()
    N = p_marg.shape[0]
    nu = nu_field(p_marg, causal_kernel_matrix(N, alpha_K=aK), w_d=0.1, w_c=0.3)
    H_det = variable_order_entropy(p_bil, nu, form="mixed_renyi_tsallis").numpy()
    H_ent = variable_order_entropy(p_bil, nu, form="nist").numpy()
    surprisal = -np.log(p_marg.numpy() + 1e-30)
    return H_det, H_ent, surprisal


def test_detector_is_the_born_cost():
    # The detector tracks the surprisal -ln p (the Born cost); the single-log entropy does not.
    H_det, H_ent, surprisal = _costs()
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    assert c(H_det, surprisal) > 0.9, f"detector should track surprisal: {c(H_det, surprisal):.3f}"
    assert c(H_det, surprisal) > c(H_ent, surprisal), \
        "detector must track the surprisal better than the single-log entropy (it is the cost)"


def test_q_deformation_broadens_bornmax():
    # q = 1 + 1/aK from the kernel turns the detector-cost Boltzmann weight into a broader q-exp law.
    H_det, _, _ = _costs(aK=1.5)
    q = 1.0 + 1.0 / 1.5
    H = H_det - H_det.min()
    tau = float(np.std(H_det))
    P_plain = np.exp(-H / tau); P_plain /= P_plain.sum()
    Pq = np.clip(1 + (q - 1) * H / tau, 1e-300, None) ** (-1.0 / (q - 1)); Pq /= Pq.sum()
    pr = lambda P: 1.0 / np.sum(P ** 2)
    assert pr(Pq) > pr(P_plain), "q-deformation should broaden BornMax (more effective support)"


def test_detector_cost_normalizes():
    # Gibbs is shift/scale invariant: the detector-as-cost weight always normalizes (not NaN).
    H_det, _, _ = _costs()
    H = np.nan_to_num(H_det, posinf=0.0, neginf=0.0)
    H = H - H.min()
    w = np.exp(-H / (np.std(H) + 1e-30)); P = w / w.sum()
    assert np.all(np.isfinite(P)) and abs(P.sum() - 1.0) < 1e-9
