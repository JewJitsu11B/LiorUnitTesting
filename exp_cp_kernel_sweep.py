"""
Experiment: where does the CP-type (arrow / P-odd,T-odd) asymmetry originate, and
how does it emerge as the memory kernel widens?

Two candidate origins (the user's question):
  (A) the rank-2 FORCING at a point  -> the bare biquaternion algebra C(x)H = M2(C).
      M2(C) is ASSOCIATIVE, so the pointwise associator [A,B,C] = (AB)C - A(BC) = 0.
      => the pointwise forcing carries NO asymmetry. (control)
  (B) INTEGRATION of the hybrid source current over space-time -> the kernel-weighted
      causal transport / memory integral. This is the only place an arrow can live.

The memory kernel decay alpha_K is the dial between them:
  alpha_K -> large  = narrow / Markovian kernel = NO memory integration ~ pathway (A)
  alpha_K -> small  = wide kernel = deep memory integration             ~ pathway (B)

Measures (all operational, stated so a reviewer can check):
  1. pointwise algebraic associator  ||[A,B,C]||      (control, expect ~0)
  2. nu-field backward bias  mean( nu(y,x) - nu(x,y) ) (T-odd arrow; entropy.backward_bias)
  3. transport time-asymmetry ||Y_forward - Y_reversed|| (propagator holonomy, T-odd)

CP vs CPT safeguard: measure 2 must be T-ODD (flip sign under sequence reversal).
That makes it CP-type content (an arrow), not a CPT violation. CPT (= P*C*T on the
channel) leaves it invariant; we check the T-odd flip explicitly.
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens, quat_mul, quat_conj, biquat_norm
from cal.propagator import transport_biquat
from cal.cal_tensor import power_law_kernel
from cal.entropy import energy_matrix, nu_field, causal_kernel_matrix, backward_bias

torch.manual_seed(0)
N = 40
AKS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3]
W_D, W_C = 0.1, 0.3   # nu-field channel weights (match the validated suite)


def associator(A, B, C):
    return quat_mul(quat_mul(A, B), C) - quat_mul(A, quat_mul(B, C))


def pointwise_associator_mean(toks):
    """Pathway (A) control: bare algebra at a point. M2(C) associative => ~0."""
    vals = [biquat_norm(associator(toks[i], toks[i + 1], toks[i + 2])).item()
            for i in range(len(toks) - 2)]
    return float(np.mean(vals))


def _marginal(toks):
    E = energy_matrix(toks)
    E = E.real if torch.is_complex(E) else E
    p_bil = E / (E.sum() + 1e-30)
    return p_bil.sum(dim=1)


def nu_backward_bias(toks, aK, reverse=False):
    """Measure 2: nu-field arrow nu(y,x)-nu(x,y), the T-odd asymmetry."""
    t = torch.flip(toks, dims=[0]) if reverse else toks
    p_marg = _marginal(t)
    K = causal_kernel_matrix(N, aK)
    nu = nu_field(p_marg, K, w_d=W_D, w_c=W_C)
    _, mean_bias = backward_bias(nu)
    return mean_bias


def transport_time_asymmetry(toks, aK, eps=0.05):
    """Measure 3: ||transport(forward order) - transport(reversed order)|| through the
    SAME tokens with the SAME per-token weights. This is the order- (time-) dependence
    of the memory integral, i.e. a propagator holonomy. Uses RAW (unnormalized) kernel
    weights so a wider kernel (small alpha_K) integrates genuinely MORE coupling, rather
    than redistributing a fixed normalized total."""
    lags = torch.arange(1, N + 1, dtype=torch.float32)
    w_raw = (1.0 + lags) ** (-aK)   # unnormalized memory weight per token
    Y0 = embed_tokens(1, base_seed=999)[0]
    Yf = Y0.clone()
    for k in range(N):
        Yf = transport_biquat(Yf, toks[k], eps * w_raw[k].item())
    Yb = Y0.clone()
    for k in range(N - 1, -1, -1):
        Yb = transport_biquat(Yb, toks[k], eps * w_raw[k].item())
    return biquat_norm(Yf - Yb).item()


def main():
    toks = embed_tokens(N, base_seed=42)

    print("=" * 72)
    print("PATHWAY (A) control: pointwise rank-2 forcing (bare algebra at a point)")
    print("=" * 72)
    pw = pointwise_associator_mean(toks)
    print(f"  mean ||[A,B,C]|| pointwise associator = {pw:.3e}")
    print(f"  -> {'~0 (associative): forcing carries NO asymmetry' if pw < 1e-4 else 'NONZERO (unexpected)'}")

    print()
    print("=" * 72)
    print("PATHWAY (B): integration of source current over space-time (kernel sweep)")
    print("=" * 72)
    print(f"  {'alpha_K':>8} {'|nu backward bias|':>20} {'transport T-asym':>18}")
    rows = []
    for aK in AKS:
        bias = nu_backward_bias(toks, aK)
        tasym = transport_time_asymmetry(toks, aK)
        rows.append((aK, bias, tasym))
        print(f"  {aK:>8.2f} {abs(bias):>20.4e} {tasym:>18.4e}")

    print()
    print("=" * 72)
    print("READOUT")
    print("=" * 72)
    # emergence: ratio of widest-kernel to narrowest-kernel asymmetry
    b_wide, b_narrow = abs(rows[-1][1]), abs(rows[0][1]) + 1e-30
    t_wide, t_narrow = rows[-1][2], rows[0][2] + 1e-30
    print(f"  nu-bias  amplification (alpha_K {AKS[0]} -> {AKS[-1]}): x{b_wide / b_narrow:.1f}")
    print(f"  transport amplification (alpha_K {AKS[0]} -> {AKS[-1]}): x{t_wide / t_narrow:.1f}")

    # T-odd (CP-type, not CPT) check: reverse the causal ARROW by transposing the kernel
    # (K is lower-triangular = forward causation; K.T = reversed causation). The arrow
    # must flip sign. Reversing the kernel direction is the correct time-reversal here,
    # NOT reversing the token sequence (which leaves the kernel's direction untouched).
    aK_mid = 0.5
    p_marg = _marginal(toks)
    K = causal_kernel_matrix(N, aK_mid)
    _, bias_fwd = backward_bias(nu_field(p_marg, K, w_d=W_D, w_c=W_C))
    _, bias_rev = backward_bias(nu_field(p_marg, K.T.contiguous(), w_d=W_D, w_c=W_C))
    print(f"\n  CP-vs-CPT safeguard (alpha_K={aK_mid}, reverse = transpose kernel arrow):")
    print(f"    backward bias, forward arrow  (K)   = {bias_fwd:+.4e}")
    print(f"    backward bias, reversed arrow (K.T) = {bias_rev:+.4e}")
    flips = (bias_fwd * bias_rev) < 0
    sym = abs(abs(bias_fwd) - abs(bias_rev)) / (max(abs(bias_fwd), abs(bias_rev)) + 1e-30)
    print(f"    flips sign under arrow reversal? {flips}   (magnitude match: {1 - sym:.1%})")
    print(f"    -> {'arrow is T-ODD => CP-type content, CPT-even (consistent w/ the theorem)' if flips else 'did not cleanly flip; inspect'}")


if __name__ == "__main__":
    main()
