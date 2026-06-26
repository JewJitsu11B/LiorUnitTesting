"""
Test A: five-way comparison of entropy/detector functionals vs surprisal (STATIC field).

Caveat carried from the session: this is the static, single-snapshot, on-null-cone field that
we already established is the WRONG field for the Born-cost claim (clamp-inflated, nu_bar-erratic).
It is run here because it is the field the manuscript's existing 0.984 number lives on, so it
tells us what to correct that number to. The closed-loop version (Test B) adds the time axis.

Five conditions, each correlated with the surprisal -ln p across positions, swept over nu_bar:
  1. shannon          fixed-order -Sum p ln p              (benchmark: static/fixed)
  2. fixed_nist       form=nist, CONSTANT nu = nu_bar      (framework entropy, field OFF)
  3. var_entropy      form=nist, VARIABLE nu-field         (framework entropy H^nu)
  4. detector         form=mixed_renyi_tsallis             (raw and clamped reported)
  5. both             multiple-R of surprisal on [var_entropy, detector_clamped]

The multiple-R for (5) is the principled "using both together" measure:
  R^2 = (r1^2 + r2^2 - 2 r1 r2 r12) / (1 - r12^2),
with r1=corr(entropy,surp), r2=corr(detector,surp), r12=corr(entropy,detector).
"""
import numpy as np
import torch
from cal.biquaternion import embed_tokens_channels
from cal.entropy import energy_matrix, causal_kernel_matrix, nu_field, variable_order_entropy

torch.manual_seed(0)
N = 140
W_D, W_C, A_K = 0.1, 0.3, 1.0


def _field(seed=42):
    torch.manual_seed(seed)
    geo, spec = embed_tokens_channels(N, base_seed=seed)
    E = energy_matrix(geo + spec)
    E = E.real if torch.is_complex(E) else E
    p = E / (E.sum() + 1e-30)
    return p, p.sum(dim=1)


def _nu_bar(nu, p_bil, phi):
    p_c = p_bil.clamp(min=1e-10)
    p_phi = (p_c * phi).sum(dim=-1).clamp(min=1e-10)
    return (nu * p_c * phi).sum(dim=-1) / p_phi


def detector_raw(p_bil, nu, phi):
    log_nu = torch.log(nu.clamp(min=1e-4))
    p_c = p_bil.clamp(min=1e-10)
    p_pow = torch.exp(nu * torch.log(p_c))
    integrand = (p_pow * (-log_nu) * phi).sum(dim=-1)
    nu_bar = _nu_bar(nu, p_bil, phi)
    sign = torch.sign(integrand)
    log_abs = torch.log(integrand.abs().clamp(min=1e-10))
    return (sign * log_abs / (1.0 - nu_bar)).numpy()


def _corr(a, b):
    a = np.nan_to_num(np.asarray(a, float), posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, float), posinf=0.0, neginf=0.0)
    s = a.std() * b.std()
    return float(np.corrcoef(a, b)[0, 1]) if s > 1e-30 else 0.0


def _multiple_R(r1, r2, r12):
    denom = 1.0 - r12 ** 2
    if abs(denom) < 1e-9:
        return float("nan")
    R2 = (r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * r12) / denom
    return float(np.sqrt(max(R2, 0.0)))


def main():
    p_bil, p_marg = _field()
    surp = -np.log(p_marg.numpy() + 1e-30)
    phi = torch.ones(N, N, dtype=torch.float32) / N
    K = causal_kernel_matrix(N, A_K)
    nu0 = nu_field(p_marg, K, w_d=W_D, w_c=W_C)
    nubar0 = float(_nu_bar(nu0, p_bil, phi).mean())

    print("=" * 100)
    print(f"Test A: five-way corr(., surprisal) on the STATIC field, swept over nu_bar")
    print(f"(N={N}, natural nu_bar={nubar0:.4f}; static/on-null-cone field, carries the wrong-field caveat)")
    print("=" * 100)
    print(f"  {'nu_bar':>7} {'1.shannon':>10} {'2.fixed':>9} {'3.var_ent':>10} "
          f"{'4.det_raw':>10} {'4.det_clmp':>11} {'5.both(R)':>10}")

    targets = [0.50, 0.70, 0.90, round(nubar0, 3), 1.00, 1.05, 1.10, 1.30, 1.50]
    for t in sorted(set(targets)):
        nu = (nu0 + (t - nubar0)).clamp(min=1e-4)
        nu_const = torch.full((N, N), float(t), dtype=torch.float32)
        # per-position row Shannon -Sum_y p(x,y) ln p(x,y) (nu-independent, varies with x)
        pc = p_bil.clamp(min=1e-12)
        H_sh = (-(pc * torch.log(pc)).sum(dim=1)).numpy()
        H_fix = variable_order_entropy(p_bil, nu_const, phi, form="nist").numpy()
        H_var = variable_order_entropy(p_bil, nu, phi, form="nist").numpy()
        H_detc = variable_order_entropy(p_bil, nu, phi, form="mixed_renyi_tsallis").numpy()
        H_detr = detector_raw(p_bil, nu, phi)
        r_sh = _corr(H_sh, surp); r_fix = _corr(H_fix, surp)
        r_var = _corr(H_var, surp); r_detr = _corr(H_detr, surp); r_detc = _corr(H_detc, surp)
        R_both = _multiple_R(r_var, r_detc, _corr(H_var, H_detc))
        mark = "  <- natural" if abs(t - nubar0) < 1e-3 else ""
        print(f"  {t:>7.3f} {r_sh:>10.4f} {r_fix:>9.4f} {r_var:>10.4f} "
              f"{r_detr:>10.4f} {r_detc:>11.4f} {R_both:>10.4f}{mark}")

    print()
    print("Read: benchmarks 1-2 are the floor; 3 (variable-order entropy) and 4 (detector) are")
    print("the framework functionals; 5 is how much surprisal-variance BOTH explain together.")
    print("det_clmp vs det_raw exposes the clamp inflation. Deterministic.")


if __name__ == "__main__":
    main()
