"""
Variable-order entropy and causal ν field.

nu_field              — Eq. 34: ν(x,y) = 1 + w_d log(1+p(y)/p(x)) + w_c K(x,y)
causal_kernel_matrix  — Eq. 35: K(x,y) = (1+|x-y|)^{-α_K} for y<x, 0 otherwise
variable_order_entropy — Eq. 30: H^ν(x) = -log ∫ p(y)^{ν(x,y)} φ(x,y) dV_g(y)
hermitian_energy      — Eq. 32: E(x) = Σ_{y≠x} ||H(P_x·P_y)||²
energy_matrix         — Bilocal: E(x,y) = ||H(P_x·P_y)||²  shape (N,N), diagonal zeroed
"""

import torch
from torch import Tensor
from .biquaternion import quat_mul, hermitian, biquat_norm, FDTYPE, CDTYPE


def hermitian_energy(tokens: Tensor) -> Tensor:
    """Position-resolved Hermitian energy (Eq. 32).

    E(x) = Σ_{y≠x} ||H(P_x · P_y)||²

    Args:
        tokens: (N, 4) complex64 — geometric channel tokens.
    Returns:
        E: (N,) float32.
    """
    N = tokens.shape[0]
    Px = tokens.unsqueeze(1).expand(N, N, 4)
    Py = tokens.unsqueeze(0).expand(N, N, 4)
    c = quat_mul(Px, Py)
    Hc = hermitian(c)
    norms_sq = (Hc.real**2 + Hc.imag**2).sum(dim=-1)  # ||H(c)||², (N,N)
    mask = ~torch.eye(N, dtype=torch.bool, device=tokens.device)
    return (norms_sq * mask.float()).sum(dim=1).to(FDTYPE)


def energy_matrix(tokens: Tensor) -> Tensor:
    """Full bilocal Hermitian energy E(x,y) = ||H(P_x·P_y)||²  shape (N,N).

    Same computation as hermitian_energy() but returns the (N,N) matrix before
    summing over y.  Diagonal zeroed (no self-interaction).
    Satisfies: energy_matrix(tokens).sum(dim=1) == hermitian_energy(tokens).
    """
    N = tokens.shape[0]
    Px = tokens.unsqueeze(1).expand(N, N, 4)
    Py = tokens.unsqueeze(0).expand(N, N, 4)
    c = quat_mul(Px, Py)
    Hc = hermitian(c)
    norms_sq = (Hc.real**2 + Hc.imag**2).sum(dim=-1)
    mask = ~torch.eye(N, dtype=torch.bool, device=tokens.device)
    return (norms_sq * mask.float()).to(FDTYPE)


def causal_kernel_matrix(N: int, alpha_K: float = 0.5) -> Tensor:
    """Causal kernel K(x,y) (Eq. 35).

    K(x,y) = (1 + |x-y|)^{-α_K}  for y < x (past)
    K(x,y) = 0                     for y >= x (future/present)

    Returns:
        K: (N, N) float32.
    """
    x = torch.arange(N, dtype=FDTYPE)
    y = torch.arange(N, dtype=FDTYPE)
    dist = (x.unsqueeze(1) - y.unsqueeze(0)).abs()
    K = (1.0 + dist).pow(-alpha_K)
    # Zero out future: y >= x
    causal_mask = y.unsqueeze(0) < x.unsqueeze(1)
    return K * causal_mask.float()


def nu_field(p: Tensor, K: Tensor, w_d: float = 0.5, w_c: float = 0.8,
             flip_density: bool = False) -> Tensor:
    """Causal ν field (Eq. 34).

    ν(x,y) = 1 + w_d log(1 + p(y)/p(x)) + w_c K(x,y)

    Args:
        p: (N,) float32 — belief density (normalized).
        K: (N, N) float32 — causal kernel matrix.
        w_d: density weight.
        w_c: causal weight.
        flip_density: if True, negate the density term (C5 control).
    Returns:
        nu: (N, N) float32.
    """
    N = p.shape[0]
    ratio = p.unsqueeze(0) / p.unsqueeze(1).clamp(min=1e-10)  # p(y)/p(x)
    density_term = w_d * torch.log(1.0 + ratio)
    if flip_density:
        density_term = -density_term
    return 1.0 + density_term + w_c * K


def variable_order_entropy(p: Tensor, nu: Tensor, phi: Tensor = None,
                            form: str = "renyi") -> Tensor:
    """Variable-order entropy (Eq. 30).

    Rényi form (default):
        H^ν(x) = (1/(1-ν̄)) log Σ_y p(y)^{ν(x,y)} φ(x,y)
    Tsallis form:
        S^ν(x) = Σ_y (p(y) - p(y)^{ν(x,y)}) / (ν(x,y) - 1)
    Shannon form (baseline):
        H(x) = -Σ_y p(y) log p(y) φ(x,y)
    NIST bilocal form (NIST poster Section A):
        H^ν(x) = -Σ_y p(x,y)^{ν(x,y)+1} · ln(ν(x,y)) · φ(x,y)
        Pass p as (N,N) bilocal matrix.  H < 0 for ν > 1, H = 0 at ν = 1, H > 0 for ν < 1.
    biquat_doc form (biquat_doc Eq. 30):
        H^ν(x) = -log( Σ_y p(y)^{ν(x,y)} φ(x,y) )
    dual_tsallis bilocal form:
        Both lens -ln(ν) and content q-deformed surprisal -ln_ν(p) inside the integral.
        H^ν(x) = Σ_y p(x,y)^{ν(x,y)} · (-ln_ν(p(x,y))) · (-ln(ν(x,y))) · φ(x,y)
        where ln_ν(p) = (p^{1-ν} - 1)/(1-ν)  [q-deformed log, limit → ln(p) as ν→1].
        Pass p as (N,N) bilocal matrix.
    mixed_renyi_tsallis bilocal form:
        Lens -ln(ν) inside (Tsallis-placed); content moment wrapped in outer log (Rényi-placed).
        H^ν(x) = (1/(1-ν̄(x))) · ln|Σ_y p(x,y)^{ν(x,y)} · (-ln(ν(x,y))) · φ(x,y)|
        where ν̄(x) is the density-weighted mean order at x.
        Pass p as (N,N) bilocal matrix.  Tracks sign separately from magnitude.

    Args:
        p: (N,) or (N,N) float32 — belief density. Use (N,N) bilocal matrix for form="nist".
        nu: (N, N) float32 — variable-order exponent.
        phi: (N, N) float32 — geometric attention kernel. Default: uniform.
        form: "renyi", "tsallis", "shannon", "nist", "biquat_doc",
              "dual_tsallis", or "mixed_renyi_tsallis".
    Returns:
        H: (N,) float32.
    """
    N = nu.shape[0]
    if phi is None:
        phi = torch.ones(N, N, dtype=FDTYPE, device=nu.device) / N

    if form == "nist":
        # Bilocal NIST: p must be (N,N); H = -Σ_y p(x,y)^{ν(x,y)+1} · ln(ν(x,y)) · φ(x,y)
        p_bil = p.clamp(min=1e-10)                                   # (N,N)
        log_nu = torch.log(nu.clamp(min=1e-4))                       # (N,N)
        p_pow = torch.exp((nu + 1.0) * torch.log(p_bil))             # p^{ν+1}
        return -(p_pow * log_nu * phi).sum(dim=-1)                    # (N,)

    if form == "biquat_doc":
        # biquat_doc Eq. 30: H = -log(Σ_y p(y)^{ν(x,y)} φ(x,y))
        p_1d = p if p.dim() == 1 else p.sum(dim=0) / N
        p_c = p_1d.clamp(min=1e-10)
        p_pow = torch.exp(nu * torch.log(p_c.unsqueeze(0)))
        return -torch.log((p_pow * phi).sum(dim=-1).clamp(min=1e-10))

    if form == "dual_tsallis":
        # Both lens and content logs inside the integral (both Tsallis-placed).
        # H = Σ_y p^ν · (-ln_ν(p)) · (-ln(ν)) · φ
        # ln_ν(p) = (p^{1-ν} - 1)/(1-ν); at ν=1 both logs vanish → H=0 (fixed point preserved).
        p_bil = p.clamp(min=1e-10)
        log_nu = torch.log(nu.clamp(min=1e-4))                          # ln(ν), (N,N)
        p_pow_nu = torch.exp(nu * torch.log(p_bil))                     # p^ν
        one_m_nu = (1.0 - nu)                                           # (1-ν)
        p_pow_1mnu = torch.exp(one_m_nu * torch.log(p_bil))             # p^{1-ν}
        # q-deformed log: handle ν≈1 via limit ln_ν(p)→ln(p) when |1-ν|<1e-4.
        # Must preserve sign of (1-ν): negative for ν>1, positive for ν<1.
        near_one = one_m_nu.abs() < 1e-4
        ln_nu_p = torch.where(
            near_one,
            torch.log(p_bil),                                           # limit at ν=1
            (p_pow_1mnu - 1.0) / one_m_nu                              # sign-preserving
        )
        return (p_pow_nu * (-ln_nu_p) * (-log_nu) * phi).sum(dim=-1)   # (N,)

    if form == "mixed_renyi_tsallis":
        # Lens -ln(ν) inside (Tsallis-placed); content wrapped in outer log (Rényi-placed).
        # H = (1/(1-ν̄)) · ln|Σ_y p^ν · (-ln(ν)) · φ|
        # ν̄(x) = density-weighted mean order at x.
        p_bil = p.clamp(min=1e-10)
        log_nu = torch.log(nu.clamp(min=1e-4))                          # ln(ν)
        p_pow_nu = torch.exp(nu * torch.log(p_bil))                     # p^ν
        integrand = (p_pow_nu * (-log_nu) * phi).sum(dim=-1)            # (N,) — may be negative
        p_phi = (p_bil * phi).sum(dim=-1).clamp(min=1e-10)
        nu_bar = (nu * p_bil * phi).sum(dim=-1) / p_phi                 # density-weighted ν̄(x)
        sign = torch.sign(integrand)
        log_abs = torch.log(integrand.abs().clamp(min=1e-10))
        # Preserve sign by multiplying back: outer-log magnitude scaled by sign
        return sign * log_abs / (1.0 - nu_bar).clamp(min=1e-6)         # (N,)

    p_clamped = p.clamp(min=1e-10)

    if form == "shannon":
        H = -(p_clamped.unsqueeze(0) * torch.log(p_clamped.unsqueeze(0)) * phi).sum(dim=-1)
    elif form == "tsallis":
        p_pow = torch.exp(nu * torch.log(p_clamped.unsqueeze(0)))
        H = ((p_clamped.unsqueeze(0) - p_pow) / (nu - 1.0).clamp(min=1e-6) * phi).sum(dim=-1)
    else:  # renyi
        nu_bar = nu.mean(dim=-1)
        p_pow = torch.exp(nu * torch.log(p_clamped.unsqueeze(0)))
        integral = (p_pow * phi).sum(dim=-1)
        H = torch.log(integral.clamp(min=1e-10)) / (1.0 - nu_bar).clamp(min=1e-6)

    return H


def backward_bias(nu: Tensor) -> tuple[Tensor, float]:
    """Compute per-position backward bias and mean.

    For pairs (x,y) with x < y: bias = ν(y,x) − ν(x,y).
    Averaged over all such pairs.

    Args:
        nu: (N, N) float32.
    Returns:
        per_position: (N,) float32 — mean backward bias at each position.
        mean_bias: float — overall mean.
    """
    N = nu.shape[0]
    # For each pair (x,y) with x < y, bias = nu[y,x] - nu[x,y]
    bias_matrix = nu.T - nu  # bias[x,y] = nu[y,x] - nu[x,y]
    upper_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=nu.device), diagonal=1)
    pair_biases = bias_matrix[upper_mask]
    mean_bias = pair_biases.mean().item()

    # Per-position: average backward bias
    per_pos = torch.zeros(N, dtype=FDTYPE, device=nu.device)
    for x in range(N):
        biases = []
        for y in range(x + 1, N):
            biases.append(nu[y, x].item() - nu[x, y].item())
        if biases:
            per_pos[x] = sum(biases) / len(biases)

    return per_pos, mean_bias
