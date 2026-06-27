"""
Quantify the period-4 trough displacement vs energy reference, and test whether any
PRINCIPLED reference pins the trough to alpha=2.

The period-4 correlation is r4(alpha) = corr_x( Re(a_x e^{i theta}), E_x ), theta = alpha*pi/2,
with the period-4 carrier  a = <W0,W1> + <W1,W2>  (Hermitian inner product over the 4
biquaternion components). Its trough is

    alpha* = 2 - (2/pi) * psi,    psi = atan2( cov(Im a, E), cov(Re a, E) ).

So alpha* = 2  (the Born/Euler point)  iff  psi = 0  iff  cov(Im a, E) = 0.

The earlier run used E = ||W0||^2 and got psi = 74 deg, alpha* ~ 1.0-1.2: displaced. Here we
sweep candidate FIXED per-position references E -- including the paper's "Hermitian energy"
(cal.entropy.hermitian_energy) and the q.q-dagger Born channel -- to settle empirically
whether any natural reference pins the trough to alpha=2. Closed-form alpha* (trough of the
covariance) and numeric alpha* (trough of the normalized correlation, what the figure plots)
are both reported. Deterministic.
"""
import math
import numpy as np
import torch
from cal.biquaternion import (quat_mul, embed_tokens_channels, biquat_norm,
                              hermitian, anti_hermitian, quat_norm_sq)
from cal.entropy import hermitian_energy, energy_matrix

torch.manual_seed(0)
N = 140
SEED = 42


def channel_harmonics(geo, spec):
    n = geo.shape[0]
    Gx = geo.unsqueeze(1).expand(n, n, 4); Gy = geo.unsqueeze(0).expand(n, n, 4)
    Sx = spec.unsqueeze(1).expand(n, n, 4); Sy = spec.unsqueeze(0).expand(n, n, 4)
    gg = quat_mul(Gx, Gy); gs = quat_mul(Gx, Sy) + quat_mul(Sx, Gy); ss = quat_mul(Sx, Sy)
    mask = (~torch.eye(n, dtype=torch.bool)).unsqueeze(-1)
    return (gg * mask).sum(1), (gs * mask).sum(1), (ss * mask).sum(1)


def herm_inner(a, b):
    return (a.conj() * b).sum(-1)


def cov(u, v):
    return float(np.cov(np.asarray(u, float), np.asarray(v, float))[0, 1])


def trough_for(a, E):
    E = np.asarray(E, float)
    cR, cI = cov(np.real(a), E), cov(np.imag(a), E)
    psi = math.atan2(cI, cR)
    al_closed = (2.0 - (2.0 / math.pi) * psi) % 4.0
    grid = np.linspace(0, 4, 4001)
    r4 = [np.corrcoef(np.real(a * np.exp(1j * g * math.pi / 2)), E)[0, 1] for g in grid]
    al_num = float(grid[int(np.argmin(r4))])
    return psi, al_closed, al_num, cI, float(np.corrcoef(np.real(a), E)[0, 1])


def main():
    geo, spec = embed_tokens_channels(N, base_seed=SEED)
    W0, W1, W2 = channel_harmonics(geo, spec)
    a = (herm_inner(W0, W1) + herm_inner(W1, W2)).numpy()
    tok = geo + spec

    refs = {
        "geo ||W0||^2  (= Born wt of W0)": (biquat_norm(W0) ** 2).numpy(),
        "hermitian_energy(tok) [PAPER]": hermitian_energy(tok).numpy(),
        "|reduced norm W0| (det/metric)": quat_norm_sq(W0).abs().numpy(),
        "||A(W0)||^2 (anti-herm/spectral)": (biquat_norm(anti_hermitian(W0)) ** 2).numpy(),
        "||H(W0)||^2 (herm part)": (biquat_norm(hermitian(W0)) ** 2).numpy(),
        "energy_matrix(tok) rowsum": energy_matrix(tok).sum(1).numpy(),
    }

    print("=" * 96)
    print(f"Period-4 trough vs energy reference   (N={N}, seed={SEED})")
    print("trough alpha* = 2  iff  cov(Im a, E) = 0   (psi = 0)")
    print("=" * 96)
    print(f"  {'reference E':>33} {'psi(deg)':>9} {'a*_closed':>10} {'a*_num':>8} "
          f"{'cov(Im a,E)':>12} {'r4(a=0)':>9}")
    for name, E in refs.items():
        psi, alc, aln, cI, r0 = trough_for(a, E)
        flag = "  <-- pins to 2" if abs(aln - 2.0) < 0.05 else ""
        print(f"  {name:>33} {math.degrees(psi):>9.2f} {alc:>10.3f} {aln:>8.3f} "
              f"{cI:>12.3e} {r0:>9.4f}{flag}")
    print()
    print("Seed-robustness of the displacement (paper ref = hermitian_energy):")
    print(f"  {'seed':>6} {'psi(deg)':>9} {'a*_num':>8} {'off alpha=2?':>12}")
    for sd in [1, 7, 42, 123, 2024, 31337]:
        g2, s2 = embed_tokens_channels(N, base_seed=sd)
        w0, w1, w2 = channel_harmonics(g2, s2)
        a2 = (herm_inner(w0, w1) + herm_inner(w1, w2)).numpy()
        E2 = hermitian_energy(g2 + s2).numpy()
        psi, _, aln, _, _ = trough_for(a2, E2)
        off = "yes" if abs(aln - 2.0) > 0.1 else "NO"
        print(f"  {sd:>6} {math.degrees(psi):>9.2f} {aln:>8.3f} {off:>12}")
    print()
    print("Read: a reference with cov(Im a,E) ~ 0 and alpha* ~ 2 pins the trough to the Born")
    print("point and vindicates the |cos(pi alpha/2)| caption. If NONE do, the trough genuinely")
    print("displaces, the 0.210 residual is real, and the caption needs the displacement note.")


if __name__ == "__main__":
    main()
