"""
Independent reproduction of the spinor double-cover decomposition (user images 1-2).

Rotating the spectral channel by e^{i theta}, theta = alpha*pi/2, the bilocal interaction
W(alpha)_x = sum_{y!=x} quat_mul(P_x, P_y) with P = geo + spec*e^{i theta} splits into three
theta-harmonics that are alpha-INDEPENDENT biquaternions:

    W(alpha) = W0 + W1 e^{i theta} + W2 e^{2 i theta},
    W0 = geo.geo   (theta-independent),
    W1 = geo.spec + spec.geo   (the MIXED channel, coefficient of e^{i theta}),
    W2 = spec.spec   (pure spectral x spectral, coefficient of e^{2 i theta}).

The Born readout |W(alpha)|^2 then carries two harmonics in alpha:
  - PERIOD-2  from <W0,W2> e^{2 i theta}      (e^{2 i theta} = e^{i pi alpha}, period 2)
  - PERIOD-4  from (<W0,W1>+<W1,W2>) e^{i theta}  (e^{i theta}=e^{i alpha pi/2}, period 4)
with <a,b> = sum_n conj(a_n) b_n the Hermitian inner product over the 4 components.

We correlate each harmonic field (across positions x) with the geometric energy E_x and sweep
alpha in [0,4]. Reproduction criteria (qualitative; seed-dependent magnitudes ok):
  (1) r(period-2, E) has PERIOD 2, zeros near integer alpha, peaks near half-integer.
  (2) r(period-4, E) has PERIOD 4 and SIGN-FLIPS between alpha=0 (+) and alpha=2 (-).
  (3) The period-4 trough alpha* and its phase offset psi are reported EXACTLY:
      r4(alpha) ~ cos(alpha*pi/2 + psi); trough at alpha* = 2 - 2 psi/pi.
      psi = 0  -> flip exactly at Born (alpha=2);  psi != 0 -> offset (and we say why).
"""
import math
import numpy as np
import torch
from cal.biquaternion import quat_mul, embed_tokens_channels, biquat_norm

torch.manual_seed(0)
N = 140
SEED = 42


def channel_harmonics(geo, spec):
    """Return per-position biquaternion harmonics W0, W1, W2 (each (N,4) complex)."""
    n = geo.shape[0]
    Gx = geo.unsqueeze(1).expand(n, n, 4); Gy = geo.unsqueeze(0).expand(n, n, 4)
    Sx = spec.unsqueeze(1).expand(n, n, 4); Sy = spec.unsqueeze(0).expand(n, n, 4)
    gg = quat_mul(Gx, Gy)
    gs = quat_mul(Gx, Sy) + quat_mul(Sx, Gy)
    ss = quat_mul(Sx, Sy)
    mask = (~torch.eye(n, dtype=torch.bool)).unsqueeze(-1)
    W0 = (gg * mask).sum(dim=1)
    W1 = (gs * mask).sum(dim=1)
    W2 = (ss * mask).sum(dim=1)
    return W0, W1, W2


def herm_inner(a, b):
    """<a,b> = sum_n conj(a_n) b_n  -> (N,) complex."""
    return (a.conj() * b).sum(dim=-1)


def _corr(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    return float(np.corrcoef(u, v)[0, 1])


def main():
    geo, spec = embed_tokens_channels(N, base_seed=SEED)
    W0, W1, W2 = channel_harmonics(geo, spec)

    # per-position complex cross-amplitudes
    b = herm_inner(W0, W2).numpy()                       # period-2 carrier (x e^{2 i th})
    a = (herm_inner(W0, W1) + herm_inner(W1, W2)).numpy()  # period-4 carrier (x e^{i th})
    E = (biquat_norm(W0) ** 2).numpy()                   # geometric energy reference

    # closed-form phase offset of the period-4 correlation: r4 ~ cos(alpha pi/2 + psi)
    cR = np.cov(np.real(a), E)[0, 1]
    cI = np.cov(np.imag(a), E)[0, 1]
    psi = math.atan2(cI, cR)
    alpha_trough = (math.pi - psi) / (math.pi / 2)       # where cos(.)= -1, mod 4
    alpha_trough = alpha_trough % 4.0

    print("=" * 78)
    print(f"Double-cover reproduction  (N={N}, seed={SEED})")
    print("=" * 78)
    print(f"  {'alpha':>6} {'r(period2,E)':>14} {'r(period4,E)':>14}")
    for al in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        th = al * math.pi / 2
        p2 = np.real(b * np.exp(2j * th))
        p4 = np.real(a * np.exp(1j * th))
        print(f"  {al:>6.2f} {_corr(p2, E):>14.4f} {_corr(p4, E):>14.4f}")

    # fine-grid trough of |r4| sign (numeric check of the closed form)
    grid = np.linspace(0, 4, 4001)
    r4 = np.array([_corr(np.real(a * np.exp(1j * g * math.pi / 2)), E) for g in grid])
    num_trough = grid[int(np.argmin(r4))]

    print()
    print(f"  period-4 sign:  alpha=0 -> {_corr(np.real(a), E):+.4f}   "
          f"alpha=2 -> {_corr(np.real(a*np.exp(1j*math.pi)), E):+.4f}   (flip expected)")
    print(f"  phase offset psi = {psi:+.4f} rad ({math.degrees(psi):+.2f} deg)")
    print(f"  period-4 trough: closed-form alpha* = {alpha_trough:.4f},  "
          f"numeric = {num_trough:.4f}")
    print(f"  => flip is {'EXACTLY at alpha=2' if abs(alpha_trough-2) < 0.02 else 'OFFSET from alpha=2'}"
          f"  (offset driven by cov(Im a, E) = {cI:+.3e}; zero -> no offset)")
    print()
    print("Reproduction holds if: r(period2) ~ period 2 (zeros at integer alpha), and")
    print("r(period4) ~ period 4 sign-flipping + at alpha=0 to - at alpha=2.")


if __name__ == "__main__":
    main()
