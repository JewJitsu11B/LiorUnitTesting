"""
Test B, step 1 (de-risked foundation): does path-accumulated transport (HOLONOMY) lift the
INTEGRATED field off the null cone (det != 0), while a flat connection leaves it on the cone?

This is the load-bearing assumption of the whole closed-loop test. Mechanism: a single transport
is conjugation X -> U X U^{-1}, which PRESERVES det, so a single null element stays null. But the
SUM of nulls transported by DIFFERENT accumulated holonomies (each rotated to a different null
orientation) spans the space and acquires det != 0. With a flat connection (identity transport)
aligned nulls sum to a null aggregate (det = 0). This is where the TIME axis enters: the holonomy
is the path-ordered product accumulated along a worldline of T steps.

Setup: T copies of one genuine zero divisor z = 1 + h i (det X_z = 0, from the two-closure test).
Curved connection from random field tokens; accumulate U_s = P_s ... P_0 (path-ordered) along T
time steps; transport each copy X_s = U_s X_z U_s^{-1} (det 0 each); sum -> aggregate.
Flat control: U_s = I -> aggregate = T X_z -> det 0.  Deterministic.
"""
import torch
from cal.biquaternion import biquat_to_matrix, embed_tokens_channels, CDTYPE
from cal.propagator import causal_propagator

torch.manual_seed(0)
T = 60
EPS = 0.3


def det2(M):
    return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]


def main():
    z = torch.tensor([1.0 + 0j, 1j, 0.0, 0.0], dtype=CDTYPE)   # zero divisor, det X_z = 0
    Xz = biquat_to_matrix(z)
    print("=" * 78)
    print(f"Test B step 1: holonomy lift off the null cone   (T={T} time steps, eps={EPS})")
    print("=" * 78)
    print(f"  single null element z=(1,h,0,0):  |det(X_z)| = {det2(Xz).abs().item():.3e}  (on the cone)")

    geo, spec = embed_tokens_channels(T, base_seed=7)
    Ps = causal_propagator(geo + spec, EPS)        # (T,2,2) step propagators along the worldline

    for label, flat in [("holonomy", False), ("flat", True)]:
        agg = torch.zeros(2, 2, dtype=CDTYPE)
        U = torch.eye(2, dtype=CDTYPE)
        worst_each = 0.0
        for s in range(T):
            if not flat:
                U = Ps[s] @ U                       # path-ordered accumulation = holonomy
            Xs = U @ Xz @ torch.linalg.inv(U)       # transported copy; conjugation preserves det
            worst_each = max(worst_each, det2(Xs).abs().item())
            agg = agg + Xs
        print(f"  {label:>9}:  max|det(each transported copy)| = {worst_each:.3e}   "
              f"|det(SUM)| = {det2(agg).abs().item():.3e}")

    print()
    print("Pass = each transported copy stays ~0 (conjugation preserves det) in BOTH cases, but the")
    print("AGGREGATE has det>0 under holonomy (spans, off the cone) and ~0 flat (stays aligned, on it).")


if __name__ == "__main__":
    main()
