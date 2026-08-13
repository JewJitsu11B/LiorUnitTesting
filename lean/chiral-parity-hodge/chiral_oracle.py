"""
Independent oracle for the chiral-parity paper (replicating chiral_projector_harness.py from
the paper's spec), and a cross-check that the Lean model's INVERSION-COUNT sign equals the
paper's PERMUTATION sign.

Middle grade k=m of n=2m. Blades = m-subsets of range(n). Signature: negatives = set S, q=|S|.
  star e_I = eps_I * sgn(I, Ic) * e_Ic,   eps_I = prod_{i in I} sgn_i,
  sgn(I,Ic) = sign of permutation taking (0..n-1) to (sorted I ++ sorted Ic).
Lemma 3.1: star^2 = (-1)^{k(n-k)} (-1)^q id = (-1)^{m^2+q} id at middle grade.
"""
import itertools, numpy as np
from math import comb

def perm_sign(perm):
    # sign of a permutation given as a sequence (parity of inversions)
    n=len(perm); s=1
    for i in range(n):
        for j in range(i+1,n):
            if perm[i]>perm[j]: s=-s
    return s

def shuffle_sign(I, Ic):
    # sign of permutation carrying (0..n-1) to (sorted I ++ sorted Ic)
    return perm_sign(sorted(I)+sorted(Ic))

def inv_count_sign(I, Ic):
    # Lean model's sign: (-1)^{#(a in I, b in Ic, a>b)}
    c=sum(1 for a in I for b in Ic if a>b)
    return (-1)**c

def eps(I, negset):
    return (-1)**len(set(I)&negset)

def build_star(m, negset):
    n=2*m
    blades=list(itertools.combinations(range(n),m))
    idx={b:i for i,b in enumerate(blades)}
    N=len(blades)
    star=np.zeros((N,N))
    sign_match=True
    for I in blades:
        Ic=tuple(sorted(set(range(n))-set(I)))
        sp=shuffle_sign(I,Ic)          # paper's permutation sign
        si=inv_count_sign(I,Ic)        # Lean model's inversion-count sign
        if sp!=si: sign_match=False
        star[idx[Ic], idx[I]] = eps(I,negset)*sp   # column I -> row Ic
    return star, sign_match, N

print(f"{'q':>2} {'sign_match':>10} {'star^2==s*I':>12} {'s(=(-1)^(m^2+q))':>16} {'rankP+':>7} {'rankP-':>7}")
m=4  # n=8, middle grade k=4, C(8,4)=70
for q in range(0, 9):
    negset=set(range(q))                       # first q directions negative
    star, sign_match, N = build_star(m, negset)
    s2 = star@star
    s_expected = (-1)**((m*m)+q)
    is_scalar = np.allclose(s2, s_expected*np.eye(N))
    Pp = 0.5*(np.eye(N)+star); Pm = 0.5*(np.eye(N)-star)
    rp = np.linalg.matrix_rank(Pp); rm = np.linalg.matrix_rank(Pm)
    print(f"{q:>2} {str(sign_match):>10} {str(is_scalar):>12} {s_expected:>16} {rp:>7} {rm:>7}")
print(f"\ndim Lambda^{m}(R^{2*m}) = C({2*m},{m}) = {comb(2*m,m)}")
print("Expected (paper Table 1): q even -> s=+1, ranks 35/35 (chiral pair);")
print("                          q odd  -> s=-1, ranks 70/70 (both full rank).")
