# S: native Cl(3,0) sympy suite - report

## VERDICT

The Clifford edition holds natively. `check_clifford_native.py`: 104 passed, 0 failed,
0 not ported, exit 0. All 67 matrix-suite labels are ported under the same label, plus
N1-N3, W1-W4, the three-way relation (T1, T2), the D checks that have a native statement,
7 algebra sanity checks (ALG) and 9 REP checks. Two statements hold only under a reading
that must be made explicit (below).

## WHAT RAN

`python check_clifford_native.py` (Python 3.14.2, sympy 1.14.0, about 36 s), last line
`104 passed, 0 failed, 0 not ported, 104 total`, EXIT 0.

- Cl(3,0) built from a blade table computed by word sorting; associativity checked on 512
  triples. No matrix represents an algebra element before block REP; sympy's imaginary
  unit is not used before REP.
- Centre: a complex number of the matrix edition is a multivector in span{1, I}, with
  grades 1, 2 checked to vanish. N(q) is compared as a multivector; |z|^2 = z z~.
- Killing forms from tr(ad ad) on a basis, three of them: over the centre on Cl3 (rank 4,
  value in the centre), real on grades 1+2 (6-dim) and on all of Cl3 (8-dim), and on the
  bivector algebra (3-dim).
- Algebra-independent, rerun unchanged: L15, C4-C6, R1-R5, R7.
- REP: blades -> Pauli words is multiplicative on all 64 pairs, images real-independent;
  dictionary (reversion = dagger, conjugate = adjugate, N = det, scalar part, grade
  splits, e_j) verified there.
- Mutation test (scratch copies): flipping the sign of e_j gives 5 FAILs; making the
  conjugation equal to reversion fails ALG4, ALG5, B2, B4 and then aborts. Exit 1 both.

## WHAT FAILED OR WAS NOT DONE

Nothing failed as the paper states it. Findings printed as NOTE lines:

1. Bell block. In the 64-dimensional real tensor product Cl3 (x)_R Cl3 the statement
   E(a,b) = 4<(a (x) b) rho>_0 = -a.b is FALSE: the 1(x)1 coefficient is 0 and a.b sits on
   I(x)I. There rho is idempotent but not primitive, and (1 + sum e_j(x)e_j)/4 differs from
   (1 - sum s_j(x)s_j)/4. C1-C3 and D17 pass only in the quotient I(x)1 = 1(x)I, the
   tensor product over the centre (32 real dims), which is what the paper writes. REP-C:
   the Kronecker product has rank 32 on the 64-dim algebra, so the matrix edition was
   always the centre-tensor product.
2. Three-way relation. The constant 1/8 holds for kappa over the centre and for the
   bivector-algebra Killing form. With the real Killing form 16<AB>_0 the constant is 1/16.
3. Changes of wording: B13 uses general paravectors E, rho with denominators cleared;
   K7 is checked as kappa(G A Gbar, G B Gbar) = N(G)^2 kappa(A,B) on blade pairs plus 3
   exact random G; P5 was a tautology in the matrix suite and is now checked against the
   Bloch radius of the closure; U3 is solved at n = s_3; R7 uses real s > 0, not s in the
   centre.

## FILES

- `C:\Users\Sam Leizerman\cal-checks\python\check_clifford_native.py` (new)
- `C:\Users\Sam Leizerman\cal-checks\python\clifford_native\results\S.md` (this file)

No existing file was edited.
