import ChiralParity.ParityCriterion
import ChiralParity.RankSplit

/-!
# Corollary 8.1 (classical four-dimensional case)

The whole chiral-projector model is parametric in `m`; the physical four-dimensional case is
`m = 2`, so `n = 2m = 4`, middle grade `k = 2`, carrier `Λ²(ℝ⁴)` of dimension `C(4,2) = 6`.

Because `k(n-k) = 2·2 = 4` is even, `m² = 4` is even, and the star-square scalar reduces to
`s = (-1)^{m²+q} = (-1)^q`. Hence:

* **Euclidean / split** (`q` even, i.e. `q ∈ {0,2,4}`): `⋆² = +id`, a genuine **real** chiral
  pair; the carrier splits `6 = 3 ⊕ 3`.
* **Lorentzian** (`q` odd, i.e. `q ∈ {1,3}`): `⋆² = −id`, so `⋆` is a **complex structure**
  and there is no real projector pair.

Everything below is the `m = 4` (`n = 8`) development of `ParityCriterion.lean` / `RankSplit.lean`
re-run at `m = 2`.
-/

namespace ChiralParity

open Matrix Finset
open Module (finrank)

/-- Item 1: at `n = 4` (`m = 2`) there are `C(4,2) = 6` middle-grade blades. -/
theorem card_Blade2 : Fintype.card (Blade 2) = 6 := by
  rw [card_Blade]; decide

/-- Item 2: the four-dimensional chirality condition. At `m = 2`, `m² = 4` is even, so the
star-square scalar is `+1` exactly when the negative-signature count `q` is even. -/
theorem starScalar_4d (S : Finset (Fin (2 * 2))) :
    starScalar S = 1 ↔ Even (qneg S) := by
  rw [starScalar_eq_one_iff]
  have h4 : (2 : ℕ) ^ 2 = 4 := by norm_num
  have h4e : Even (4 : ℕ) := by decide
  rw [h4, Nat.even_add]
  simp [h4e]

/-- Item 3: in the Euclidean / split case (`q` even) `⋆² = +id` — the real chiral regime. -/
theorem star4d_sq_even (S : Finset (Fin (2 * 2))) (hq : Even (qneg S)) :
    starMat S * starMat S = 1 := by
  rw [starMat_sq, (starScalar_4d S).mpr hq, one_smul]

/-- Item 4: in the Lorentzian case (`q` odd) `⋆² = −id` — the Hodge star is a **complex
structure**, so no real projector pair exists. -/
theorem star4d_sq_odd (S : Finset (Fin (2 * 2))) (hq : Odd (qneg S)) :
    starMat S * starMat S = -1 := by
  have hs : starScalar S = -1 := by
    rw [starScalar_eq_neg_one_iff]
    have h4 : (2 : ℕ) ^ 2 = 4 := by norm_num
    rw [h4]
    exact (by decide : Even 4).add_odd hq
  rw [starMat_sq, hs, neg_one_smul]

/-- Item 5: in the real regime (`q` even) `P₊` is idempotent. -/
theorem Pplus4d_idem (S : Finset (Fin (2 * 2))) (hq : Even (qneg S)) :
    Pplus S * Pplus S = Pplus S :=
  Pplus_sq_of_even S ((starScalar_4d S).mpr hq)

/-- Item 6a: `P₊` carries trace `3 = ½·6` at `n = 4`. -/
theorem trace_Pplus_4d (S : Finset (Fin (2 * 2))) : (Pplus S).trace = 3 := by
  unfold Pplus
  rw [Matrix.trace_smul, Matrix.trace_add, trace_starMat (by norm_num) S, add_zero,
    Matrix.trace_one, card_Blade2, smul_eq_mul]
  norm_num

/-- Item 6b: `P₋` carries trace `3 = ½·6` at `n = 4`. -/
theorem trace_Pminus_4d (S : Finset (Fin (2 * 2))) : (Pminus S).trace = 3 := by
  unfold Pminus
  rw [Matrix.trace_smul, Matrix.trace_sub, trace_starMat (by norm_num) S, sub_zero,
    Matrix.trace_one, card_Blade2, smul_eq_mul]
  norm_num

/-- **Item 7 (Corollary 8.1, real split).** In the Euclidean / split case (`q` even) the two
chiral projectors are genuine complementary idempotents of rank `3` each: the middle-grade
carrier splits as `6 = 3 ⊕ 3`. -/
theorem rank_split_4d (S : Finset (Fin (2 * 2))) (hq : Even (qneg S)) :
    (Pplus S).rank = 3 ∧ (Pminus S).rank = 3 := by
  refine ⟨?_, ?_⟩
  · have hbridge : ((Pplus S).rank : ℚ) = (Pplus S).trace :=
      rank_eq_trace_of_idempotent (Pplus S) (Pplus4d_idem S hq)
    rw [trace_Pplus_4d] at hbridge
    exact_mod_cast hbridge
  · have hbridge : ((Pminus S).rank : ℚ) = (Pminus S).trace :=
      rank_eq_trace_of_idempotent (Pminus S)
        (Pminus_sq_of_even S ((starScalar_4d S).mpr hq))
    rw [trace_Pminus_4d] at hbridge
    exact_mod_cast hbridge

end ChiralParity
