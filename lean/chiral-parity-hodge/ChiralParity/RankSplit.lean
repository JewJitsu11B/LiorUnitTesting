import ChiralParity.EqualSplit
import ChiralParity.ParityCriterion

/-!
# Proposition 4.1 (rank / dimension split): `70 = 35 ⊕ 35`

This upgrades the trace-level "equal split" of `EqualSplit.lean` to the concrete
**rank / dimension** split of the middle-grade carrier at `n = 8`, `m = 4`.

* Step A (trace values). The blade index `Blade 4` has `Fintype.card = C(8,4) = 70`, and each
  chiral projector `P₊ = ½(1+⋆)`, `P₋ = ½(1−⋆)` carries trace `35` (the star is traceless).
* Step B (the real upgrade: rank = trace). Over the field `ℚ`, an idempotent matrix `P`
  (`P*P = P`) satisfies `rank P = trace P`. On the chiral carrier this is available exactly when
  `s = starScalar S = 1`, i.e. `Even (qneg S)` (`carrier_condition`). Hence
  `rank P₊ = rank P₋ = 35` and `rank P₊ + rank P₋ = 70`: the carrier splits as `70 = 35 ⊕ 35`.

The rank = trace bridge is assembled from `LinearMap.IsProj.trace`
(`trace = finrank` of the range of a projection), the definition of `Matrix.rank`
(`finrank` of the range of `mulVecLin`), and the `LinearMap.trace` ↔ `Matrix.trace`
identification through the standard basis.
-/

namespace ChiralParity

open Matrix Finset
open Module (finrank)

variable {m : ℕ}

/-! ## Step A — the trace values -/

/-- The middle-grade blade index has `C(2m, m)` elements. -/
theorem card_Blade (m : ℕ) : Fintype.card (Blade m) = Nat.choose (2 * m) m := by
  show Fintype.card {s : Finset (Fin (2 * m)) // s.card = m} = Nat.choose (2 * m) m
  rw [Fintype.card_finset_len, Fintype.card_fin]

/-- At `n = 8` (`m = 4`) there are `C(8,4) = 70` middle-grade blades. -/
theorem card_Blade4 : Fintype.card (Blade 4) = 70 := by
  rw [card_Blade]; decide

/-- `P₊` carries trace `35` at `n = 8`. -/
theorem trace_Pplus_n8 (S : Finset (Fin (2 * 4))) : (Pplus S).trace = 35 := by
  unfold Pplus
  rw [Matrix.trace_smul, Matrix.trace_add, trace_starMat (by norm_num) S, add_zero,
    Matrix.trace_one, card_Blade4, smul_eq_mul]
  norm_num

/-- `P₋` carries trace `35` at `n = 8`. -/
theorem trace_Pminus_n8 (S : Finset (Fin (2 * 4))) : (Pminus S).trace = 35 := by
  unfold Pminus
  rw [Matrix.trace_smul, Matrix.trace_sub, trace_starMat (by norm_num) S, sub_zero,
    Matrix.trace_one, card_Blade4, smul_eq_mul]
  norm_num

/-! ## Step B — rank = trace for the ℚ-idempotents, giving rank 35 -/

/-- When the star-square scalar is `+1`, `P₋` is idempotent (companion to `Pplus_sq_of_even`). -/
theorem Pminus_sq_of_even (S : Finset (Fin (2 * m))) (h : starScalar S = 1) :
    Pminus S * Pminus S = Pminus S := by
  simp only [Pminus]
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, mul_sub, sub_mul, sub_mul]
  simp only [mul_one, one_mul]
  rw [starMat_sq, h]
  module

/-- **Rank = trace for a ℚ-idempotent.** For a square matrix `P` over `ℚ` with `P * P = P`,
the natural-number `Matrix.rank P` (cast to `ℚ`) equals the `ℚ`-trace `P.trace`.

The endomorphism `f = toLin' P` is a projection onto its range (idempotency gives
`LinearMap.IsProj (range f) f`); `LinearMap.IsProj.trace` then equates its `LinearMap.trace`
with `finrank ℚ (range f)`, which is `Matrix.rank P` by definition, while the `LinearMap`
trace equals the `Matrix` trace through the standard basis. -/
theorem rank_eq_trace_of_idempotent {n : Type*} [Fintype n] [DecidableEq n]
    (P : Matrix n n ℚ) (hP : P * P = P) :
    (P.rank : ℚ) = P.trace := by
  set f : (n → ℚ) →ₗ[ℚ] (n → ℚ) := Matrix.toLin' P with hf
  -- f is idempotent as a linear map
  have hidem : f ∘ₗ f = f := by
    rw [hf, ← Matrix.toLin'_mul, hP]
  -- f is a projection onto its range
  have hproj : LinearMap.IsProj (LinearMap.range f) f := by
    refine ⟨fun x => LinearMap.mem_range_self f x, ?_⟩
    intro x hx
    obtain ⟨y, hy⟩ := LinearMap.mem_range.1 hx
    rw [← hy, ← LinearMap.comp_apply, hidem]
  -- trace = finrank of the range
  have htr : LinearMap.trace ℚ (n → ℚ) f = (finrank ℚ (LinearMap.range f) : ℚ) :=
    hproj.trace
  -- Matrix.rank is finrank of the range of mulVecLin = toLin' = f
  have hrank : P.rank = finrank ℚ (LinearMap.range f) := by
    rw [hf, Matrix.toLin'_apply']
    rfl
  -- the linear-map trace equals the matrix trace through the standard basis
  have hmtr : LinearMap.trace ℚ (n → ℚ) f = P.trace := by
    rw [LinearMap.trace_eq_matrix_trace ℚ (Pi.basisFun ℚ n) f,
      LinearMap.toMatrix_eq_toMatrix', hf, LinearMap.toMatrix'_toLin']
  rw [hrank, ← htr, hmtr]

/-- `rank P₊ = 35` at `n = 8` whenever `q` is even (so `P₊` is idempotent). -/
theorem rank_Pplus_n8 {S : Finset (Fin (2 * 4))} (hq : Even (qneg S)) :
    (Pplus S).rank = 35 := by
  have hs : starScalar S = 1 := (carrier_condition S).mpr hq
  have hbridge : ((Pplus S).rank : ℚ) = (Pplus S).trace :=
    rank_eq_trace_of_idempotent (Pplus S) (Pplus_sq_of_even S hs)
  rw [trace_Pplus_n8] at hbridge
  exact_mod_cast hbridge

/-- `rank P₋ = 35` at `n = 8` whenever `q` is even (so `P₋` is idempotent). -/
theorem rank_Pminus_n8 {S : Finset (Fin (2 * 4))} (hq : Even (qneg S)) :
    (Pminus S).rank = 35 := by
  have hs : starScalar S = 1 := (carrier_condition S).mpr hq
  have hbridge : ((Pminus S).rank : ℚ) = (Pminus S).trace :=
    rank_eq_trace_of_idempotent (Pminus S) (Pminus_sq_of_even S hs)
  rw [trace_Pminus_n8] at hbridge
  exact_mod_cast hbridge

/-- The two chiral projectors have equal rank (both `35`) on the carrier. -/
theorem rank_Pplus_eq_rank_Pminus {S : Finset (Fin (2 * 4))} (hq : Even (qneg S)) :
    (Pplus S).rank = (Pminus S).rank := by
  rw [rank_Pplus_n8 hq, rank_Pminus_n8 hq]

/-- **Proposition 4.1 (concrete rank split).** `rank P₊ + rank P₋ = 70 = C(8,4)`:
the middle-grade carrier splits as `70 = 35 ⊕ 35`. -/
theorem rank_split_n8 {S : Finset (Fin (2 * 4))} (hq : Even (qneg S)) :
    (Pplus S).rank + (Pminus S).rank = 70 := by
  rw [rank_Pplus_n8 hq, rank_Pminus_n8 hq]

end ChiralParity
