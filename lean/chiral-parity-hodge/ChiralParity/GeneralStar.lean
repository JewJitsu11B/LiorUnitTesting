import ChiralParity.General

/-!
# General-grade Hodge star-square (Lemma 3.1, arbitrary grade)

Generalizes `StarSquare.lean` from the middle grade (`k = j = m`, `n = 2m`) to an arbitrary grade
`k` with complementary grade `j` (`k + j = n`). The main result `gstar_sq` states
`⋆∘⋆ = (-1)^{k·j + q} · id` on `Λᵏ`.
-/

namespace ChiralParity

open Matrix Finset

variable {n : ℕ}

/-- Helper 1: the signature signs over complementary index sets multiply to `(-1)^{|S|}`. -/
theorem geps_mul_geps_compl (S I : Finset (Fin n)) :
    geps S I * geps S Iᶜ = (-1) ^ (S.card) := by
  unfold geps
  rw [← pow_add]
  congr 1
  -- (I ∩ S).card + (Iᶜ ∩ S).card = S.card
  have hdisj : Disjoint (I ∩ S) (Iᶜ ∩ S) := by
    apply Finset.disjoint_left.mpr
    intro a ha hb
    rw [Finset.mem_inter] at ha hb
    exact (Finset.mem_compl.mp hb.1) ha.1
  have hunion : (I ∩ S) ∪ (Iᶜ ∩ S) = S := by
    rw [← Finset.union_inter_distrib_right, Finset.union_compl, Finset.univ_inter]
  rw [← Finset.card_union_of_disjoint hdisj, hunion]

/-- Helper 3: the complement of a `k`-blade in `Fin n` has card `j` when `k + j = n`. -/
theorem card_compl_eq {k j : ℕ} (hkj : k + j = n) {I : Finset (Fin n)} (hI : I.card = k) :
    Iᶜ.card = j := by
  rw [Finset.card_compl, Fintype.card_fin, hI]
  omega

/-- Helper 2: inversion counts over complementary index sets sum to `k·j`. -/
theorem ginv_add_ginv_compl {k j : ℕ} {I : Finset (Fin n)} (hI : I.card = k) (hIc : Iᶜ.card = j) :
    ginv I + ginv Iᶜ = k * j := by
  set P := I ×ˢ Iᶜ with hP
  -- (a) P.card = k * j
  have hPcard : P.card = k * j := by
    rw [hP, Finset.card_product, hI, hIc]
  -- (c) ginv Iᶜ = (P.filter (fun p => p.1 < p.2)).card
  have hinvIc : ginv Iᶜ = (P.filter (fun p => p.1 < p.2)).card := by
    unfold ginv
    rw [compl_compl]
    apply Finset.card_nbij' (fun p => Prod.swap p) (fun p => Prod.swap p)
    · intro a ha
      rw [Finset.mem_filter, Finset.mem_product] at ha ⊢
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · exact ha.1.2
      · exact ha.1.1
      · exact ha.2
    · intro a ha
      rw [Finset.mem_filter, Finset.mem_product] at ha ⊢
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · exact ha.1.2
      · exact ha.1.1
      · exact ha.2
    · intro a _
      exact Prod.swap_swap a
    · intro a _
      exact Prod.swap_swap a
  -- (b) ginv I = (P.filter (fun p => p.2 < p.1)).card  (by definition)
  have hinvI : ginv I = (P.filter (fun p => p.2 < p.1)).card := rfl
  -- (d) on P, ¬ (p.2 < p.1) ↔ p.1 < p.2
  have hsplit : (P.filter (fun p => p.2 < p.1)).card + (P.filter (fun p => p.1 < p.2)).card
      = P.card := by
    have hcongr : (P.filter (fun p => ¬ p.2 < p.1)) = (P.filter (fun p => p.1 < p.2)) := by
      apply Finset.filter_congr
      intro p hp
      rw [hP, Finset.mem_product] at hp
      have hne : p.1 ≠ p.2 := by
        intro h
        have h1 : p.1 ∈ I := hp.1
        have h2 : p.2 ∈ Iᶜ := hp.2
        rw [h] at h1
        exact (Finset.mem_compl.mp h2) h1
      constructor
      · intro h
        exact lt_of_le_of_ne (not_lt.mp h) hne
      · intro h
        exact not_lt.mpr (le_of_lt h)
    rw [← hcongr]
    exact Finset.filter_card_add_filter_neg_card_eq_card _
  rw [hinvI, hinvIc, hsplit, hPcard]

/-- Main (Lemma 3.1, general grade): `⋆∘⋆ = (-1)^{k·j + q} · id` on `Λᵏ`. -/
theorem gstar_sq {k j : ℕ} (hkj : k + j = n) (S : Finset (Fin n)) :
    gstar n k j S * gstar n j k S = gscalar n k j S • (1 : Matrix (GBlade n k) (GBlade n k) ℚ) := by
  ext I J
  rw [Matrix.mul_apply]
  -- The complement blade (a j-blade)
  have hIcard : (I : Finset (Fin n)).card = k := I.property
  have hIccard : ((I : Finset (Fin n))ᶜ).card = j := card_compl_eq hkj I.property
  set Ic : GBlade n j := ⟨(I : Finset (Fin n))ᶜ, hIccard⟩ with hIc
  -- Evaluate the sum via sum_eq_single at Ic
  rw [Finset.sum_eq_single Ic]
  · -- The Ic term
    -- gstar n k j S I Ic = gstarSign S I  (since (Ic:Finset) = Iᶜ)
    have h1 : gstar n k j S I Ic = ((gstarSign S (I : Finset (Fin n)) : ℤ) : ℚ) := by
      unfold gstar
      rw [if_pos rfl]
    -- gstar n j k S Ic J = if (J:Finset) = I then gstarSign S Iᶜ else 0
    have h2 : gstar n j k S Ic J
        = if (J : Finset (Fin n)) = (I : Finset (Fin n))
          then ((gstarSign S ((I : Finset (Fin n))ᶜ) : ℤ) : ℚ) else 0 := by
      unfold gstar
      simp only [hIc, compl_compl]
    rw [h1, h2]
    -- RHS
    rw [Matrix.smul_apply, Matrix.one_apply, smul_eq_mul]
    by_cases hJI : (J : Finset (Fin n)) = (I : Finset (Fin n))
    · rw [if_pos hJI]
      have hJIblade : I = J := by
        apply Subtype.ext
        exact hJI.symm
      rw [if_pos hJIblade, mul_one]
      -- scalar identity at ℤ level, then cast
      have key : (gstarSign S (I : Finset (Fin n)))
            * (gstarSign S (I : Finset (Fin n))ᶜ)
          = (-1 : ℤ) ^ (k * j + S.card) := by
        unfold gstarSign
        rw [mul_mul_mul_comm, ← pow_add, geps_mul_geps_compl S (I : Finset (Fin n)),
          ginv_add_ginv_compl I.property (card_compl_eq hkj I.property), ← pow_add]
        congr 1
        omega
      have hcast : ((gstarSign S (I : Finset (Fin n)) : ℤ) : ℚ)
            * ((gstarSign S (I : Finset (Fin n))ᶜ : ℤ) : ℚ)
          = (((-1 : ℤ) ^ (k * j + S.card) : ℤ) : ℚ) := by
        rw [← Int.cast_mul, key]
      rw [hcast]
      unfold gscalar
      push_cast
      ring
    · rw [if_neg hJI]
      have hIJblade : ¬ (I = J) := by
        intro h
        exact hJI (congrArg (fun (b : GBlade n k) => (b : Finset (Fin n))) h).symm
      rw [if_neg hIJblade, mul_zero, mul_zero]
  · -- terms K ≠ Ic are zero
    intro K _ hKIc
    have hKne : (K : Finset (Fin n)) ≠ (I : Finset (Fin n))ᶜ := by
      intro h
      apply hKIc
      apply Subtype.ext
      rw [hIc]
      exact h
    have : gstar n k j S I K = 0 := by
      unfold gstar
      rw [if_neg (fun h => hKne h)]
    rw [this, zero_mul]
  · -- Ic ∈ univ trivially; sum_eq_single side goal
    intro h
    exact absurd (Finset.mem_univ Ic) h

end ChiralParity
