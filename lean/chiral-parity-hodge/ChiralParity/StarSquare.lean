import ChiralParity.Carrier

namespace ChiralParity

open Matrix Finset

variable {m : ℕ}

/-- Helper 1: the signature signs over complementary index sets multiply to `(-1)^{q}`. -/
theorem eps_mul_eps_compl (S I : Finset (Fin (2 * m))) :
    eps S I * eps S Iᶜ = (-1) ^ (qneg S) := by
  unfold eps qneg
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

/-- Helper 2: inversion counts over complementary index sets sum to `m²`. -/
theorem inv_add_inv_compl {I : Finset (Fin (2 * m))} (hI : I.card = m) :
    inv I + inv Iᶜ = m * m := by
  -- Iᶜ.card = m
  have hIc : Iᶜ.card = m := by
    rw [Finset.card_compl, Fintype.card_fin, hI]
    omega
  set P := I ×ˢ Iᶜ with hP
  -- (a) P.card = m * m
  have hPcard : P.card = m * m := by
    rw [hP, Finset.card_product, hI, hIc]
  -- (c) inv Iᶜ = (P.filter (fun p => p.1 < p.2)).card
  have hinvIc : inv Iᶜ = (P.filter (fun p => p.1 < p.2)).card := by
    unfold inv
    rw [compl_compl]
    -- ((Iᶜ ×ˢ I).filter (fun p => p.2 < p.1)).card = ((I ×ˢ Iᶜ).filter (fun p => p.1 < p.2)).card
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
  -- (b) inv I = (P.filter (fun p => p.2 < p.1)).card  (by definition)
  have hinvI : inv I = (P.filter (fun p => p.2 < p.1)).card := rfl
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

/-- Main (Lemma 3.1): the Hodge star squares to `(-1)^{m²+q}·id` at the middle grade. -/
theorem starMat_sq (S : Finset (Fin (2 * m))) :
    starMat S * starMat S = starScalar S • (1 : Matrix (Blade m) (Blade m) ℚ) := by
  ext I J
  rw [Matrix.mul_apply]
  -- The complement blade
  have hIcard : (I : Finset (Fin (2 * m))).card = m := I.property
  have hIccard : ((I : Finset (Fin (2 * m)))ᶜ).card = m := by
    rw [Finset.card_compl, Fintype.card_fin, hIcard]
    omega
  set Ic : Blade m := ⟨(I : Finset (Fin (2 * m)))ᶜ, hIccard⟩ with hIc
  -- Evaluate the sum via sum_eq_single at Ic
  rw [Finset.sum_eq_single Ic]
  · -- The Ic term
    -- starMat S I Ic = starSign S I  (since (Ic:Finset) = Iᶜ)
    have h1 : starMat S I Ic = ((starSign S (I : Finset (Fin (2 * m))) : ℤ) : ℚ) := by
      unfold starMat
      rw [if_pos rfl]
    -- starMat S Ic J = if (J:Finset) = I then starSign S Iᶜ else 0
    have h2 : starMat S Ic J
        = if (J : Finset (Fin (2 * m))) = (I : Finset (Fin (2 * m)))
          then ((starSign S ((I : Finset (Fin (2 * m)))ᶜ) : ℤ) : ℚ) else 0 := by
      unfold starMat
      simp only [hIc, compl_compl]
    rw [h1, h2]
    -- RHS
    rw [Matrix.smul_apply, Matrix.one_apply, smul_eq_mul]
    by_cases hJI : (J : Finset (Fin (2 * m))) = (I : Finset (Fin (2 * m)))
    · rw [if_pos hJI]
      have hJIblade : I = J := by
        apply Subtype.ext
        exact hJI.symm
      rw [if_pos hJIblade, mul_one]
      -- scalar identity at ℤ level, then cast
      have key : (starSign S (I : Finset (Fin (2 * m))))
            * (starSign S (I : Finset (Fin (2 * m)))ᶜ)
          = (-1 : ℤ) ^ (m ^ 2 + qneg S) := by
        unfold starSign
        rw [mul_mul_mul_comm, ← pow_add, eps_mul_eps_compl S (I : Finset (Fin (2 * m))),
          inv_add_inv_compl I.property, ← pow_add]
        congr 1
        rw [pow_two]
        omega
      have hcast : ((starSign S (I : Finset (Fin (2 * m))) : ℤ) : ℚ)
            * ((starSign S (I : Finset (Fin (2 * m)))ᶜ : ℤ) : ℚ)
          = (((-1 : ℤ) ^ (m ^ 2 + qneg S) : ℤ) : ℚ) := by
        rw [← Int.cast_mul, key]
      rw [hcast]
      unfold starScalar
      push_cast
      ring
    · rw [if_neg hJI]
      have hIJblade : ¬ (I = J) := by
        intro h
        exact hJI (congrArg (fun (b : Blade m) => (b : Finset (Fin (2 * m)))) h).symm
      rw [if_neg hIJblade, mul_zero, mul_zero]
  · -- terms K ≠ Ic are zero
    intro K _ hKIc
    have hKne : (K : Finset (Fin (2 * m))) ≠ (I : Finset (Fin (2 * m)))ᶜ := by
      intro h
      apply hKIc
      apply Subtype.ext
      rw [hIc]
      exact h
    have : starMat S I K = 0 := by
      unfold starMat
      rw [if_neg (fun h => hKne h)]
    rw [this, zero_mul]
  · -- Ic ∈ univ trivially; sum_eq_single side goal
    intro h
    exact absurd (Finset.mem_univ Ic) h

end ChiralParity
