import ChiralParity.Carrier

/-!
# Proposition 4.1 (trace / equal-split content)

The middle-grade Hodge star `⋆` has zero diagonal, hence zero trace; the chiral projectors
`P₊ = ½(1 + ⋆)` and `P₋ = ½(1 − ⋆)` sum to the identity, differ by `⋆`, and — because the
star is traceless — carry equal trace (the "equal split" of the carrier).
-/

namespace ChiralParity

open Matrix Finset

/-- The middle-grade star has zero diagonal: `⋆ e_I` has no `e_I` component, because `I ≠ Iᶜ`
when `I` is nonempty (`card I = m > 0`). -/
theorem starMat_diag_zero {m} (hm : 0 < m) (S : Finset (Fin (2 * m))) (I : Blade m) :
    starMat S I I = 0 := by
  have hne : (I : Finset (Fin (2 * m))) ≠ (I : Finset (Fin (2 * m)))ᶜ := by
    intro h
    have hI : (I : Finset (Fin (2 * m))).Nonempty := by
      rw [← Finset.card_pos, I.property]; exact hm
    obtain ⟨x, hx⟩ := hI
    have hxc : x ∈ (I : Finset (Fin (2 * m)))ᶜ := h ▸ hx
    rw [Finset.mem_compl] at hxc
    exact hxc hx
  unfold starMat
  rw [if_neg hne]

/-- The middle-grade star is traceless. -/
theorem trace_starMat {m} (hm : 0 < m) (S : Finset (Fin (2 * m))) :
    (starMat S).trace = 0 := by
  rw [Matrix.trace]
  simp only [Matrix.diag]
  exact Finset.sum_eq_zero (fun I _ => starMat_diag_zero hm S I)

/-- `P₊ + P₋ = 1`. -/
theorem Pplus_add_Pminus {m} (S : Finset (Fin (2 * m))) :
    Pplus S + Pminus S = 1 := by
  unfold Pplus Pminus
  module

/-- `P₊ − P₋ = ⋆`. -/
theorem Pplus_sub_Pminus {m} (S : Finset (Fin (2 * m))) :
    Pplus S - Pminus S = starMat S := by
  unfold Pplus Pminus
  module

/-- The equal split: the two chiral projectors carry equal trace, since `⋆` is traceless. -/
theorem trace_Pplus_eq_trace_Pminus {m} (hm : 0 < m) (S : Finset (Fin (2 * m))) :
    (Pplus S).trace = (Pminus S).trace := by
  unfold Pplus Pminus
  simp only [Matrix.trace_smul, Matrix.trace_add, Matrix.trace_sub, trace_starMat hm S,
    add_zero, sub_zero]

end ChiralParity
