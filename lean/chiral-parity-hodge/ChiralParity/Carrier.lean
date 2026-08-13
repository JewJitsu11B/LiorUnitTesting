import Mathlib

/-!
# Middle-grade Hodge star on the biquaternionic carrier — foundation

Faithful Lean model of "The Parity Criterion for Native Chiral Projectors:
Middle-Grade Hodge Duality on the Biquaternionic Carrier" (Leizerman, 2026).

We formalize the MIDDLE grade `k = m` of an `n = 2m`-dimensional carrier, where the Hodge
star is an endomorphism of `Λᵐ` (Iᶜ has card `2m - m = m`). Following the paper's harness, the
star is built directly from the explicit sign formula, with NO library Hodge star.

Representation (matches the paper's blade matrix):
* blades = `m`-element subsets of `Fin (2m)`;
* signature = a set `S` of negative directions, `q = |S|`, `ε_I = (-1)^{|I ∩ S|}`;
* the shuffle sign `sgn(I, Iᶜ)` is realized as an INVERSION COUNT
  `inv I = #{(a,b) ∈ I × Iᶜ : a > b}` (its parity equals the permutation sign);
* `⋆ e_I = ε_I · (-1)^{inv I} · e_{Iᶜ}`.

The load-bearing identity (Lemma 3.1) `⋆² = (-1)^{m²+q}·id` then reduces to two clean Finset
facts: `inv I + inv Iᶜ = m²` and `ε_I · ε_{Iᶜ} = (-1)^q`.
-/

namespace ChiralParity

open Matrix Finset

/-- The middle-grade blade index: `m`-element subsets of `Fin (2*m)`. -/
abbrev Blade (m : ℕ) : Type := {s : Finset (Fin (2 * m)) // s.card = m}

variable {m : ℕ}

/-- Signature sign `ε_I = ∏_{i∈I} sgn i`, with `S` the negative directions:
`ε_I = (-1)^{|I ∩ S|}`. -/
def eps (S I : Finset (Fin (2 * m))) : ℤ := (-1) ^ ((I ∩ S).card)

/-- Inversion count of the shuffle `(I, Iᶜ)`: pairs `(a ∈ I, b ∈ Iᶜ)` with `a > b`.
Its parity equals the permutation sign `sgn(I, Iᶜ)`. -/
def inv (I : Finset (Fin (2 * m))) : ℕ :=
  ((I ×ˢ Iᶜ).filter (fun p => p.2 < p.1)).card

/-- The star coefficient on a blade: `ε_I · (-1)^{inv I}`. -/
def starSign (S I : Finset (Fin (2 * m))) : ℤ := eps S I * (-1) ^ (inv I)

/-- The middle-grade Hodge star as an explicit `ℚ`-matrix on blades:
`⋆ e_I = starSign_I · e_{Iᶜ}`. -/
def starMat (S : Finset (Fin (2 * m))) : Matrix (Blade m) (Blade m) ℚ :=
  fun I J =>
    if (J : Finset (Fin (2 * m))) = (I : Finset (Fin (2 * m)))ᶜ
    then ((starSign S (I : Finset (Fin (2 * m))) : ℤ) : ℚ) else 0

/-- Number of negative directions `q = |S|`. -/
def qneg (S : Finset (Fin (2 * m))) : ℕ := S.card

/-- The star-square scalar `s = (-1)^{m² + q}` (Lemma 3.1 at the middle grade). -/
def starScalar (S : Finset (Fin (2 * m))) : ℚ := (-1) ^ (m ^ 2 + qneg S)

/-- Chiral projector `P₊ = ½(id + ⋆)` (Definition 2.3). -/
noncomputable def Pplus (S : Finset (Fin (2 * m))) : Matrix (Blade m) (Blade m) ℚ :=
  (2⁻¹ : ℚ) • ((1 : Matrix (Blade m) (Blade m) ℚ) + starMat S)

/-- Chiral projector `P₋ = ½(id − ⋆)`. -/
noncomputable def Pminus (S : Finset (Fin (2 * m))) : Matrix (Blade m) (Blade m) ℚ :=
  (2⁻¹ : ℚ) • ((1 : Matrix (Blade m) (Blade m) ℚ) - starMat S)

end ChiralParity
