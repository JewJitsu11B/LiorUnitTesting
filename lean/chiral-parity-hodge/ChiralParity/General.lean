import Mathlib

/-!
# General-grade Hodge star (foundation)

Generalizes `Carrier.lean` from the middle grade to an arbitrary grade `k` of an `n`-dimensional
carrier, with complement grade `j = n - k`. To avoid `n - (n - k)` type mismatches, we parametrize
by `k` and `j` with the hypothesis `k + j = n` supplied at the lemma level; the definitions
themselves need no subtraction.

* `GBlade n k` — `k`-element subsets of `Fin n`.
* `gstar n k j S : Matrix (GBlade n k) (GBlade n j) ℚ` — the star `⋆ : Λᵏ → Λʲ`
  (`⋆ e_I = ε_I (-1)^{inv I} e_{Iᶜ}`), row = input `k`-blade, nonzero entry at the complement.
* `gscalar n k j S = (-1)^{k·j + q}` — the star-square scalar (Lemma 3.1: `⋆∘⋆ = gscalar · id`).

The middle grade of `Carrier.lean` is the case `k = j = m`, `n = 2m`.
-/

namespace ChiralParity

open Matrix Finset

/-- General `k`-blade index: `k`-element subsets of `Fin n`. -/
abbrev GBlade (n k : ℕ) : Type := {s : Finset (Fin n) // s.card = k}

variable {n : ℕ}

/-- Signature sign `ε_I = (-1)^{|I ∩ S|}` on `Fin n`. -/
def geps (S I : Finset (Fin n)) : ℤ := (-1) ^ ((I ∩ S).card)

/-- Inversion count of the shuffle `(I, Iᶜ)` on `Fin n`. -/
def ginv (I : Finset (Fin n)) : ℕ := ((I ×ˢ Iᶜ).filter (fun p => p.2 < p.1)).card

/-- The star coefficient `ε_I · (-1)^{inv I}`. -/
def gstarSign (S I : Finset (Fin n)) : ℤ := geps S I * (-1) ^ (ginv I)

/-- The Hodge star `⋆ : Λᵏ → Λʲ` (`j = n - k`) as an explicit `ℚ`-matrix.
Row = input `k`-blade `I`; its single nonzero entry sits in column `Iᶜ` (a `j`-blade). -/
def gstar (n k j : ℕ) (S : Finset (Fin n)) : Matrix (GBlade n k) (GBlade n j) ℚ :=
  fun I J =>
    if (J : Finset (Fin n)) = (I : Finset (Fin n))ᶜ
    then ((gstarSign S (I : Finset (Fin n)) : ℤ) : ℚ) else 0

/-- The star-square scalar `(-1)^{k·j + q}` (`k·j = k(n-k)` when `k + j = n`). -/
def gscalar (n k j : ℕ) (S : Finset (Fin n)) : ℚ := (-1) ^ (k * j + S.card)

end ChiralParity
