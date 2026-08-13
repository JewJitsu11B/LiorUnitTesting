import Mathlib

/-!
# The biquaternionic carrier `C ⊗ H = M₂(ℂ)`

Faithful Lean transcription of the definitions of
"A Biquaternionic Source-Closure Derivation of the Born Rule ..." (Leizerman, Aug 2026).

We work concretely in `Matrix (Fin 2) (Fin 2) ℂ`, exactly the paper's identification
`C ⊗ H = M₂(ℂ)` (Definition 2.3). Under this representation:

* the central imaginary `i` of the `C` factor is the scalar `Complex.I` (it commutes with all);
* the dagger conjugation `q†` is the conjugate-transpose (Definition 2.4(ii));
* the quaternionic bar `q̄` is the adjugate, the unique map with `q * q̄ = N(q) • 1`
  (Definition 2.4(i)); on `M₂` this is `Matrix.adjugate`;
* the reduced norm `N(q) = q q̄` is the determinant (Lemma 4.1(a)).

Every definition here is a transcription target for the adversarial faithfulness check:
each `def` must equal the paper's formula, and no physics is assumed.
-/

namespace BornRule

open Matrix

/-- The biquaternion carrier `C ⊗ H`, represented as `M₂(ℂ)` (Definition 2.3). -/
abbrev Carrier : Type := Matrix (Fin 2) (Fin 2) ℂ

/-- Quaternion unit `i` under `C ⊗ H = M₂(ℂ)`. -/
def qi : Carrier := !![Complex.I, 0; 0, -Complex.I]

/-- Quaternion unit `j`. -/
def qj : Carrier := !![0, 1; -1, 0]

/-- Quaternion unit `k`. -/
def qk : Carrier := !![0, Complex.I; Complex.I, 0]

/-- Dagger conjugation `q†`: the conjugate transpose (Definition 2.4(ii)). -/
def dagger (M : Carrier) : Carrier := Mᴴ

/-- Quaternionic bar `q̄`: the adjugate, satisfying `q * q̄ = N(q) • 1` (Definition 2.4(i)). -/
def bar (M : Carrier) : Carrier := adjugate M

/-- Reduced norm `N(q) = q q̄`, equal to the determinant (Lemma 4.1(a)). -/
def redNorm (M : Carrier) : ℂ := M.det

/-- `e_n = n₁ i + n₂ j + n₃ k` for a real 3-vector `n` (Definition 2.7). -/
def en (n : Fin 3 → ℝ) : Carrier :=
  (n 0 : ℂ) • qi + (n 1 : ℂ) • qj + (n 2 : ℂ) • qk

/-- The Hermitian split unit `E_n = i · e_n`, with `E_n² = +1` (Definition 2.7). -/
def En (n : Fin 3 → ℝ) : Carrier := Complex.I • en n

/-- Closed null source current `J_s(n) = 1 + s·E_n`; the physical branches are `s = ±1`
(Definition 2.7, Proposition 5.2). -/
def Jbranch (s : ℂ) (n : Fin 3 → ℝ) : Carrier := (1 : Carrier) + s • En n

/-- Projector `P_s(n) = J_s(n) / 2` (Definition 2.7). -/
noncomputable def Pbranch (s : ℂ) (n : Fin 3 → ℝ) : Carrier := (2⁻¹ : ℂ) • Jbranch s n

/-- Internal trace readout `B(q) = 2 Re Sc(q) = Re(tr q)`, the probability functional
(Definition 2.7 / eq. (7)). Here `Sc(q) = (tr q)/2` is the grade-0 part, so `2 Sc = tr`. -/
noncomputable def traceRead (M : Carrier) : ℝ := (M.trace).re

/-- Unit-vector predicate `‖n‖² = 1` used as a hypothesis throughout. -/
def IsUnit3 (n : Fin 3 → ℝ) : Prop := (n 0)^2 + (n 1)^2 + (n 2)^2 = 1

end BornRule
