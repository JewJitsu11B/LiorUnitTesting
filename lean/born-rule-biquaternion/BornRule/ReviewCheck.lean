import BornRule.T61_Born
import BornRule.L43_NullIdempotents
import BornRule.P52_Closure

namespace BornRule
open Matrix

/-! # Adversarial ReviewCheck: non-vacuity + oracle agreement for the Born readout. -/

/-! ## Item 1 — witnesses satisfy `IsUnit3` (hypotheses are non-vacuous). -/

example : IsUnit3 ![0, 0, 1] := by norm_num [IsUnit3]

example : IsUnit3 ![3/5, 0, 4/5] := by norm_num [IsUnit3]

-- extra witnesses used below
example : IsUnit3 ![1, 0, 0] := by norm_num [IsUnit3]
example : IsUnit3 ![0, 0, -1] := by norm_num [IsUnit3]

/-! ## Item 2 — Born values match the oracle `(1 + m·n)/2` at concrete vectors. -/

-- parallel, dot = 1  →  1
example : traceRead (Pbranch 1 ![0, 0, 1] * Pbranch 1 ![0, 0, 1]) = 1 := by
  rw [born_plus_plus]; norm_num

-- orthogonal, dot = 0  →  1/2
example : traceRead (Pbranch 1 ![1, 0, 0] * Pbranch 1 ![0, 0, 1]) = 1/2 := by
  rw [born_plus_plus]; norm_num

-- antipodal, dot = -1  →  0
example : traceRead (Pbranch 1 ![0, 0, 1] * Pbranch 1 ![0, 0, -1]) = 0 := by
  rw [born_plus_plus]; norm_num

-- dot = 4/5  →  9/10
example : traceRead (Pbranch 1 ![3/5, 0, 4/5] * Pbranch 1 ![0, 0, 1]) = 9/10 := by
  rw [born_plus_plus]; norm_num

/-! ## Item 3 — normalization at a concrete vector. -/

example :
    traceRead (Pbranch 1 ![1, 0, 0] * Pbranch 1 ![0, 0, 1])
      + traceRead (Pbranch (-1) ![1, 0, 0] * Pbranch 1 ![0, 0, 1]) = 1 :=
  born_normalization ![1, 0, 0] ![0, 0, 1]

/-! ## Item 4 — idempotency is real at a witness (hypotheses actually discharge). -/

example : Pbranch 1 ![0, 0, 1] * Pbranch 1 ![0, 0, 1] = Pbranch 1 ![0, 0, 1] :=
  Pbranch_idem (by norm_num [IsUnit3]) (by norm_num)

/-! ## Item 5 — probability lower bound, CONDITIONAL on the dot lower bound.

We do NOT prove `-1 ≤ m·n` here (that is Cauchy–Schwarz / unit-sphere machinery,
out of scope). We only show the readout is `≥ 0` *given* that bound, i.e. the
formula is a genuine probability lower bound once `m·n ≥ -1` is supplied. -/

example (m n : Fin 3 → ℝ)
    (h : (-1 : ℝ) ≤ m 0 * n 0 + m 1 * n 1 + m 2 * n 2) :
    0 ≤ traceRead (Pbranch 1 m * Pbranch 1 n) := by
  rw [born_plus_plus]; linarith

/-! ## Adversarial probe — `born_plus_plus` has NO unit hypothesis on `m, n`.

At the NON-unit vector `![1,1,1]` (‖·‖² = 3, dot = 3) the readout evaluates to
`(1 + 3)/2 = 2 > 1`, which is NOT a valid probability. The lemma still "works"
as a pure trace identity, demonstrating the interpretation gap. -/

example : traceRead (Pbranch 1 ![1, 1, 1] * Pbranch 1 ![1, 1, 1]) = 2 := by
  rw [born_plus_plus]; norm_num

end BornRule
