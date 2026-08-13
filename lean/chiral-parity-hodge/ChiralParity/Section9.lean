import ChiralParity.Forcing
namespace ChiralParity

/-!
# Section 9: closing the "hollow bottom" of the signature-forcing result

The `(4,4)` forcing of Theorem 9.10 rested on two invariants that `Forcing.lean` took as bare
arithmetic predicates (`weyl`, `majorana`). This file grounds each in its actual origin:

* **Weyl side (proved, no axiom).** The Weyl condition is the statement that the Clifford volume
  element `ω` squares to `+1`. With `n = p + q`, `ω² = (-1)^{n(n-1)/2 + q}`. We define this sign as
  `wsign p q` and prove **Corollary 9.7** *computably*: `wsign p q = 1 ↔ (p - q) ≡ 0 (mod 4)`
  (`weyl_iff_wsign`), under the even-dimension hypothesis in which Weyl spinors exist. The
  equivalence is genuinely false for odd `n` (e.g. `(p,q) = (1,0)`: `wsign = 1` yet `p - q = 1`),
  which is exactly why `Forcing.mw_iff_mod8` already carries a parity hypothesis; we carry the same
  one honestly.

* **Majorana / representation-type side (explicit cited axiom).** Whether the spinor module admits
  a real structure is the Atiyah-Bott-Shapiro classification of real Clifford algebras. Mathlib
  v4.14.0 does not have it and the paper cites it; we represent it as one clearly-labeled
  `CITED-AXIOM` (`majorana_iff_realstructure`) over an opaque predicate `RealSpinorStructure`.

`forcing_n8_from_clifford` then re-derives `(4,4)` from the *computable* Weyl sign plus the
*cited* real-structure predicate, so the previously-silent gap under Section 9 is now either proved
or explicitly attributed.
-/

/-- The square-sign of the Clifford volume element in signature `(p, q)`:
`ω² = (-1)^{n(n-1)/2 + q}` with `n = p + q`. The Weyl (chirality) condition is `wsign p q = +1`. -/
def wsign (p q : ℕ) : ℤ := (-1) ^ ((p + q) * (p + q - 1) / 2 + q)

/-- Arithmetic core of the Weyl reduction. For even `n = p + q`, the triangular-plus-`q` exponent
is even iff `n ≡ 2q (mod 4)`, i.e. iff `p - q ≡ 0 (mod 4)`. Pure `ℕ`/`ℤ` arithmetic, no axiom. -/
theorem wsign_exponent_even_iff (n q : ℕ) (hn : n % 2 = 0) :
    Even (n * (n - 1) / 2 + q) ↔ ((n : ℤ) - 2 * q) % 4 = 0 := by
  rw [Nat.even_iff]
  set x := n * (n - 1) with hxdef
  -- `x = n(n-1)` is even, so `x/2` is a genuine integer and `2·(x/2) = x`.
  have hev : Even x := by
    rw [hxdef]
    rcases n with _ | m
    · simp
    · rw [Nat.succ_sub_one, Nat.mul_comm]; exact Nat.even_mul_succ_self m
  have hx2 : x % 2 = 0 := Nat.even_iff.mp hev
  -- `x mod 4` in closed form: `(n mod 4)/2` doubled. (0,0,2,2 as n mod 4 runs 0,1,2,3.)
  have hx4 : x % 4 = 2 * ((n % 4) / 2) := by
    rw [hxdef, Nat.mul_mod]
    rcases n with _ | m
    · rfl
    · rw [Nat.succ_sub_one]
      have h1 : (m + 1) % 4 = (m % 4 + 1) % 4 := by omega
      rw [h1]
      set r := m % 4 with hr
      have hlt : r < 4 := by omega
      interval_cases r <;> decide
  clear_value x
  -- Bridge: for even `x`, the parity of `x/2` is `(x mod 4)/2`.
  have hbridge : (x / 2) % 2 = (x % 4) / 2 := by omega
  have hTval : (x / 2) % 2 = (n % 4) / 2 := by omega
  omega

/-- **Corollary 9.7 (Weyl reduction), computable form.** The Clifford volume element squares to
`+1` iff `p - q ≡ 0 (mod 4)`, under the even-dimension hypothesis `p + q` even in which Weyl
spinors exist. Fully proved, no axiom. (For odd `p + q` the equivalence is false, e.g. `(1,0)`.) -/
theorem weyl_iff_wsign {p q : ℕ} (hpar : (p + q) % 2 = 0) :
    wsign p q = 1 ↔ weyl (p : ℤ) (q : ℤ) := by
  unfold wsign weyl
  rw [neg_one_pow_eq_one_iff_even (by decide : (-1 : ℤ) ≠ 1),
    wsign_exponent_even_iff (p + q) q hpar]
  push_cast
  omega

/-! ## Majorana / representation-type side: the Atiyah-Bott-Shapiro classification -/

/-- Opaque predicate: the spinor module of `Cl(p,q)` admits a real (Majorana) structure. Its
content is supplied only through the cited classification axiom below. -/
opaque RealSpinorStructure : ℤ → ℤ → Prop

/-- Atiyah-Bott-Shapiro classification of real Clifford algebras Cl(p,q) as matrix algebras
    over ℝ/ℂ/ℍ; the spinor module admits a real structure (Majorana) iff p - q ≡ 0,1,2 (mod 8).
    Not in Mathlib v4.14.0; cited from Atiyah-Bott-Shapiro (1964), Lawson-Michelsohn Spin Geometry.
    Paper: Lemma 9.4 / Prop 9.5. -/
axiom majorana_iff_realstructure (p q : ℤ) :  -- CITED-AXIOM
    RealSpinorStructure p q ↔
      ((p - q) % 8 = 0 ∨ (p - q) % 8 = 1 ∨ (p - q) % 8 = 2)

/-- The paper's `majorana` congruence predicate coincides with admitting a real spinor structure,
by the cited ABS classification. (Derived from the single axiom; not itself an axiom.) -/
theorem majorana_iff_real (p q : ℤ) : majorana p q ↔ RealSpinorStructure p q := by
  rw [majorana_iff_realstructure]; rfl

/-! ## The `(4,4)` forcing, grounded in the volume-element sign and the cited classification -/

/-- **Theorem 9.10, grounded form.** In `n = 8` with a nonempty light cone, the *computable* Weyl
sign `wsign p q = 1` together with the ABS-cited real-structure predicate force `(p, q) = (4, 4)`.
The Weyl input is `weyl_iff_wsign` (proved); the Majorana input is `majorana_iff_realstructure`
(cited). This closes the previously-silent Section 9 gap. -/
theorem forcing_n8_from_clifford (p q : ℕ) (hpq : p + q = 8) (hp : 0 < p) (hq : 0 < q)
    (hW : wsign p q = 1) (hM : RealSpinorStructure (p : ℤ) (q : ℤ)) : p = 4 ∧ q = 4 := by
  have hpar : (p + q) % 2 = 0 := by omega
  have hw : weyl (p : ℤ) (q : ℤ) := (weyl_iff_wsign hpar).mp hW
  have hm : majorana (p : ℤ) (q : ℤ) := (majorana_iff_real _ _).mpr hM
  exact forcing_n8_full p q hpq hp hq hw hm

/-! ## Anchor checks -/

/-- The split signature `(4,4)` has `wsign = +1` (volume element squares to `+1`). -/
example : wsign 4 4 = 1 := by decide

/-- The quaternionic candidate `(6,2)` also has `wsign = +1` (Weyl holds) but is excluded by the
Majorana side, consistent with `Forcing`'s anchor checks. -/
example : wsign 6 2 = 1 := by decide

end ChiralParity
