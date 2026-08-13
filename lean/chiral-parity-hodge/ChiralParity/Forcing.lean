import Mathlib
namespace ChiralParity

/-!
# Mod-8 signature forcing (Section 9 of the chiral-projector paper)

A signature is a pair `(p, q)` with `n = p + q`. Two Clifford invariants:

* **Weyl**: `w = +1 ⟺ p - q ≡ 0 (mod 4)`.
* **Majorana** (real spinor type): `⟺ p - q ≡ 0, 1, 2 (mod 8)`.

This file formalizes:

* `mw_iff_mod8` — Corollary 9.7 (even `n` case): Majorana ∧ Weyl ⟺ `p - q ≡ 0 (mod 8)`.
* `forcing_n8` / `forcing_n8_full` — Theorem 9.10: in `n = 8`, Weyl + Majorana + a nonempty
  light cone force `(p, q) = (4, 4)`.
* Anchor checks from Remark 9.8.
-/

/-- Weyl condition: `p - q ≡ 0 (mod 4)`. -/
def weyl (p q : ℤ) : Prop := (p - q) % 4 = 0

/-- Majorana (real spinor type) condition: `p - q ≡ 0, 1, 2 (mod 8)`. -/
def majorana (p q : ℤ) : Prop :=
  (p - q) % 8 = 0 ∨ (p - q) % 8 = 1 ∨ (p - q) % 8 = 2

/-- **Corollary 9.7** (even `n`). With `p - q` even, the conjunction of the Weyl condition
(`p - q ≡ 0 mod 4`) and the Majorana condition (`p - q ≡ 0, 1, 2 mod 8`) is equivalent to the
single Majorana–Weyl condition `p - q ≡ 0 (mod 8)`. -/
theorem mw_iff_mod8 (p q : ℤ) (hpar : (p - q) % 2 = 0) :
    (weyl p q ∧ majorana p q) ↔ (p - q) % 8 = 0 := by
  unfold weyl majorana
  omega

/-- **Theorem 9.10** (arithmetic core). With `p + q = 8`, a nonempty light cone `0 < p`, `0 < q`,
and the Majorana–Weyl condition `p - q ≡ 0 (mod 8)`, the unique solution is `(p, q) = (4, 4)`. -/
theorem forcing_n8 (p q : ℕ) (hpq : p + q = 8) (hp : 0 < p) (hq : 0 < q)
    (hmw : ((p : ℤ) - q) % 8 = 0) : p = 4 ∧ q = 4 := by
  omega

/-- **Theorem 9.10** (full form). Same conclusion from the separate Weyl and Majorana hypotheses:
`p + q = 8` makes `p - q` even, so `mw_iff_mod8` collapses Weyl ∧ Majorana to `p - q ≡ 0 (mod 8)`,
and `forcing_n8` finishes. -/
theorem forcing_n8_full (p q : ℕ) (hpq : p + q = 8) (hp : 0 < p) (hq : 0 < q)
    (hw : weyl (p : ℤ) (q : ℤ)) (hm : majorana (p : ℤ) (q : ℤ)) : p = 4 ∧ q = 4 := by
  have hpar : ((p : ℤ) - q) % 2 = 0 := by omega
  have hmw : ((p : ℤ) - q) % 8 = 0 := (mw_iff_mod8 (p : ℤ) (q : ℤ) hpar).mp ⟨hw, hm⟩
  exact forcing_n8 p q hpq hp hq hmw

/-! ## Anchor checks (Remark 9.8) -/

/-- The split signature `(4, 4)` satisfies both Weyl and Majorana. -/
example : weyl 4 4 ∧ majorana 4 4 := by unfold weyl majorana; decide

/-- The split signature `(4, 4)` has a nonempty light cone. -/
example : (0 : ℤ) < 4 ∧ (0 : ℤ) < 4 := by decide

/-- The quaternionic candidate `(6, 2)`: Weyl holds (`p - q = 4 ≡ 0 mod 4`) but Majorana fails
(`4 ∉ {0, 1, 2} mod 8`), so it is excluded. -/
example : weyl 6 2 ∧ ¬ majorana 6 2 := by unfold weyl majorana; decide

/-- The quaternionic candidate `(2, 6)`: Weyl holds but Majorana fails (`p - q = -4 ≡ 4 mod 8`). -/
example : weyl 2 6 ∧ ¬ majorana 2 6 := by unfold weyl majorana; decide

/-- The definite signatures `(8, 0)` and `(0, 8)` pass Weyl and Majorana but fail the light-cone
condition (one of `p`, `q` is zero). -/
example : ¬ ((0 : ℤ) < 8 ∧ (0 : ℤ) < 0) := by decide

/-- `(8, 0)` and `(0, 8)` satisfy Majorana–Weyl (`±8 ≡ 0 mod 8`), confirming that only the light
cone excludes them. -/
example : ((8 : ℤ) - 0) % 8 = 0 ∧ ((0 : ℤ) - 8) % 8 = 0 := by decide

end ChiralParity
