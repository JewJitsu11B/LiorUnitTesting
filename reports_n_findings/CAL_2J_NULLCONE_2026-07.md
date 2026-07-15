# The 2J closure, the null cone, and what alpha=2 does not do

Audit date: 2026-07-15. Companion scripts: `py tests/exp_integrate_twice_period4.py`,
`exp_cayley_hamilton_2J.py`, `exp_null_cone_needs_channels.py`,
`exp_branch_cut_integer_only.py`, `exp_wedge_vs_geoprod.py`, `exp_z4_double_cover.py`.

Every script is standalone, seeded, ASCII-only, cross-checked against `cal.biquaternion`,
and carries a handedness guard. Each was written by one agent and independently
adversarially verified by another; the verifier's job was to refute, and in five of six
cases it found real defects and fixed them. Those fixes are recorded below.

Cite by LaTeX label, not line number: the manuscripts are under active edit and line
numbers move. Labels used here: `def:state`, `def:alpha`, `def:conj`, `ax:cal`,
`ax:current`, `ax:angle`, `lem:unique`, `lem:twoclosures`, `prop:antiparallel`,
`thm:born`.

---

## 1. The result that survives: the 2J closure derives, end to end

Chain, with nothing imported from outside the framework:

| step | statement | source |
|---|---|---|
| 1 | closure: both amplitude vectors null, sharing one quaternionic direction | `def:J-closed`, "the two coincide" |
| 2 | therefore `det J = 0` identically, for all values of both exp maps | verified, max abs det 2.0e-15 |
| 3 | normalisation `<a^dag a>_0 = 1` gives `tr J = 2` | `tr(a a^dag) = 2 <a^dag a>_0`, exact |
| 4 | Cayley-Hamilton (degree 2 because `[H:C] = 2`) gives `J^2 = 2J` | residual ~1e-16 |
| 5 | `J = a a^dag` is Hermitian because the dagger is native | `def:conj` |
| 6 | therefore `<J^dag J>_0 = 2` | orthogonal projection, rank 1 |

Two facts carry the whole thing and both are the user's:

- **`tr J = 2` is not a hypothesis. It is the Born normalisation.** `tr(a a^dag) =
  2 <a^dag a>_0` holds exactly for every `a`, null or not. So "trace equals two" and
  "the state is normalised" are the same sentence.
- **`det J = N(A) g^2 + 2<A,B> g s + N(B) s^2`** (sympy residual exactly 0), with
  `N(A) = sum A_mu^2` and `<A,B> = sum A_mu B_mu` **bilinear**, not sesquilinear. The
  null condition is a quadratic form in the two exp maps whose coefficients are built
  from the amplitude vectors alone.

**The headline consequence: `H` and `S` never enter the null condition.** The exp maps
carry the dynamics; the shared quaternionic direction carries the nullity. That division
of labour is what the quaternion-channel parameterisation is *for*, and it is why a
complex scalar cannot do the job: a scalar has nowhere to put the direction, so it can
only reach the cone by vanishing, which kills the state.

Corollary, verified: **closure IS the null cone.** Not "closure lands on it" -- the
shared-direction condition and `det J = 0` are the same statement.

---

## 2. Negatives. Keep these; they are the corrections.

### N1. `J^2 = 2J <=> det J = 0 AND tr J = 2` is FALSE

The `<=` direction holds unconditionally (Cayley-Hamilton), and that is the direction
section 1's chain uses, so the derivation is unaffected. The `=>` direction fails:

```
J = 0    satisfies J^2 = 2J with tr J = 0, NOT 2
J = 2 I  satisfies J^2 = 2J with det J = 4, NOT 0
```

`J = 2I` is the maximally non-null element of `M_2(C)` and still obeys `J^2 = 2J`.
**So `J^2 = 2J` does not put the current on the null cone.** Anyone reading "2J = |J|^2"
as a premise and inferring nullity has the implication backwards.

Repair, verified over 4 exact rows plus 200 random controls (80/200 with LHS true, so not
a double-negative):

```
J^2 = 2J  AND  rank J = 1   <=>   det J = 0  AND  tr J = 2
```

The solution set of `J^2 = 2J` is three strata: `{0}`, `{2E : E rank-1 idempotent}`,
`{2I}` -- ranks 0, 1, 2. That ladder is complete because `M_2(C)` has no rank 3.

### N2. `<J^dag J>_0 = 2 x rank` is FALSE off the Hermitian branch

`J^2 = 2J` makes `J/2` **idempotent**, which is not an **orthogonal projection** (that
needs `E^dag = E` as well). Both live in the solution set:

```
                        E type  rank  E^dag == E?  tr(E^dag E)  <J^dag J>_0
        orthogonal rank-1 proj     1         True       1.0000       2.0000
     OBLIQUE rank-1 idempotent     1        False       1.0846       2.1693
```

True statement: `<J^dag J>_0 = 2 tr(E^dag E) >= 2 rank(E)`, equality iff `E` is an
orthogonal projection. Closed form (independently derived by the verifier): for
`E = v w^dag/(w^dag v)`, `<J^dag J>_0 = 2/|<v_hat, w_hat>|^2 >= 2`, equality iff `w || v`.

**det and trace cannot distinguish the orthogonal from the oblique rank-1 idempotent.**
Both give `tr = 2`, `det = 0`, `J^2 = 2J`. So `tr J = 2 AND det J = 0` does **not** pin
the Born value to 2. Hermiticity is a genuinely separate ingredient -- supplied, for free,
by `J = a a^dag`, and not by `J = a b` for a generic partner `b`.

### N3. The wedge cannot give `2J`. Only the geometric product can.

`a ^ b = (ab - ba)/2` is a commutator and is **traceless identically**. Cayley-Hamilton
then gives `W^2 = -det(W) I`, so a wedge is never `2W`; on the null cone it is
**nilpotent**, `W^2 = 0`. The wedge lands on the nilpotent branch, the geometric product
on the idempotent branch. "Either wedge or geo prod" is too loose: the `2J` claim needs
the geo prod specifically, with a null factor and unit scalar part.

Rank descent verified: `rank(ab) <= min(rank a, rank b)`. Once a null element (rank 1)
enters a product the rank is capped at 1 and nothing downstream restores it -- a one-way
door. This is the "return back down to rank-1" claim, and it is a rank inequality.

### N4. The Z_4/Z_2 cover is real; "tensor and spinor windings" is not

Verified: `g` = "integrate once" acts on the real channel by `-1` (order 2) and the
complex channel by `1/i = -i` (order 4). `ker(rho_real) = {1, g^2} ~ Z_2`, so the real
rep factors through `Z_4/{+-1}` while the complex rep is faithful. One rep faithful, one
factoring through the quotient by the centre: that is the covering relation.

It is the spinor double cover restricted to a cyclic subgroup, up to conjugacy.
`U = -i sigma_x` has order 4 in SU(2), order 2 in SO(3), and `U^2 = -I` is invisible
downstairs. **But `U != M(i)`**: cal's `i` embeds to `i sigma_y`, and `<U>` and `<M(i)>`
share only `{+-I}`. They are conjugate (`V<U>V^dag = <M(i)>`, `V = exp(-i pi sigma_z/4)`),
so the covering structure is identical, but "`<i>` IS `<U>`" would be false.

**What is NOT proved:** the channels are scalar amplitudes with no rep index; nothing
rotates them; `n` is an integer so `Z_4` is finite and there is no path, hence no winding
number. A winding needs a continuous parameter, which needs the fractional integral back.
The rate reading (complex advances pi/2 per integration, real advances pi, so the complex
is at half rate -- the spinor-vs-vector relation, in the right direction) is suggestive
but a rate ratio is not a rep index.

---

## 3. Manuscript defects (born_axiomatic.tex)

### D1. `W` names two objects. `ax:current` should define `J`, not `W`.

- `def:state`: `W = <psi|psi_0>`, "a single complex scalar ... not a full biquaternion".
  **This is correct.** An inner product of two vectors is one number. No complaint.
- `ax:current`: `W(x,t) = A e^{-H/lambda} + B e^{iS/heff}`, "the source current". But
  `ax:cal` already calls the source current `J(x)`, and `def:state` says `psi = J(x)`.
- The glossary makes the collision explicit: "`W` closed two-channel amplitude,
  `A e^{-H/lambda}+B e^{iS/heff}`; at the delta-limit, `W = <psi|psi_0> = g + i t_p`."

Empirically, on a central scalar:

```
  sigma acts trivially on every central scalar? True
  dagger == plain complex conjugation on the centre? True
```

`sigma` (quaternion reversal) is the **identity** on the central C -- there is no vector
part to negate. So `dagger = sigma . kappa` collapses to `kappa`, and the abstract's
"unique anti-involution, composite of two commuting involutions" **is complex
conjugation**, whose uniqueness on C is Galois (`Gal(C/R) = Z_2`), not biquaternions.
`sum_mu |W_mu|^2` becomes `|W_0|^2 + 0 + 0 + 0`: not wrong, **empty**.

Every structure the paper credits for forcing the closure is absent from a scalar:
`sigma` nontrivial, two distinct involutions, `det = 0` with `W != 0`, rank 1,
a `W^2 = 2W` family, `(4,4) -> (1,3)`, and `lem:twoclosures`(c)'s zero divisor -- all
gone. `lem:unique`'s dimension ladder is a theorem about `R^8`; the scalar's space of
real quadratic forms is `2*3/2 = 3`, and positivity plus U(1)-invariance leaves `|W|^2`
in three lines of undergraduate algebra.

**Fix:** rename `ax:current`'s `W` to `J`. Keep `def:state`'s `W` as the overlap. Then
`<J^dag J>_0 = sum_mu |J_mu|^2` has a `mu` that ranges, and every structural claim lands
on the object that has the structure.

### D2. Reality and positivity are free. The abstract's causal claim is false.

The abstract says reality and non-negativity are "derived: at alpha=2 the rotation of
Axiom `ax:angle` places the dagger closure on the real axis". Swept over `[0,4)`:

```
     alpha     theta     <W^dag W>_0     imag part   real?   >= 0?
    0.0000    0.0000        9.042531      0.00e+00    True    True
    1.0000    1.5708        4.237953      0.00e+00    True    True
    2.0000    3.1416        9.042531      0.00e+00    True    True
    3.5000    5.4978       12.439882      0.00e+00    True    True
  violations of real-and-non-negative across the whole alpha sweep: 0
```

`<W^dag W>_0 = sum_mu |W_mu|^2` is a sum of squared moduli: real and non-negative for
**every** biquaternion, at **every** alpha, with **any** kernel. Reality and positivity
are outputs of the **dagger** (`def:conj`), not of the closure order. You had them before
`ax:angle` was written.

This is worth fixing because the true statement is stronger: positivity was never in
doubt, so claiming to derive it invites a referee to notice it was free.

### D3. `tr[G,S]` is identically zero. The interference term was never nonzero.

`lem:unique`'s `tr(W^dag W) = ||G||^2 + ||S||^2 + tr[G,S]` is a **true identity**
(verified: 18.085062 both sides). But `tr[G,S] = 0` by the cyclic property of the trace,
for any two matrices, always -- regardless of alpha, of dagger-parity, of the kernel.
Presenting it as the interference that alpha=2 cancels credits alpha=2 with work
`tr(AB) = tr(BA)` already did.

### D4. `def:alpha` calls alpha the order. The framework's order is `nu`.

`def:alpha`: "The *fractional order* alpha in [0,4) is the single dial of the accumulation
law, weighing present against memory." That welds two jobs onto one symbol. The framework
already separates them: the author's own slide defines "Effective Order (Surfer's
Marginal)" as `nu-bar(x) = INT nu p_Psi dV / INT p_Psi dV`, while the law reads
`T = alpha(present) - (1-alpha) INT k(tau)(past) dtau`. And `born_axiomatic.tex` already
uses `nu_min` in the contraction constant `kappa' <= (1 - nu_min)/2`.

**alpha is the blend coefficient. nu is the order.** The alpha-does-two-jobs problem is a
symbol error in `def:alpha`, not a defect in the framework.

Sharpened by the integer-integration result: if alpha is literally the number of
integrations, `ax:cal`'s `alpha J` multiplies `J` by a count, and the mod-4 belongs
entirely to the count -- the coefficient in `alpha J` has no period and at alpha=4 gives
`4J`, not `J`.

### D5. `ax:angle` and `lem:unique` use decompositions that agree only at alpha = 0, 2

`ax:angle` writes `W = G + S_c e^{i theta}`; `lem:unique` writes `W = G + S` with
`S^dag = -S`. But `(S_c e^{i theta})^dag = -e^{-i theta} S_c`, which equals
`-S_c e^{i theta}` only when `theta = 0` or `pi`. So the rotated spectral channel is
dagger-odd **only at the two even corners**. Flagged as needing independent verification;
not yet ported to a test script.

---

## 4. What integrating the superposition literally gives you

No operator, no `I^alpha`, no imported `h^4 = 1`:

```
J        = A*exp(-s*u) + B*exp(I*s*w)
INT^1 J  = -A*exp(-s*u)/u   - I*B*exp(I*s*w)/w
INT^2 J  =  A*exp(-s*u)/u**2  - B*exp(I*s*w)/w**2      <- ADDED became SUBTRACTED
INT^4 J  =  A*exp(-s*u)/u**4  + B*exp(I*s*w)/w**4      <- back to (+,+)
```

`INT e^{Ls} ds = e^{Ls}/L`, and `(i w)^2 = -w^2` while `(-u)^2 = +u^2`. The minus is
`i^2 = -1` and nothing else. **`prop:antiparallel` follows with no channel-angle axiom,
no `theta = alpha pi/2`, and no `ax:angle`.**

`1/i^n` has period 4 (`1, -i, -1, i, 1`), so `[0,4)` is half-open because four
integrations is a full turn of `i`. **The Z_4 is derived, not posited** -- it does not
need `h^4 = 1` or the biquaternion. Caveat recorded in the script: the period-4 is exact
for the sign/phase structure after stripping the common `u^n`, not for the full
coefficient unless `w = 1`.

Verified separately: `INT^2` is exactly the inverse of `d^2/ds^2`, so "integrate twice"
is "invert a second-order operator". (The script's own 5c section refutes the
unrestricted form of this: `INT2(D2(s^3+s+1)) - f = -s - 1`; the two differ exactly by
the `span{1,s}` kernel terms. Kept as a negative.)

### Why exp maps and not logs

`z^a = exp(a log z)` and `log z = ln|z| + i(arg z + 2 pi k)`. The branch factor
`e^{2 pi i a k}` equals 1 for every `k` **iff `a` is an integer**. The geometric channel's
rate `lambda = -u` sits **on** the principal branch cut:

```
         eps   arg(-u + i eps)   arg(-u - i eps)       gap
       1e-01       3.091634258      -3.091634258   6.18327
       1e-12       3.141592654      -3.141592654   6.28319
```

The jump is `2 pi` and does not shrink. So `(-u)^{-alpha}` is a **branch choice, not a
number**, at fractional alpha -- including at alpha = 1.9965071. At integer `n`,
`1/(-u)^n` by repeated multiplication equals the log route exactly: the log is
**removable**. There is no repeated-multiplication fallback at fractional order.

Integrate rather than differentiate because `d/ds` multiplies by `L` (gain `w`, unbounded
as `w -> inf`) while `INT` divides by `L` (gain `1/w -> 0`). Both give the same `-1` at
order 2; only integration is bounded. An accumulation law built on differentiation would
amplify every fine detail of the past instead of forgetting it.

---

## 5. Auditor errors, recorded

Five conclusions printed over contradicting data in one session. Recorded because the
failure mode is the finding:

1. `<J^dag J>_0 = 2 x rank` asserted as general; true only on the Hermitian branch (N2).
2. `J^2 = 2J <=> det = 0 AND tr = 2` asserted as a biconditional; `=>` is false (N1).
3. Weyl multiplier numeric run with `eps*T = 0.4` (regulator that did not regulate),
   conclusion printed over divergent data.
4. Sandwich bug in the wedge table: `(A b1 - b2 A)/2` is not a commutator; printed
   `tr = 1.13` for an object proved traceless three sections earlier.
5. `MINIMUM (antiparallel)` labelled on alpha=2 while the adjacent column showed
   alpha=2 equal to alpha=0 and the minimum at alpha=1.

Also: quadratic forms on `R^8` is `36`, not the `20` asserted from an earlier run; the
provenance of that 20 is unresolved and the claim is withdrawn pending re-derivation.

Verifier-found defects in the ported scripts (all fixed): a dead verdict flag that let
section 2 PASS over four contradicting rows (proved by mutation, not inspection); a
rank test that could not fail (`matrix_rank` on a 2x2 is bounded by 2 by shape); 200
"random controls" that were all double-negatives; a sandwich-class redraw inside a
scaling loop; and a near-tautological cal cross-check (`(A/u^2)/A` cancels `A`).
