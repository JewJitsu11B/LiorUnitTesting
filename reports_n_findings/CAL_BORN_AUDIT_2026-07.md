# Audit: the Born-rule derivation at the delta-kernel, alpha = 2 closure

Dated 2026-07-14. Companion scripts: `py tests/exp_clifford_conjugation_identification.py`,
`exp_born_form_uniqueness.py`, `exp_born_paravector_forward_cone.py`,
`exp_born_invariance_vs_positivity.py`, `exp_born_fixedpoint_channel_width.py`,
`exp_frft_even_order_reality.py`, `exp_holomorphic_dual_channels.py`,
`exp_polar_form_interference.py`.

Consistent with the suite's standing rule, negatives here are reported and kept as negatives, and
the auditor's own withdrawn claims are kept visible in section 6.

## 1. Scope and a citation caveat

Audited: `SourceDocs/born_axiomatic.tex` (327 lines) and `SourceDocs/CAL_Unified_Manual.tex`,
algebraically and against the source text. 25 reader/verifier agents across two passes; the second
pass returned 0 of 8 findings refuted under both adversarial refutation and steelman review.

**Caveat on line numbers -- read this before chasing a citation.** `CAL_Unified_Manual.tex` is
under active edit and its line numbers move. It was folded from 6661 to 5013 lines during the audit
(the earlier version is preserved as `SourceDocs/CAL_Unified_Manual_pre-fold.tex`) and had grown to
5149 lines by the time this report was finished. Line numbers cited for the manual are **as of the
5013-line state** and several have already shifted -- e.g. "not a quaternionic imaginary" moved
l.541 -> l.556, and thm:Born_fp's tau claim moved l.1357 -> l.1472.

The **stable anchors are the LaTeX labels and the verbatim quotes**, both given alongside every
manual citation below. Use those; treat the line numbers as a hint. The relevant labels:

| label | what it is |
|---|---|
| `def:source` | the two-channel source current (the additive definition) |
| `def:J-closed` | the closed form, and the `theta_R = theta_I` closure condition |
| `def:channel-angle` | `theta = alpha.pi/2` |
| `thm:Born_fp` | the Born fixed point, and the "no condition on tau" claim |
| `thm:scalar` | the `A = C^2/4` scalar identity (what `Thm. IX.6` should point at) |

Citations to `born_axiomatic.tex` (327 lines) and to the pre-fold manual are stable; neither is
being edited. Some material cited below survives **only** in the pre-fold file. Every citation
names its file.

## 2. Confirmed / positive

**2.1 The dagger form is the Born rule.** `<W^dag W>_0 = sum_mu |W_mu|^2 = |w|^2`: quadratic,
positive definite, and summing to 1 over a complete basis. Not in dispute at any point.

**2.2 The dagger IS reversion; the bar IS Clifford conjugation.** Under `C(x)H = Cl(3,0)` with the
central imaginary identified as the pseudoscalar `e1e2e3`, the paper's two involutions are the two
standard Clifford operations. Verified against the `cal` package to `max|diff| = 0.00e+00` over 200
random biquaternions, with discriminating controls separating cleanly (bar-vs-reversion 7.882).

*Why it matters:* Lemma 4.3 currently argues for the dagger as a bespoke composite `sigma . kappa`.
It is the canonical geometric-algebra reversion norm, which is positive definite on `Cl(3,0)` as a
textbook fact. The lemma can cite that instead of constructing it.

**2.3 `W^dag W` is a forward-cone paravector.** It is always Hermitian PSD, hence grade 0 + grade 1
only (zero bivector and pseudoscalar content), lying in the closed forward light cone of the (1,3)
Hermitian slice, with `t^2 - |x|^2 = det(W^dag W) = |N(W)|^2 >= 0`, null exactly on the zero
divisors. Matrix rank 2 generically, 1 on the null cone. The Born weight is its time component.

*Why it matters:* this is a sharper statement of Cor 6.2 and Cor 6.4 than either currently makes,
and it is available for free.

**2.4 Lemma 4.3's uniqueness has a one-line repair** (see 3.2 for the defect). Requiring invariance
under **left multiplication by the dagger-unit group** `{u : u^dag u = 1} = U(2)` pins the dagger
form to dimension 1, since `<(uq)^dag (uq)>_0 = <q^dag u^dag u q>_0` and `u^dag u = 1`.

The minimal sufficient repair keeps the lemma's own dagger involution and adds **only** this left
action: no bar and no conjugation are needed, because `(uq)^dag = q^dag u^dag` conjugates left
multiplication into right multiplication, so the Z_2 plus left-U(2) already generates the two-sided
unitary action. The surviving form is recovered numerically as `S = I_8` (`max|S - I| = 4.0e-09`)
and evaluates to cal's `<q^dag q>_0` exactly.

The cost should be stated plainly: this is **unitarity** (the spinor action plus a global phase), a
physical premise, not an algebraic fact. So the lemma's "This is a statement about the algebra
alone" cannot be kept -- the algebra alone gives dimension 20, not 1 (see 3.2). It is still one
line, still far cheaper than Gleason, and unlike Prop 5.2 it is true.

*Correction to an earlier draft of this audit.* The repair was first stated as left multiplication
by **unit quaternions (SU(2))**. That is **insufficient**, and the script keeps the point as a
negative. A second family survives it: `G(q) = a||Re q||^2 + b||Im q||^2` with `a != b` is positive
definite, bar/dagger/conjugation-invariant, and invariant under `q -> u q` for every unit *real*
quaternion (`max|diff| = 9.5e-06`), with a non-constant ratio to the dagger form. A real `u` acts on
`Re q` and `Im q` separately and so cannot see that split. `G` dies only to the h-phase `u = h`,
which is dagger-unit (`h^dag h = +1`) but **not** bar-unit (`N(h) = -1`). Hence the group is the
dagger-unit group, not SU(2). The one-line proof was never affected -- it only ever used
`u^dag u = 1` -- but the naming was wrong.

## 3. Defects

**3.1 (headline) The Born fixed point requires `tau = 2`, and the theorem denies it.**

`thm:Born_fp` states (`CAL_Unified_Manual.tex`, label `thm:Born_fp`, ~l.1472 and moving; and
`CAL_Unified_Manual_pre-fold.tex` l.2256-2257): the closure "is fixed by the closure order alpha = 2 alone and **requires no condition
on the channel width tau**." Its own proof preamble (pre-fold l.2261) concedes: "Steps 1-4 verify
that it is a fixed point of T and **read off the channel width it requires**."

The proof is right. Steps 1-4 (pre-fold l.2273, 2285, 2296, 2311) give
`H = -ln|psi|^2`, `G = A|psi|^{2/tau}`, `S = B|psi|^{2/tau}`, `W = (A-B)|psi|^{2/tau}`, and
`W^dag W = (A-B)^2 |psi|^{4/tau}`. Step 1 fed the map the trial `p = |psi|^2`, so a fixed point
requires `p' = p`, i.e. `4/tau = 2`, i.e. `tau = 2`. Numerically the map returns its input only
there (L1 distance 0.645, 0.332, 0.139, **0.000**, 0.108, 0.332 at tau = 0.5, 1.0, 1.5, **2.0**,
2.5, 4.0).

The paper's own defence is the problem: "whatever tau-dependent profile the channels carry, the
conjugate product is a positive, real, **normalizable** density." True, and insufficient. Every tau
gives a normalizable density (`|psi|^8`, `|psi|^4`, `|psi|^1` all integrate to 1). Exactly one gives
Born. `tau` does not enter the closure *order*, which is 2; it enters the closure *output*, as the
exponent `4/tau`. `born_axiomatic.tex` l.285 repeats the claim: "channel width (shapes the
amplitude; does not enter the closure)".

Agent verdict: `claim_supported`, high confidence, independently reproduced. `tau` is never fixed to
2 anywhere in either version.

**3.2 Lemma 4.3's uniqueness claim is false as stated, and the gap is 20-dimensional.**

The lemma asserts the dagger form is "the unique real, non-negative, conjugation-invariant quadratic
form" -- i.e. that the invariant space has dimension 1 -- and that "this is a statement about the
algebra alone". But the only invariance it actually states is `<(W^dag)^dag W^dag>_0 = <W W^dag>_0`,
which is invariance under the dagger involution itself: a Z_2.

Explicit counterexample: `F(q) = a|q0|^2 + b(|q1|^2+|q2|^2+|q3|^2)` with `a != b` (a=2, b=1) is real,
non-negative, positive definite (min over the unit sphere = min(a,b) = 1.0001), and invariant under
bar, under dagger, and under `q -> u q u^dag` -- every invariance the lemma states -- while not being
proportional to the dagger form (ratio `F/dagger` ranges 1.017 to 1.748 over random q, so it is not
a rescaling). Conjugation is blind to it because `q -> u q u^dag` fixes `q_0` exactly
(`max|dq_0| = 5.3e-07`) and acts on `(q1,q2,q3)` by a recovered real `R` with `R^T R = I`,
`det R = +1.000000`: it never moves weight between the `q_0` slot and the `q_k` slots.

Measured directly, the dimension of the space of invariant real quadratic forms on R^8 (Reynolds
average, SVD nullspace, clean gap ~2e+00 vs ~5e-07):

| invariance imposed | dim |
|---|---|
| dagger involution alone (**what the lemma states**) | **20** |
| + bar | 14 |
| + conjugation `q -> u q u^dag` | 4 |
| + left SU(2) | 2 |
| + left dagger-unit group U(2) | **1** |

So the lemma asserts dimension 1 while invoking only the Z_2, under which the true dimension is 20.
Repair in 2.4.

**3.3 The amplitude is a complex scalar; the quaternion structure is unused at the Born step.**
Def 2.6 and Axiom 4 make `W = <psi|psi_0>` "a single complex scalar in the central C factor", and
the manual concurs ("a complex scalar carrying no tensor or fiber index"). Read literally, Axiom 2's
`W` has grade content {0, 3} and zero bivector part. Theorem 6.1 therefore reduces to
`w-bar w = |w|^2 >= 0` for `w` in C -- true, but a fact about C, not about `C(x)H`. The algebra earns
its keep in Lemmas 4.1(c), 4.2 and Cor 6.4; not in the Born derivation.

**3.4 Lemma 4.4's interference cancellation is circular.** `|z|^2 = (Re z)^2 + (Im z)^2` is an
identity true of every complex number; decomposing into Re/Im and then observing that no cross term
appears establishes nothing. The interference is inside `g^2`: with `g = G + Sc cos(theta)`,
`g^2 = G^2 + 2 G Sc cos(theta) + Sc^2 cos^2(theta)`. Prop 5.2 writes the same term explicitly as
`2 G Sc cos(theta)`. The two are the same identity in two parameterizations; 4.4 hides the term that
5.2 displays.

**3.5 Def 2.2's corner list has a sign error.** `born_axiomatic.tex` l.67 states the Caputo factor as
`(i.omega)^{-alpha} = omega^{-alpha} e^{-i.alpha.pi/2}`, then lists "At the integer corners,
`e^{i.theta} = 1, i, -1, -i` for alpha = 0,1,2,3" -- which is the conjugate. At alpha = 1 the operator
factor is `-i`, not `+i`. Invisible at alpha = 0 and 2 (where `e^{i.pi} = e^{-i.pi} = -1`), so the
alpha = 2 closure is protected from it; wrong at alpha = 1 and 3. The manual is clean here: it writes
the corners as `cos(alpha.pi/2) = 1, 0, -1, 0`, which is even and therefore sign-blind.

**3.6 "Split (2,2)" should be (4,4). RESOLVED IN BOTH FILES.**
`born_axiomatic.tex` l.38 (abstract) and `lem:signature` described the carrier as having a "split
(2,2) norm" and a "split-signature (2,2) carrier". `Re N(q) = sum(x_mu^2 - y_mu^2)` on R^8 has
eigenvalues `[-1,-1,-1,-1,+1,+1,+1,+1]`: signature **(4,4)**.

**Provenance -- an earlier draft of this report had this wrong.** It was first written up as a typo
in the Born paper contradicting its parent, on the grounds that `CAL_Unified_Manual.tex` had it
right in the closure-chain table (~l.393): "Derived from the carrier's `(4,4) = (1,3) (+) (3,1)`
split". That table was right, but the manual *also* asserted `(2,2)` in its projection-cascade
subsection: the heading read `(6,2) -> (2,2) -> (1,3)`, and the prose called `C(x)H` "the `(2,2)`
algebraic carrier ... whose intrinsic split-signature norm is `(2,2)`". The manual contradicted
*itself*, l.393 against the cascade, and `born_axiomatic.tex` was faithfully copying the cascade.
The Born paper was not mistyping; it was inheriting.

The tell was internal to that paragraph: it insisted "the dimension count is preserved (eight real
either way)" four lines after labelling those eight dimensions `(2,2)`, which is four-dimensional.
Under `(2,2)` the first arrow silently loses four dimensions while the prose claims it does not.
Only `(4,4)` makes the cascade's own bookkeeping work: `(6,2) -> (4,4) -> (1,3)` is `8 -> 8 -> 4`,
arrow 1 a change of operative quadratic form (dimension preserved, as stated) and arrow 2 the
Hermitian projection discarding the `(3,1)` anti-Hermitian summand.

`(2,2)` is a legitimate quaternionic signature -- it is the split-quaternion norm, carried here by
the real span of `{1, i, hj, hk}` giving `x0^2 + x1^2 - x2^2 - x3^2`. It is a four-dimensional
subalgebra, not the eight-dimensional carrier that `Herm(q)` projects from. That is almost
certainly where the number came from.

**Both files are now fixed.** The manual's cascade reads `(6,2) -> (4,4) -> (1,3)` with an explicit
dimension column, and adds a derivation this audit did not have: for `Z = a + hb` with `a,b` in `H`,
`N(Z) = (|a|^2 - |b|^2) + 2h<a,b>`, whose real part is manifestly neutral (confirmed here on random
trials). It now states outright that `(2,2)` "belongs to the split-quaternion subalgebra
`span_R{1, i, hj, hk} = M_2(R)` ... a four-dimensional *part* of the carrier rather than the
carrier", independently reaching the same subalgebra identified above. `born_axiomatic.tex` l.38,
l.54 and `lem:signature` now say `(4,4)`, with `lem:signature` naming the `(3,1)` kernel it
annihilates.

**3.6a The glossary's direct sum should be a superposition.** `CAL_Unified_Manual.tex` l.4762
(pre-fold l.6412) glosses the source current as "cost/thermal `(+)` phase/action", using the
direct-sum symbol. Both body texts disagree with it: `born_axiomatic.tex` l.106 says "The source
current is the **superposition** of...", and `CAL_Unified_Manual.tex` l.541 says "the **sum** of...".

The distinction matters and runs the paper's way, not against it. A direct sum asserts independent
(orthogonal) subspaces, giving `<1|2> = 0` and hence `|c1 + c2|^2 = |c1|^2 + |c2|^2` with **no cross
term and no interference** -- which would destroy the mechanism l.2683 depends on ("the Born rule
recovers probability from the interference of the two"). The two channels are both multiples of the
identity in the central C factor, so they are parallel rather than orthogonal, and the correct word
is superposition.

**Status: FIXED.** The glossary entry in `CAL_Unified_Manual.tex` now reads "cost/thermal `+`
phase/action (a superposition, not a direct sum: the channels are parallel, so the cross term
survives and is what the Born weight reads)". The other 22 uses of `(+)` in the manual are genuine
direct sums (spinor polarization `V = W (+) W*`, the `(4,4) = (1,3) (+) (3,1)` split, and
representation decompositions) and were left alone.

**3.7 A and B cancel; A = B divides by zero; "structure constants" is unbacked.** The normalized
measure is `(A-B)^2|W_psi|^2 / [(A-B)^2 int |W_psi|^2]`, so A and B drop out entirely and Axiom 2's
coefficients do no work in the result. At `A = B` the amplitude is identically zero and the stated
normalization divides by zero, so the derivation needs `A != B`, which no axiom supplies. Axiom 2
says A and B are "fixed by the algebra's structure constants"; grep for "structure constant" in the
manual returns only *fine*-structure constant. In the manual they are simply "the real channel
amplitudes" (l.537) and "the real amplitude coefficients" (l.1009).

**3.8 The Born weight is a time component, not a Lorentz scalar.** In the paravector encoding the
manuscript commits to, grade-0 IS the timelike direction, and SL(2,C) boosts mix grade 0 into grade
1: `t` changes while `t^2-|x|^2` is invariant. This is *not* by itself fatal -- it is the ordinary
Dirac-current situation, where `rho = psi^dag psi` is the time component of a conserved 4-current --
but the repair requires current conservation plus a spacelike-slice integral, which is precisely the
"external partition integral" Thm 6.1 advertises not needing. The claim that has to give is "no
external partition integral", not the algebra.

## 4. Editorial

**4.1 The delegation is circular.** `CAL_Unified_Manual.tex` l.1369 says the Born result is "Derived
in full in \cite{CAL_born2}". `born_axiomatic.tex` l.258 says the fixed point "is Theorem VI.1 of
\cite{CALmain} and is **cited, not re-proved**". Each defers to the other.

**4.2 The fold removed the proof but kept the claim.** `thm:Born_fp`'s Steps 1-4 -- including the
"read off the channel width it requires" admission -- exist only in
`CAL_Unified_Manual_pre-fold.tex`. The current manual retains the theorem statement at l.1357 and
delegates the derivation to the Born paper, which does not carry it (grep "read off the channel
width": pre-fold 1, current 0, born_axiomatic 0). The tau arithmetic is currently in no live file.

**4.3 Load-bearing citations do not resolve.** `born_axiomatic.tex` cites `\cite[Thm.~IX.6]{CALmain}`
**three times** -- including for the `A = C^2/4` scalar identity that Cor 6.5 and Thm 6.1 Step 4 both
lean on -- plus `\cite[Thm.~IX.4]` and `\cite[\S IX]`. The manual has Parts I, II, III, V, VI, VII
(the pre-fold version had I-VII; the fold dropped Part IV). **There is no Part IX in either
version.** The theorem itself exists under the label `thm:scalar`, just not under that number.
Part VI is "The fine-structure constant", so "Theorem VI.1" is not the Born fixed point either.
Likely stale numbering from an earlier draft, but it should be repaired, since Thm IX.6 is doing
real work.

## 5. Proposals assessed

**5.1 Holomorphic duals: good mathematics, absent from the text.** With `f = -2 log(psi)` for
holomorphic `psi`, `Re f = -ln|psi|^2 = H` (exactly Step 1's driver) and `Im f = -2 arg(psi)`.
Verified: harmonic, with both Cauchy-Riemann relations holding, for `psi = z`, `z^2+1`, `e^z`. So H
does have a holomorphic dual and it is `arg(psi)` -- the wavefunction's phase, which is what
interference is made of. This is a better structure than the manuscript's.

But it is not in the manuscript. Agent verdict `claim_unsupported`, high confidence: no
Cauchy-Riemann condition, analyticity requirement, or Hilbert-transform relation appears anywhere;
"Cauchy" and "Riemann" have zero occurrences in the current file; `sec:geospec` builds a pointwise
algebraic Re/Im slot-pairing on the algebra, not a function-theoretic relation; `sec:source`
(l.537-541) introduces `H^D` and `S^D` as two separately named drivers with no relation asserted.

And the manuscript takes the opposite road at pre-fold l.2295: `S := B exp(i S^nu/hbar_eff) =
B|psi|^{2/tau}`. For real `S^nu` the modulus is identically `B`, so this forces `S^nu` imaginary
(`S^nu = -2i hbar_eff ln|psi| / tau`) -- a Wick rotation. An imaginary S is not H's harmonic
conjugate; it is `i` times it.

**5.2 The quaternionic polar form is incompatible.** The proposal
`J = rho e^{n_hat theta} = exp[-H/tau + n_hat S/hbar_eff]` fails three ways:

- *No interference.* `|rho e^{n theta}|^2 = rho^2` identically, because `|e^{n theta}| = 1`. The
  action drops out of the Born weight entirely. The manuscript needs the opposite: the manual says "the Born rule `P = |W|^2` recovers
  probability from the **interference of the two**" (~l.2798). Interference comes
  from addition of amplitudes, never from the polar decomposition of one.
- *It contradicts centrality by name.* `CAL_Unified_Manual.tex`, just after label `def:source`
  (~l.556): "The unit `i` is the privileged commuting imaginary of the C factor, **not a
  quaternionic imaginary**"; and (~l.622) the phase "must not depend on a **preferred quaternionic
  direction**". A unit imaginary quaternion is
  exactly that, and does not commute with the orthogonal generators. The centrality argument is what
  *generates* the substrate ("Quaternions alone are insufficient because they lack a
  privileged commuting imaginary direction", in the same subsection as `def:source`), so adopting the polar form collapses the reason the
  framework has `C(x)H` at all.
- *It collapses the algebra.* `J = rho e^{n theta}` lives entirely in the slice
  `C_n = span{1, n_hat}`: 2 of the algebra's 8 real dimensions.

`Laplacian J = 0` is true but trivial, and weaker than even that phrasing suggests. `Delta = 4
d_z d_zbar`, so `e^F` is harmonic whenever `F` is holomorphic (`d_zbar F = 0`) **or**
anti-holomorphic (`d_z F = 0`). Verified: `F = zbar` also gives `lap(Re e^F) = 0`. A control that
is neither (`F = |z|^2`) is the only one that breaks harmonicity. So the harmonicity of `J` does
not even single out holomorphy, let alone this particular `F`.

**5.3 The decisive incompatibility.** `def:J-closed` (`CAL_Unified_Manual.tex`, label
`def:J-closed`, ~l.1115) sets
"At closure the two coincide, `theta_R = theta_I =: varphi(x,t)`". Impose holomorphy on top: if
`u = v` and Cauchy-Riemann holds (`u_x = v_y`, `u_y = -v_x`), then `u_x = u_y` and `u_y = -u_x`,
forcing `u_x = u_y = 0`. Both drivers are constant and `J` is constant. **The manuscript's own
closure condition trivializes the holomorphic-duality claim**; they are mutually exclusive except
for constant fields.

**5.4 "The complex channel becomes real when alpha is even" -- half right.** The FrFT multiplier
`e^{-i alpha pi n/2}` is real on every Hermite mode exactly when alpha is even, and `F^alpha` maps
real functions to real functions iff alpha is even (`F^2` is parity, confirmed against `f(-x)`).
But it only *preserves* reality; it cannot *create* it. `F^alpha` at even alpha is real-linear, so it
commutes with Re and Im and cannot move anything between them. Fed the complex action channel
`B e^{iS(x)}`, the output is complex at every alpha including the even ones. Reality at alpha = 2 is
inherited from a real input, never produced -- and Axiom 2 calls the action channel "an imaginary
unitary (action) channel".

## 6. Auditor corrections

Claims the auditor made and withdrew, kept for the record:

- "A real amplitude means no interference." **False.** Real amplitudes interfere via sign;
  `|a+b|^2 != |a-b|^2`, with a true null at `a = b`.
- "Degree 4 is the alpha = 4 corner's shape." **There is no alpha = 4**; the domain is `[0,4)` with
  corners 0,1,2,3, and alpha = 0 is real-but-trivial (the present term vanishes).
- "Signature is a convention." **Not here.** `N(q) = sum q_mu^2` plus the slice's reality condition
  (`q_k = i a_k`) produces `(+,-,-,-)` with no convention inserted, provided the metric is identified
  with the algebra's own reduced norm.
- "The algebra offers positive XOR invariant." **Too strong.** `|N(W)|^2` is both -- but quartic and
  degenerate on the null cone, and it returns `|w|^4` rather than `|w|^2`. The dilemma is real for
  *quadratic* forms only.
- "`theta_R = alpha.pi/2` forces `|psi|` constant." **False**, agent-refuted: `theta_R` and
  `theta_I` are only "in the same family as" `alpha.pi/2`, never equal to it. (The correct version of
  this objection is 5.3.)
- An extended argument against `|N(W)|^2` as a proposed Born form. The author never proposed it; it
  arose from a notation collision on "N" (the auditor's reduced norm vs the author's `| |`).
