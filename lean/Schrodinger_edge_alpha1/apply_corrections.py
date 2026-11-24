"""Apply the corrections found by the independent checks to the Schrodinger-edge paper.
Reads reference/paper_snapshot.tex (the paper as it stood), writes paper/cal_schrodinger_collapse.tex.
Every replacement must match exactly once."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "reference", "paper_snapshot.tex")
dst = os.path.join(here, "paper", "cal_schrodinger_collapse.tex")
s = open(src, encoding="utf-8").read()

R = []


def rep(a, b):
    R.append((a, b))


# abstract
rep(r"Each mode evolves as $\exp[-(i+\sqrt{1-\alpha})(\hbar/2m)k^2t]$.",
    r"To leading order each mode evolves as $\exp[-(i+\sqrt{1-\alpha})(\hbar/2m)k^2t]$, times the uniform loss.")

# Lemma W
rep(r"The right side is analytic in $\varphi$ off the points $2\pi\mathbb Z$.",
    r"The right side is analytic in $\varphi$ off the rays $2\pi k+i[0,\infty)$, $k\in\mathbb Z$.")
rep(r"Numerically, \eqref{eq:jonquiere} agrees with direct summation to $10^{-14}$ wherever the series converges, and with an independent arbitrary-precision evaluation to $3\times10^{-16}$ (\S\ref{sec:num}).",
    r"Numerically, the double-precision evaluation of~\eqref{eq:jonquiere} agrees with direct summation to $10^{-14}$ wherever the series converges and with an arbitrary-precision evaluation to $5\times10^{-15}$; at $40$ digits the two sides agree to $10^{-44}$ (\S\ref{sec:num}).")

# Thm rotation: gap, next order
rep(r"against $\pi/4=0.78540$." + "\n" + r"\end{proof}",
    r"against $\pi/4=0.78540$. The gap is the next term, $-\tfrac14\sqrt\eps\ln\eps$ (Remark~\ref{rem:nextorder})." + "\n" + r"\end{proof}")
rep(r"and the mismatch $\eps\pi/2$, entering through $\ln(i\varphi)$, is the term $(\pi/4)\eps^{3/2}$." + "\n",
    r"and the mismatch $\eps\pi/2$, entering through $\ln(i\varphi)$, is the term $(\pi/4)\eps^{3/2}$." + "\n\n"
    r"\begin{remark}[Next order; a singular limit]\label{rem:nextorder}" + "\n"
    r"Carrying the expansion one order further," + "\n"
    r"\begin{align*}" + "\n"
    r"{\rm Re}\,\varphi&=\sqrt{\eps\hat g}\,\Big[1+\eps\Big(\tfrac14\ln(\eps\hat g)+\tfrac12-\tfrac{\hat g}3-\tfrac1{8\hat g}\Big)\Big]+\dots,\\" + "\n"
    r"{\rm Im}\,\varphi&=\tfrac12\eps(1-\hat g)+\tfrac\pi4\eps^{3/2}\sqrt{\hat g}+\eps^2\Big[-\tfrac{\hat g}4\ln(\eps\hat g)+\tfrac{(1-\hat g)(2-\hat g)}4\Big]+\dots" + "\n"
    r"\end{align*}" + "\n"
    r"The logarithms make $\alpha\to1^-$ a singular limit: the edge is not reached by a power series in $\eps$, and convergence is slow. At $\hat g=1$ the exact decay exceeds the leading term $(\pi/4)\eps^{3/2}$ by $13.6\%$ at $\eps=10^{-2}$ and by $2.9\%$ at $\eps=10^{-4}$. Exact roots at $40$ digits, for $\eps$ down to $10^{-40}$, return the coefficient $\pi/4$ to $34$ digits and agree with both displays (\S\ref{sec:num})." + "\n"
    r"\end{remark}" + "\n")

# Thm unitary second display
rep(r"\exp\Big[-\big(i+\sqrt{1-\alpha}\big)\frac{\hbar}{2m}k^2t-\frac\pi4(1-\alpha)\,t\Big].",
    r"\exp\Big[-\big(i+\sqrt{1-\alpha}-\tfrac\pi4(1-\alpha)\big)\frac{\hbar}{2m}k^2t-\frac\pi4(1-\alpha)\,t\Big].")

# Prop orientation
rep(r"At $\eps=10^{-4}$ the counter-rotating fraction of the norm is $10^{-7}$--$10^{-8}$ (Table~\ref{tab:packet}).",
    r"At $\eps=10^{-4}$ the counter-rotating amplitude is $10^{-7}$--$10^{-8}$ of the co-rotating one (Table~\ref{tab:packet}).")
rep(r"and a two-cycle history sends $70$--$97\%$ of the norm into the counter-rotating mode.",
    r"and a two-cycle history excites the counter-rotating mode at $0.68$--$0.97$ of the co-rotating amplitude, as much as a zero past does.")

# Thm feedback
rep(r"gives $J_{n+1}=J_n$: the carrier is frozen.",
    r"gives $J_{n+1}=J_n$: the carrier is frozen. This holds for any past whose memory sum converges, for instance a past of finite length. With flat weights and a past that does not decay the sum diverges, and the statement is then the factor $1-\alpha=0$ taken first.")
rep(r"\item The only bounded response to a source at $n_0$ decays in both directions from $n_0$. With flat weights it is proportional to $e^{-s\lvert n-n_0\rvert}$. It is nonzero before the source acts, so it is advanced.",
    r"\item Add a source, $J_{n+1}=T_n+S_n$. The only bounded response to an impulse $S_n=\delta_{n,n_0}$ decays in both directions from $n_0$. With flat weights at $k=0$ it is" + "\n"
    r"\[" + "\n"
    r"J_n=-\frac{1-e^{-s}}{2\sinh s}\,e^{-s(n_0-n)}\quad(n\le n_0),\qquad J_n=+\frac{1-e^{-s}}{2\sinh s}\,e^{-s(n-n_0-1)}\quad(n>n_0)." + "\n"
    r"\]" + "\n"
    r"It is nonzero before the source acts, so it is advanced, and it changes sign across the source.")
rep(r"For~(3), with flat weights the homogeneous recurrence is $J_{n+1}-(1+\alpha)J_n+J_{n-1}=0$. Its two-sided bounded Green's function is $e^{-s\lvert n-n_0\rvert}/(2\sinh s)$.",
    r"For~(3), with flat weights at $k=0$, subtracting consecutive updates gives $J_{n+1}-(1+\alpha)J_n+J_{n-1}=S_n-S_{n-1}$: the three-term recurrence is driven by the difference of the source. The bounded Green's function of the left side is $G_n=-e^{-s\lvert n\rvert}/(2\sinh s)$, and the response to an impulse at $n_0$ is $G_{n-n_0}-G_{n-n_0-1}$.")

# Thm normalizable proof
rep(r"Equality throughout needs $z=1$ and $\hat g=1$, where $F(1)=2(1-\alpha)\neq0$.",
    r"Equality throughout needs $z=1$, where $F(1)=(1-\alpha)(1+\hat g)\neq0$.")

# small hypotheses
rep(r"\item $J$ is a product state $\iff$", r"\item for $J\ne0$, $J$ is a product state $\iff$")
rep(r"For example, $J_{AB}=\Phi^+\eta^\dagger$ has rank one", r"For example, $J_{AB}=\Phi^+\eta^\dagger$ with $\eta$ a unit vector has rank one")
rep(r"\langle k^2\rangle\big]$ per radian. The readout drift", r"\langle k^2\rangle\big]$ per radian, to leading order in $1-\alpha$. The readout drift")

# numerics: independent checks paragraph
rep(r"Methods are described in Appendix~\ref{app:num}.",
    r"Methods are described in Appendix~\ref{app:num}." + "\n\n"
    r"\emph{Independent checks.} The statements were re-tested with separate code: exact algebra in sympy ($70$ checks), exact roots at $40$ digits in mpmath, direct iteration in NumPy, and a Lean~4 file with Mathlib. The one exact check that failed is corrected in Theorem~\ref{thm:feedback}(3). Lean proves Theorem~\ref{thm:normalizable}, Lemma~\ref{lem:flat}, the limit $w_j\to1$ of Proposition~\ref{prop:embedding}, Proposition~\ref{prop:carrier}(3)--(4), Proposition~\ref{prop:lorentz}(1) and~(3), and the proof step of Lemma~\ref{lem:minors}, with no \texttt{sorry} and no added axioms. Theorem~\ref{thm:rotation} is confirmed, not proved: its expansion is reproduced symbolically and matched by exact roots. A root count by the argument principle finds no root with $\lvert z\rvert\ge1$ for $0<\alpha<1$, and inside the circle only the conjugate pair. Numbers below that the script does not print come from these checks.")
rep(r"An independent arbitrary-precision evaluation agrees to $3\times10^{-16}$.",
    r"An arbitrary-precision evaluation agrees with it to $5\times10^{-15}$, and at $40$ digits the two sides of~\eqref{eq:jonquiere} agree to $10^{-44}$.")

# packet table
rep(r"and compared with free Schr\"odinger evolution after aligning the global phase.",
    r"and compared with free Schr\"odinger evolution after aligning the global phase and rescaling the CAL state to unit norm. The last two columns give the norm lost and the distance without that rescaling.")
rep(r"\begin{tabular}{cccccc}" + "\n" + r"\toprule" + "\n" + r"$1-\alpha$ & $\langle r^2\rangle$ & updates & $\lVert\text{CAL}-\text{Schr}\rVert$ & floor ($k^4$) & counter-rotating\\",
    r"\begin{tabular}{cccccccc}" + "\n" + r"\toprule" + "\n" + r"$1-\alpha$ & $\langle r^2\rangle$ & updates & $\lVert\text{CAL}-\text{Schr}\rVert$ & floor ($k^4$) & counter & norm lost & unscaled\\")
for old, extra in [
    (r"$10^{-2}$ & $0.06$ & $1010$ & $6.96\times10^{-2}$ & $1.14\times10^{-2}$ & $5.6\times10^{-5}$\\", r" & $61.8\%$ & $6.19\times10^{-1}$\\"),
    (r"$10^{-3}$ & $0.06$ & $3168$ & $2.63\times10^{-2}$ & $1.24\times10^{-2}$ & $7.2\times10^{-7}$\\", r" & $10.1\%$ & $1.04\times10^{-1}$\\"),
    (r"$10^{-4}$ & $0.24$ & $2501$ & $4.83\times10^{-2}$ & $4.81\times10^{-2}$ & $1.2\times10^{-7}$\\", r" & $0.89\%$ & $4.89\times10^{-2}$\\"),
    (r"$10^{-4}$ & $0.12$ & $5001$ & $2.57\times10^{-2}$ & $2.48\times10^{-2}$ & $4.3\times10^{-8}$\\", r" & $1.11\%$ & $2.79\times10^{-2}$\\"),
    (r"$10^{-4}$ & $0.06$ & $10002$ & $1.46\times10^{-2}$ & $1.26\times10^{-2}$ & $1.8\times10^{-8}$\\", r" & $1.52\%$ & $2.10\times10^{-2}$\\"),
]:
    rep(old, old[:-2] + extra)
rep(r"The floor is the distance between the exact modes and Schr\"odinger, which is the $k^4$ terms; it halves with $\langle r^2\rangle$. As $\alpha\to1^-$ the error falls onto the floor.",
    r"The floor is the distance between the exact modes, with their decay removed, and Schr\"odinger, which is the $k^4$ terms; it halves with $\langle r^2\rangle$. As $\alpha\to1^-$ the error falls onto the floor. The shape converges in every row; the amplitude does not in the first two, where $62\%$ and $10\%$ of the norm are lost. The counter column is the ratio of counter- to co-rotating amplitude, from a fit with undamped exponentials.")
rep(r"With a two-cycle history instead, $70$--$97\%$ of the norm goes into the counter-rotating mode.",
    r"With a two-cycle history instead, the counter-rotating amplitude is $0.68$--$0.97$ of the co-rotating one.")

# closure, feedback, normalizable tables
rep(r"With both directions present they do not.}",
    r"With both directions present they do not. One random carrier, with $\langle r^2\rangle=0.12$; over eight draws the readout drift at $1-\alpha=10^{-4}$ ranges from $2\times10^{-4}$ to $2.6\times10^{-3}$.}")
rep(r"The criterion gives $b>0.095$ for $\lambda=0.1$ and $b>0.501$ for $\lambda=0.01$.",
    r"The criterion gives $b>0.097$ for $\lambda=0.1$ and $b>0.513$ for $\lambda=0.01$ (with flat weights instead, $0.095$ and $0.501$).")
rep(r"exponential, width $100$ & $10^{-2}$ & $0.99205$ & $7.98\times10^{-3}$\\",
    r"exponential, width $100$, cut at $400$ lags & $10^{-2}$ & $0.99205$ & $7.98\times10^{-3}$\\" + "\n"
    r"exponential, width $100$, untruncated & $10^{-2}$ & $0.99008$ & $9.97\times10^{-3}$\\")
rep(r"summable power law $j^{-1.5}$ & $10^{-2}$", r"summable power law $j^{-1.5}$, cut at $400$ lags & $10^{-2}$")
rep(r"from the companion matrix of the recurrence with $400$ lags.}",
    r"from the companion matrix of the recurrence with $400$ lags. Kernels are cut at $400$ lags and renormalized. For the power law $j^{-1.5}$ the largest root rises with the cut ($0.99217$, $0.99566$, $0.99766$ at $800$, $1600$, $3200$ lags): the untruncated kernel has a branch point at $z=1$, and its slowest decay is algebraic, not exponential.}")
rep(r"over the second half of $6000$ updates.", r"over the second half of $6000$ updates ($3000$ for the runs without a brake).")

bad = 0
for a, b in R:
    c = s.count(a)
    if c != 1:
        print("MATCHES %d: %s" % (c, a[:70]))
        bad += 1
    else:
        s = s.replace(a, b)
if bad:
    sys.exit(1)
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("applied %d replacements -> %s" % (len(R), dst))
