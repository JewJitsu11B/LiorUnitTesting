"""Apply the corrections found by the independent checks to the Feynman alpha = 2 paper.
Reads reference/paper_snapshot.tex (the paper as it stood), writes paper/cal_feynman_alpha2.tex.
Every replacement must match exactly once."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "reference", "paper_snapshot.tex")
dst = os.path.join(here, "paper", "cal_feynman_alpha2.tex")
s = open(src, encoding="utf-8").read()
R = []


def rep(a, b):
    R.append((a, b))


# abstract
rep(r"Just below, both signs of frequency decay at $(\pi/4)(2-\alpha)\mu^2/\omega$; just above, the causal response runs away at the same rate.",
    r"Just below, both signs of frequency decay at $(\pi/4)(2-\alpha)\mu^2/\omega$, and then give way to an algebraic tail $t^{-\alpha}$; just above, the causal response runs away at the same rate.")
rep(r"On mixed frequencies it does not.", r"On mixed frequencies it is not, in general.")
rep(r"Numerical runs confirm each statement." + "\n" + r"\end{abstract}",
    r"Numerical runs and independent machine checks confirm each statement." + "\n" + r"\end{abstract}")

# Prop symbol proof
rep(r"Then $\Gamma(\alpha-1)/\Gamma(\alpha)=1/(\alpha-1)$ gives the first formula, and its limit $\alpha\to1$ gives the second.",
    r"For $\alpha>1$, $\Gamma(\alpha-1)/\Gamma(\alpha)=1/(\alpha-1)$ gives the first formula. For $0<\alpha\le1$ the two Gamma integrals diverge separately while their difference converges; both sides are analytic in $\alpha$ for ${\rm Re}\,\alpha>0$, so the formula extends by continuation, and its value at $\alpha=1$ is the second.")

# Thm near2
rep(r"Write $\omega=E/\hbar$, and $\mu^2=\omega^2-c^2k^2$ at the root.",
    r"Write $\omega=E/\hbar$, and $\mu^2=\omega^2-c^2k^2$ at the root (complex for $\alpha\ne2$; its value at $\alpha=2$ serves to this order).")
rep(r"So below $2$ both signs decay at the same rate, and nothing selects a sign. Above $2$ both grow: the causal response runs away, as for the Abraham--Lorentz--Dirac equation~\cite{Dirac}." + "\n" + r"\end{enumerate}",
    r"So below $2$ both signs decay at the same rate, and nothing selects a sign. Above $2$ both grow: the causal response runs away, as for the Abraham--Lorentz--Dirac equation~\cite{Dirac}." + "\n"
    r"\item Below $2$ the exponential decay is that of the poles only. The symbol also has branch points at $\omega=\pm c\lvert k\rvert$, whose contribution to a sourced solution decays as $t^{-\alpha}$ and oscillates at $c\lvert k\rvert$, not at $\omega$. At late times this tail dominates." + "\n" + r"\end{enumerate}")
rep(r"the symbol of $T$ changes by $-i\pi\delta\,{\rm sgn}(\omega)$, and its $\omega$-derivative is $4\omega/\mu^2$. Hence $\Delta\omega=i\pi\delta\mu^2/(4\lvert\omega\rvert)$.",
    r"the imaginary part of the symbol of $T$ changes by $-\pi\delta\,{\rm sgn}(\omega)$, and its $\omega$-derivative is $4\omega/\mu^2$. Hence ${\rm Im}\,\Delta\omega=\pi\delta\mu^2/(4\lvert\omega\rvert)$. The real part of the symbol also changes at first order, through $\alpha$, $\alpha-1$ and $\lvert m_\alpha\rvert$; that shifts ${\rm Re}\,\omega$ at order $\delta$ and leaves ${\rm Im}\,\omega$ unchanged at this order.")
rep(r"Table~\ref{tab:near2} checks the rate, the count of growing modes, and a time-stepped run." + "\n" + r"\end{proof}",
    r"Table~\ref{tab:near2} checks the rate, the count of growing modes, and a time-stepped run." + "\n\n"
    r"(3) is numerical, with the branch-point origin identified: at $\alpha=1.9$, $k=1$ a time-stepped run follows the exponential to $t\approx150$ and then a power law of fitted exponent $1.90$--$1.91$ at frequency $1.0000=c\lvert k\rvert$; at $t=600$ the response is $1.6\times10^{-6}$, where the pole alone gives $5\times10^{-15}$ (\S\ref{sec:num}, independent checks)." + "\n" + r"\end{proof}" + "\n\n"
    r"\begin{remark}[How good the first-order rate is]\label{rem:near2order}" + "\n"
    r"Exact roots for $\delta=\pm10^{-2}$ down to $\pm10^{-8}$ give ${\rm Im}\,\omega=\tfrac\pi4\delta\mu^2/\lvert\omega\rvert\,(1+c_1\delta)$, converging linearly in $\delta$. With $\mu$ and $\omega$ taken at the root, $c_1=-0.500$, $-0.503$, $-0.688$, $-0.955$ at $k=0$, $0.1$, $1$, $5$. At $k=0$ the rate is exact: ${\rm Im}\,\omega=\rho\cos(\pi/\alpha)$ with $\rho=\lvert\omega\rvert$. The theorem concerns $\alpha$ near $2$; for $0<\alpha<1$ the same law has a real growing mode." + "\n"
    r"\end{remark}")

# Prop checker normalization
rep(r"In the chiral basis that entry is the mass term $\mu\,G_\mu$ of the $1+1$ Dirac propagator." + "\n" + r"\end{remark}",
    r"In the chiral basis that entry is the mass term $\mu\,G_\mu$ of the $1+1$ Dirac propagator." + "\n\n"
    r"On normalization: $\Phi_{2n+1}$ is the corner amplitude times the volume of the set of zigzags, measured in leg durations; it is not a density in $x$. Of the divisor $2ic\mu$, the factor $2c$ is the Jacobian to a density in $x$ and $i\mu$ is one corner. Both $\Phi_{2n+1}$ and $L_\eps$ count one fixed orientation of the first and last steps; the corner-odd paths of both orientations number twice that." + "\n" + r"\end{remark}")

# Thm unitary (3)
rep(r"\item On a solution with both signs of frequency, $\lVert J(t)\rVert^2$ is not constant: the cross terms oscillate at frequency $2\omega$.",
    r"\item On a solution with both signs of frequency, $\lVert J(t)\rVert^2$ is in general not constant: the cross terms $2\,{\rm Re}\int\tr\big(a_+^\dagger a_-\big)e^{2i\omega t}\,d^dk$ oscillate at frequency $2\omega$. It is constant exactly when that integral vanishes for all $t$, for instance when the two signs occupy different matrix entries.")
rep(r"$\lvert\hat J\rvert^2$ contains $2{\rm Re}\big(\bar a_+a_-e^{2i\omega t}\big)$.",
    r"$\tr(\hat J^\dagger\hat J)$ contains $2{\rm Re}\,\tr\big(a_+^\dagger a_-\big)e^{2i\omega t}$. With $\hat J=E_{11}e^{-i\omega t}+E_{22}e^{i\omega t}$ both signs are present and $\hat J\hat J^\dagger=1$ for all $t$.")

# Cor feynman
rep(r"is the limit of the group law of Theorem~\ref{thm:unitary}(1)." + "\n" + r"\end{corollary}",
    r"is the limit of the group law of Theorem~\ref{thm:unitary}(1). The rescaled kernel on the left of the first display is not itself a group at finite $c$, because of its prefactor $mc^2/\hbar\omega$; the group law belongs to $e^{-i(\hat H-mc^2)t/\hbar}$." + "\n" + r"\end{corollary}")

# numerics
rep(r"so that $\mu=1$, unless stated." + "\n",
    r"so that $\mu=1$, unless stated." + "\n\n"
    r"\emph{Independent checks.} The statements were re-tested with separate code: exact and symbolic algebra in sympy ($95$ checks), mpmath at $40$ digits or more ($231$ checks, the one failure being the caption figure corrected in Table~\ref{tab:near2}), a direct simulation on a null lattice in NumPy ($29$ checks), and a Lean~4 file with Mathlib ($38$ theorems, no \texttt{sorry}, no added axioms). Lean proves the algebraic cores: that $h_\alpha$ is constant on $(0,\pi/2)$ exactly at $\alpha=2$ for $\alpha>0$; the dispersion relation of Theorem~\ref{thm:KG} with $\mu\tau_c=1/\sqrt2$ and the sign flip of Remark~\ref{rem:phase}; $(-\mu^2)^n=(i\mu)^{2n}$ and the step from the $n$-th term to the next; an absolute-convergence majorant for the path sum; constancy of the closure on one sign of frequency and the cross terms on two; and the bound $0\le\hbar^2k^2/2m-(\hbar\omega-mc^2)\le\hbar^4k^4/(8m^3c^2)$. Not proved in Lean: Proposition~\ref{prop:symbol} itself, Corollary~\ref{cor:green}, the descent, the Bessel resummations, the lattice limit of the checkerboard, Theorem~\ref{thm:near2}, and the step from one Fourier mode to $L^2$. The simulation of $2J+K_2J=S$ agrees with the spectral Klein--Gordon solution with second-order convergence, and its measured $\mu\tau_c$ extrapolates to $0.7071068$. The script does not run the spacetime form of the law, the term-by-term checkerboard equality, the $I_0$ and $I_1$ sums, the convergence bound, the charge inequality or the bound of Corollary~\ref{cor:feynman}; those are covered by these checks. Next-order terms found by them: the Lorentz defect continues with $\alpha(2-\alpha)(4+6\alpha-3\alpha^2)\varphi^4/360$, and the nonrelativistic dispersion error with $\hbar^5k^6/(16m^5c^4)$; the constant $1/8$ of Theorem~\ref{thm:NR} is sharp." + "\n")
rep(r"At $\alpha=2.01$ and $k=5$ the growth rate, $1.5\times10^{-3}$, is below the counting threshold.",
    r"At $\alpha=2.01$ and $k=5$ the growth rate, $1.56\times10^{-3}$ from the exact root ($1.5\times10^{-3}$ at first order), is below the counting threshold, so the count $0$ there is the threshold's; the true count is $2$. The $\lvert J\rvert$ values that follow are samples of an oscillating signal at single instants, not envelopes, with source centre $10$ and width $1$.")
rep(r"converges to $2c\,i\mu\,G_\mu=i\mu J_0$ at first order in $\eps$.}",
    r"converges to $2c\,i\mu\,G_\mu=i\mu J_0$ at first order in $\eps$. The walk has its first step fixed; started at the origin with a free first step the first row is $3.891\times10^{-2}$.}")
rep(r"\caption{The interior tail of the $3+1$ massive propagator, built from the summed histories.}",
    r"\caption{The interior tail of the $3+1$ massive propagator, built from the summed histories. At $t=8$ the grid reaches $r\le4$ only, $376$ of the $751$ points of the stated window, and the error $2.46\times10^{-6}$ sits at the grid edge, where the difference is one-sided; in the interior it is $8.4\times10^{-8}$.}")
rep(r"\emph{Descent.} Central differences on the $(u,v)$ grid, along the diagonal $u+v=2t$.",
    r"\emph{Descent.} Central differences on the $(u,v)$ grid, one-sided at its edge, along the diagonal $u+v=2t$.")
rep(r"which avoids the light-cone branch points.", r"which avoids the light-cone branch points, and $\lvert{\rm Re}\,\omega\rvert\le12$.")

bad = 0
for a, b in R:
    c = s.count(a)
    if c != 1:
        print("MATCHES %d: %s" % (c, a[:80]))
        bad += 1
    else:
        s = s.replace(a, b)
if bad:
    sys.exit(1)
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("applied %d replacements -> %s" % (len(R), dst))
