import sympy as sp
from sympy import I, sqrt, cos, sin, pi, symbols, conjugate, Abs, Rational, simplify, nsimplify

# Biquaternion = [a0,a1,a2,a3], complex coeffs (central imaginary = sp.I).
# Quaternion units e1,e2,e3: e_a e_b = -delta_ab + eps_abc e_c ; e0=1.
def bmul(p,q):
    p0,p1,p2,p3=p; q0,q1,q2,q3=q
    return [ p0*q0 - p1*q1 - p2*q2 - p3*q3,
             p0*q1 + p1*q0 + p2*q3 - p3*q2,
             p0*q2 - p1*q3 + p2*q0 + p3*q1,
             p0*q3 + p1*q2 - p2*q1 + p3*q0 ]
def dag(q):
    return [ conjugate(q[0]), -conjugate(q[1]), -conjugate(q[2]), -conjugate(q[3]) ]
def s0(q): return q[0]
def Obs(n):   # O_n = -I (n.e)
    return [0, -I*n[0], -I*n[1], -I*n[2]]

print("=== 1. Observable O_n = -I(n.e) is a faithful Hermitian +/-1 element ===")
a=symbols("a",real=True)
Oa=Obs((cos(a),sin(a),0))
print("O_a^dag - O_a =", [sp.simplify(x-y) for x,y in zip(dag(Oa),Oa)], "(expect zeros)")
print("O_a^2         =", [sp.simplify(x) for x in bmul(Oa,Oa)], "(expect [1,0,0,0])")

print()
print("=== 2. Correlation E(a,b) = -<O_a O_b>_0 ===")
b=symbols("b",real=True)
Ob=Obs((cos(b),sin(b),0))
prod=bmul(Oa,Ob)
print("O_a O_b full        =", [sp.simplify(x) for x in prod])
print("scalar (Born <.>_0) =", sp.simplify(s0(prod)), "(expect cos(a-b))")
print("e3 (H vector part)  =", sp.simplify(prod[3]), "(what <.>_0 discards)")
print("E(a,b)              =", sp.simplify(-s0(prod)))

print()
print("=== 3. CHSH at optimal settings a=0, ap=90, b=45, bp=135 deg ===")
def Ev(t1,t2): return -s0(bmul(Obs((cos(t1),sin(t1),0)),Obs((cos(t2),sin(t2),0))))
A,Ap,B,Bp=0,pi/2,pi/4,3*pi/4
S=sp.simplify(Ev(A,B)-Ev(A,Bp)+Ev(Ap,B)+Ev(Ap,Bp))
print("S =", S, "  |S| =", sp.simplify(Abs(S)), "  2*sqrt(2) =", sp.simplify(2*sqrt(2)))
print("matches 2*sqrt(2):", sp.simplify(Abs(S)-2*sqrt(2))==0)

print()
print("=== 4. Control: force an H-vector leak lambda into the Born read ===")
lam=symbols("lam",real=True)
def EvL(t1,t2):
    p=bmul(Obs((cos(t1),sin(t1),0)),Obs((cos(t2),sin(t2),0)))
    return -s0(p) - lam*p[3]   # p[3] = e3 coeff = quaternionic vector part
SL=sp.simplify(EvL(A,B)-EvL(A,Bp)+EvL(Ap,B)+EvL(Ap,Bp))
print("CHSH(lam)          =", SL)
print("CHSH(0)            =", SL.subs(lam,0))
print("dCHSH/dlam at 0    =", sp.diff(SL,lam).subs(lam,0), "(nonzero => test detects H leak)")
print("CHSH(lam=3/10)     =", sp.simplify(SL.subs(lam,Rational(3,10))), "=", float(SL.subs(lam,Rational(3,10))))
