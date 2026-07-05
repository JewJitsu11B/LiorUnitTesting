import sympy as sp
from sympy import I, symbols, cos, sin, conjugate

def bmul(p,q):
    p0,p1,p2,p3=p; q0,q1,q2,q3=q
    return [ p0*q0-p1*q1-p2*q2-p3*q3, p0*q1+p1*q0+p2*q3-p3*q2,
             p0*q2-p1*q3+p2*q0+p3*q1, p0*q3+p1*q2-p2*q1+p3*q0 ]
def dag(q): return [conjugate(q[0]),-conjugate(q[1]),-conjugate(q[2]),-conjugate(q[3])]
def s0(q): return q[0]
def Obs(n): return [0,-I*n[0],-I*n[1],-I*n[2]]

print("=== CORE (fully general): conjugation transport preserves the Born scalar ===")
r0,r1,r2,r3=symbols("r0 r1 r2 r3",real=True)
x0,x1,x2,x3=symbols("x0 x1 x2 x3")
R=[r0,r1,r2,r3]; Rd=dag(R); X=[x0,x1,x2,x3]
sc=sp.expand(s0(bmul(bmul(R,X),Rd)))
print("scalar of R X R^dag =", sp.factor(sc), " = |R|^2 * x0 ; on unit sphere |R|=1 => x0")
print("  => for ANY unitary R and ANY X, <R X R^dag>_0 = <X>_0. No vector->scalar leak. lambda=0.")

print()
print("=== Conjugation always maps a valid observable to a valid observable ===")
b=symbols("b",real=True); Ob=Obs((cos(b),sin(b),0))
Obc=bmul(bmul(R,Ob),Rd)
print("  R O_b R^dag hermitian? dag-diff =", [sp.simplify(x-y) for x,y in zip(dag(Obc),Obc)], "(all zero => stays an observable)")

print()
print("=== CONTROL: a non-conjugation map (left-mult by rotation about e1) genuinely leaks ===")
chi=symbols("chi",real=True)
Rc=[cos(chi/2),sin(chi/2),0,0]
ObL=bmul(Rc,Ob)
print("  left-mult scalar(e0) part =", sp.simplify(s0(ObL)), "(nonzero => leaked vector into scalar)")
print("  left-mult hermitian? dag-diff =", [sp.simplify(x-y) for x,y in zip(dag(ObL),ObL)], "(nonzero => NOT a valid observable)")
a=symbols("a",real=True); Oa=Obs((cos(a),sin(a),0))
print("  E via left-mult =", sp.simplify(-s0(bmul(Oa,ObL))), "(not a clean dot product => lambda != 0)")
