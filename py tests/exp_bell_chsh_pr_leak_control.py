import math
# The Born read kept only the SCALAR part cos(a-b); it DISCARDED the e3 (H-vector) part sin(a-b).
# Control question: if that discarded vector part were instead read (weight lam), can CHSH exceed
# the Tsirelson bound 2sqrt2? Re-optimize the settings for the leaky correlation (my first control
# wrongly froze the no-leak optimum, where the odd term cancels).
def E(t1,t2,lam): return -math.cos(t1-t2) - lam*math.sin(t1-t2)   # scalar + leaked vector part
print("lam :  |CHSH| at leak-adapted optimal settings   vs   2sqrt2*sqrt(1+lam^2)")
for lam in [0.0, 0.3, 0.5, 1.0]:
    phi = math.atan(lam)
    A,Ap = 0.0, math.pi/2
    B,Bp = math.pi/4 - phi, 3*math.pi/4 - phi   # shift b-side by phi to re-optimize
    S = E(A,B,lam)-E(A,Bp,lam)+E(Ap,B,lam)+E(Ap,Bp,lam)
    print(f"{lam:>4}:  |CHSH| = {abs(S):.5f}          predicted = {2*math.sqrt(2)*math.sqrt(1+lam*lam):.5f}")
print()
print("Tsirelson 2sqrt2 =", 2*math.sqrt(2), "   PR-box max = 4")
print("=> lam=0 (Born scalar only): 2sqrt2.  lam=1 (full vector part read): 4 = PR box.")
print("The discarded H-vector part is EXACTLY the super-quantum content; <.>_0 dropping it is")
print("what pins the correlation at Tsirelson instead of letting it run to the PR maximum.")
