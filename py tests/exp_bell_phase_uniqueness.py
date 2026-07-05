import numpy as np
sx=np.array([[0,1],[1,0]],dtype=complex); sy=np.array([[0,-1j],[1j,0]],dtype=complex); sz=np.array([[1,0],[0,-1]],dtype=complex)
def nds(n): return n[0]*sx+n[1]*sy+n[2]*sz
def coh(n):
    th=np.arccos(np.clip(n[2],-1,1)); ph=np.arctan2(n[1],n[0])
    return np.array([np.cos(th/2), np.exp(1j*ph)*np.sin(th/2)],dtype=complex)
def fib(N):
    ga=np.pi*(3-np.sqrt(5)); P=[]
    for i in range(N):
        z=1-2*(i+0.5)/N; r=np.sqrt(max(0,1-z*z)); t=ga*i; P.append(np.array([r*np.cos(t),r*np.sin(t),z]))
    return P
singlet=np.array([0,1,-1,0],dtype=complex)/np.sqrt(2)
def vec(t): return np.array([np.sin(t),0,np.cos(t)])
A,Ap,B,Bp=vec(0),vec(np.pi/2),vec(np.pi/4),vec(3*np.pi/4)
def E(rho,a,b): return np.real(np.trace(rho@np.kron(nds(a),nds(b))))
def chsh(rho): return abs(E(rho,A,B)-E(rho,A,Bp)+E(rho,Ap,B)+E(rho,Ap,Bp))
def run(N,wfun,label):
    psi=np.zeros(4,dtype=complex)
    for n in fib(N):
        th=np.arccos(np.clip(n[2],-1,1)); ph=np.arctan2(n[1],n[0])
        psi+=wfun(th,ph)*np.kron(coh(n),coh(-n))
    nr=np.linalg.norm(psi); psi=psi/nr if nr>1e-12 else psi
    rho=np.outer(psi,psi.conj())
    ov=abs(singlet.conj()@psi)**2
    print(f"  {label:42s}: singlet overlap={ov:.4f}  CHSH={chsh(rho):.4f}")
N=6000
print("Which phase weight w(theta,phi) yields the singlet? (only uniform e^{-i phi} should hit overlap 1)")
run(N, lambda t,p: np.exp(-1j*p),                         "e^{-i phi}  (uniform azimuthal, m=+1)")
run(N, lambda t,p: np.exp(-1j*(1-np.cos(t))*p),           "e^{-i (1-cos th) phi}  (Berry-weighted winding)")
run(N, lambda t,p: np.exp(-1j*2*np.sin(t/2)**2*p),        "e^{-i 2 sin^2(th/2) phi}  (Berry connection)")
run(N, lambda t,p: np.exp(-1j*p)*np.exp(1j*t),            "e^{-i phi} * e^{i theta}  (theta phase prefactor)")
run(N, lambda t,p: np.exp(-1j*p)*(1+0.4*np.cos(t)),       "e^{-i phi} * (1+0.4 cos th)  (real theta modulation)")
run(N, lambda t,p: np.exp(-1j*p)*np.exp(-0.5j*(1-np.cos(t))*p),"e^{-i phi} * berry-loop  (mixed)")
