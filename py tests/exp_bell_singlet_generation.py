import numpy as np
sx=np.array([[0,1],[1,0]],dtype=complex); sy=np.array([[0,-1j],[1j,0]],dtype=complex); sz=np.array([[1,0],[0,-1]],dtype=complex)
def nds(n): return n[0]*sx+n[1]*sy+n[2]*sz
def coh(n):
    th=np.arccos(np.clip(n[2],-1,1)); ph=np.arctan2(n[1],n[0])
    return np.array([np.cos(th/2), np.exp(1j*ph)*np.sin(th/2)],dtype=complex)
def fib(N):
    ga=np.pi*(3-np.sqrt(5)); P=[]
    for i in range(N):
        z=1-2*(i+0.5)/N; r=np.sqrt(max(0,1-z*z)); t=ga*i
        P.append(np.array([r*np.cos(t),r*np.sin(t),z]))
    return P
singlet=np.array([0,1,-1,0],dtype=complex)/np.sqrt(2)
SWAP=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=complex)
PiL=np.outer(singlet,singlet.conj()); PiS=np.eye(4)-PiL
def build(N,m):
    psi=np.zeros(4,dtype=complex)
    for n in fib(N):
        w=np.exp(-1j*m*np.arctan2(n[1],n[0])); psi+=w*np.kron(coh(n),coh(-n))
    nr=np.linalg.norm(psi)
    return psi/nr if nr>1e-12 else psi
def E(rho,a,b): return np.real(np.trace(rho@np.kron(nds(a),nds(b))))
def marg(rho,a): return np.real(np.trace(rho@np.kron(nds(a),np.eye(2))))
def vec(t): return np.array([np.sin(t),0,np.cos(t)])
A,Ap,B,Bp=vec(0),vec(np.pi/2),vec(np.pi/4),vec(3*np.pi/4)
def chsh(rho): return abs(E(rho,A,B)-E(rho,A,Bp)+E(rho,Ap,B)+E(rho,Ap,Bp))
N=4000
print("phase power m : Schmidt s.v. | symW antiW | <Swap> | |<singlet|psi>|^2 | CHSH | margA")
for m in [0,1,2,-1]:
    psi=build(N,m); rho=np.outer(psi,psi.conj())
    s=np.round(np.linalg.svd(psi.reshape(2,2),compute_uv=False),3)
    wS=np.real(psi.conj()@PiS@psi); wL=np.real(psi.conj()@PiL@psi)
    sw=np.real(psi.conj()@SWAP@psi); ov=abs(singlet.conj()@psi)**2
    print(f"m={m:+d}: schmidt={s}  symW={wS:.3f} antiW={wL:.3f}  <Swap>={sw:+.3f}  overlap={ov:.3f}  CHSH={chsh(rho):.4f}  margA={marg(rho,A):+.1e}")
print("2sqrt2 =", round(2*np.sqrt(2),4))
