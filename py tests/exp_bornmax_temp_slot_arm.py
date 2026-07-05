import numpy as np

np.random.seed(12345)

def make_psi(K):
    # random complex, normalized so sum |psi|^2 = 1
    re = np.random.randn(K)
    im = np.random.randn(K)
    psi = re + 1j*im
    p = np.abs(psi)**2
    p = p / p.sum()
    # rebuild |psi| consistent with normalized prob
    amp = np.sqrt(p)
    return amp, p  # amp = |psi|, p = |psi|^2

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

# ---- Framework closure identities (verbatim) ----
# Detector cost at closure:      H_det = -ln|psi|^2
# Variable-order entropy at nu=1: H^nu = -ln nu = 0
# Thermal channel:               theta_R = -H_det / tau
#
# HYPOTHESIS H-B: entropy in COST slot, detector in TEMPERATURE slot
#   P_i ~ exp( -H^nu_i / H_det_i )
# Born target used ONLY as comparison; |psi| enters weight only via H_det = -ln|psi|^2.

print("="*70)
print("HYPOTHESIS H-B: P_i ~ exp( -H^nu_i / H_det_i )")
print("  cost slot     = H^nu  (variable-order entropy)")
print("  temperature   = H_det = -ln|psi|^2")
print("="*70)

for K in [5, 8, 12, 20]:
    amp, p = make_psi(K)
    H_det = -np.log(p)            # detector cost, > 0 since p<1
    print(f"\n----- K={K} -----")
    print("p (=|psi|^2)     :", np.round(p, 5))
    print("H_det=-ln p      :", np.round(H_det, 5))

    # TASK 1: fixed point nu=1 => H^nu = 0 for all i
    Hnu_fp = np.zeros(K)
    num = -Hnu_fp / H_det        # = 0 / H_det = 0
    P_fp = softmax(num)          # exp(0)=1 each => uniform
    print("\n[Task1] nu=1 fixed point: H^nu=0 for all i")
    print("  P_fp           :", np.round(P_fp, 5), " (uniform = 1/K)")
    print("  max|P_fp - p|  :", np.max(np.abs(P_fp - p)))
    print("  is uniform?    :", np.allclose(P_fp, 1.0/K))

    # TASK 2: off fixed point. Try non-uniform per-state entropy profiles.
    # Admissible variable-order entropy: H^nu_i = -ln(nu_i), nu_i in (0,1].
    # So H^nu_i >= 0 always (order-based, entropy functional).
    # Try to force normalized exp(-H^nu_i / H_det_i) = p_i.
    # Solve for the numerator that WOULD reproduce Born:
    #   P_i = p_i  =>  exp(num_i)/Z = p_i  => num_i = ln p_i + ln Z
    # with num_i = -H^nu_i / H_det_i.
    # Choose ln Z as free additive const (softmax shift-invariant); the shape
    # that matters is num_i - num_j = ln p_i - ln p_j.
    #
    # Required numerator (up to additive const): num_i = ln p_i = -H_det_i.
    # Then -H^nu_i / H_det_i = -H_det_i  =>  H^nu_i = H_det_i^2.
    required_num = np.log(p)                 # = -H_det
    required_Hnu = -required_num * H_det      # = H_det * H_det = H_det^2
    print("\n[Task2/3] To match Born, required numerator (-H^nu/H_det) must equal ln p:")
    print("  required num_i = ln p_i        :", np.round(required_num, 5))
    print("  => required H^nu_i = H_det_i^2  :", np.round(required_Hnu, 5))
    print("  compare H_det^2                 :", np.round(H_det**2, 5))
    print("  H^nu == H_det^2 forced?         :", np.allclose(required_Hnu, H_det**2))

    # Is required_Hnu an admissible entropy H^nu = -ln(nu), nu in (0,1]?
    # H^nu >= 0 requires nu<=1; here required_Hnu = H_det^2 >= 0, so nu_i=exp(-H_det^2).
    nu_needed = np.exp(-required_Hnu)
    print("  implied nu_i = exp(-H_det^2)    :", np.round(nu_needed, 6),
          " all in (0,1]:", np.all((nu_needed>0)&(nu_needed<=1)))
    # BUT: is H^nu = H_det^2 the ENTROPY, or a hand-tuned fn of |psi|?
    # H_det^2 = (ln|psi|^2)^2  -- a function of |psi| itself, NOT of an
    # independent order nu. To get it we had to SET nu_i = exp(-(ln p_i)^2),
    # i.e. tune the "order" to encode |psi|. That is cheating: the entropy
    # functional is order-based (-ln nu), it is not supposed to know |psi|.

    # Verify the cheat reproduces Born exactly:
    num_cheat = -required_Hnu / H_det        # = -H_det = ln p
    P_cheat = softmax(num_cheat)
    err_cheat = np.max(np.abs(P_cheat - p))
    print("  P with cheated numerator        :", np.round(P_cheat,5))
    print("  max|P_cheat - p| (unfaithful)   :", err_cheat)

    # FAITHFUL attempts: genuine entropy profiles H^nu_i = -ln(nu_i),
    # nu_i chosen WITHOUT reference to |psi|. Try several and report best error.
    best_err = np.inf
    best_desc = None
    trials = {}
    # (a) fixed point already: H^nu=0 -> uniform (done)
    trials["nu=1 (fixed pt, H^nu=0)"] = np.zeros(K)
    # (b) uniform order nu=c<1 -> H^nu = const -> num_i = -c/H_det (varies!)
    for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
        Hnu = np.full(K, -np.log(c))
        trials[f"uniform nu={c}"] = Hnu
    # (c) random admissible orders (no |psi| info)
    for t in range(2000):
        nu = np.random.uniform(1e-3, 1.0, size=K)
        trials[f"rand#{t}"] = -np.log(nu)
    # (d) monotone reparam / positive rescales of temperature
    rescales = [0.25, 0.5, 1.0, 2.0, 4.0]
    for Hnu_name, Hnu in list(trials.items()):
        for s in rescales:
            temp = s * H_det                 # positive scalar rescale of temp
            num = -Hnu / temp
            P = softmax(num)
            err = np.max(np.abs(P - p))
            if err < best_err:
                best_err = err
                best_desc = f"{Hnu_name}, temp_scale={s}"
    print(f"\n  BEST FAITHFUL (order-based H^nu, no |psi| in numerator):")
    print(f"    best max|P - p| = {best_err:.6f}  via [{best_desc}]")

# Summary numbers on canonical K=8
amp, p = make_psi(8)
H_det = -np.log(p)
P_fp = softmax(np.zeros(8))
print("\n" + "="*70)
print("SUMMARY (K=8):")
print("  fixed-point P (nu=1) :", np.round(P_fp,5), "-> uniform 1/K")
print("  Born target p        :", np.round(p,5))
print("  max|P_fp - p|        :", float(np.max(np.abs(P_fp - p))))
print("  required H^nu to match Born = H_det^2 = (ln|psi|^2)^2 (a fn of |psi|, NOT entropy)")
print("  => only matches if UNFAITHFUL")
print("="*70)
