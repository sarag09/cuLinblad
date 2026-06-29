import numpy as np
import qutip as qt
import time
 
def simulate_transmons(
    N,                  # number of transmons
    d,                  # Hilbert-space cutoff per transmon
    omega,              # list of transmon frequencies (GHz)
    alpha,              # list of anharmonicities (GHz)
    g,                  # list of nearest-neighbor couplings (GHz)
    T1,                 # list of relaxation times (us)
    Tphi,               # list of pure-dephasing times (us)
    Omega_funcs,        # list of drive envelope functions Omega_i(t)
    w_drive,            # list of drive carrier freqs (GHz)
    tlist,              # time grid; propagator only uses tlist[-1]
    driven_sites        # list of qubit indices that are actually driven
):
    # 1) Build local annihilation operators
    a_ops = [
        qt.tensor([qt.destroy(d) if j == i else qt.qeye(d) for j in range(N)])
        for i in range(N)
    ]
 
    # 2) Bare transmon + anharmonicity
    H0 = sum(
        omega[i] * a_ops[i].dag() * a_ops[i]
        + (alpha[i] / 2) * a_ops[i].dag() * a_ops[i].dag() * a_ops[i] * a_ops[i]
        for i in range(N)
    )
 
    # 3) Nearest-neighbor coupling
    Hc = sum(
        g[i] * (a_ops[i].dag() * a_ops[i + 1] + a_ops[i] * a_ops[i + 1].dag())
        for i in range(N - 1)
    )
 
    # 4) Time-dependent drive terms
    def make_coeff(Omega, wd):
        def coeff(t, args=None):
            return Omega(t) * np.exp(1j * wd * t)
        return coeff
 
    def make_coeff_conj(Omega, wd):
        def coeff(t, args=None):
            return np.conj(Omega(t) * np.exp(1j * wd * t))
        return coeff
 
    H_td = []
    for i in driven_sites:
        H_td.append([a_ops[i], make_coeff(Omega_funcs[i], w_drive[i])])
        H_td.append([a_ops[i].dag(), make_coeff_conj(Omega_funcs[i], w_drive[i])])
 
    # 5) Combine Hamiltonian
    H = [H0 + Hc] + H_td
 
    # 6) Collapse operators (relaxation + pure dephasing)
    c_ops = []
    for i in range(N):
        c_ops.append(np.sqrt(1.0 / T1[i]) * a_ops[i])
        n_i = a_ops[i].dag() * a_ops[i]
        c_ops.append(np.sqrt(1.0 / Tphi[i]) * n_i)
 
    # 7) Compute time evolution propagator at final time
    U_final = qt.propagator(
        H,
        tlist[-1],
        c_ops,
        options={
            "nsteps": int(1e6),
            "method": "vern9",
            "atol": 1e-9,
            "rtol": 1e-7
        }
    )
 
    return U_final, a_ops
 
 
if __name__ == "__main__":
    # --- User parameters ---
    N = 2
    d = 2
 
    omega = [5.0 + 0.05 * i for i in range(N)]
    alpha = [-0.3] * N
    g = [0.02] * (N - 1)
    T1 = [20.0] * N
    Tphi = [30.0] * N
 
    # Drive only half the qubits
    driven_sites = list(range(N // 2))
    if len(driven_sites) == 0:
        driven_sites = [0]
 
    def gauss(t, t0=20.0, sigma=10.0, amp=0.1):
        return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
 
    Omega_funcs = [lambda t, f=gauss: f(t) for _ in range(N)]
    w_drive = omega[:]
    tlist = np.linspace(0.0, 40.0, 501)
 
    start = time.time()
    U_final, a_ops = simulate_transmons(
        N, d, omega, alpha, g,
        T1, Tphi,
        Omega_funcs, w_drive,
        tlist,
        driven_sites
    )
    end = time.time()
 
    print(f"Time taken: {end - start:.6f} seconds")
    print(f"N: {N}")
    print(f"Driven sites: {driven_sites}")
    print(f"Final propagator: {U_final}")