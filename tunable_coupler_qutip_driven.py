"""
QuTiP reference for the DRIVEN N-qubit/M-coupler tunable-coupler model, evolving only
the COMPUTATIONAL-subspace density matrices (qubits in {0,1}, couplers in |0>).

Matches cli_main.cpp make_tunable_coupler_benchmark_config(..., enable_drives=true) +
build_tunable_coupler_model term-for-term:
  static:  H0 = sum_l omega_l a_l^dag a_l + (alpha_l/2) a_l^dag a_l^dag a_l a_l
           Hc = sum_{l<m} g (a_l^dag a_m + a_l a_m^dag - a_l^dag a_m^dag - a_l a_m)   (all pairs)
  drives:  omega_k(t) = omega_k - A*sin(pi t/T)  on driven elements
           (1) diagonal:  Delta_omega(t) * (a_k^dag a_k)              per driven k
           (2) coupling:  [g*sqrt(w_i(t)w_j(t)/(w_i w_j)) - g] * op_ij  per edge incident to a driven element
  dissipators: sqrt(1/T1) a_l,  sqrt(1/Tphi) (a_l^dag a_l)

driven_sites = first num_qubits//2 qubits (sites 0..) + first num_couplers//2 couplers
(sites num_qubits..). For 4Q4C: Q1,Q2,C1,C2; for 2Q1C: Q1.

Saves propcheck/<tag>_qutip_comp.npy with shape ((2^num_qubits)^2, D, D), ordered the
same as the C++ -dump_tc -tc_drives -tc_computational dump.

Usage:  python tunable_coupler_qutip_driven.py <tag> <num_qubits> <num_couplers> <d> [nproc]
"""
import sys
import os
import numpy as np
import qutip as qt
from multiprocessing import Pool

tag = sys.argv[1]
NQ = int(sys.argv[2])
NC = int(sys.argv[3])
d = int(sys.argv[4])
N = NQ + NC
NPROC = int(sys.argv[5]) if len(sys.argv) > 5 else min(32, os.cpu_count() or 8)

D = d ** N
TF = 40.0
A = 0.5            # drive amplitude (must match cli_main.cpp)
T_GATE = TF

omega = [5.0 + 0.05 * i for i in range(N)]
alpha = [-0.3] * N
g = 0.02
T1 = [20.0] * N
Tphi = [30.0] * N
OPTS = {"method": "vern9", "atol": 1e-12, "rtol": 1e-10, "nsteps": 10_000_000}

nq_drive = NQ // 2
nc_drive = NC // 2
driven_sites = list(range(nq_drive)) + [NQ + c for c in range(nc_drive)]
driven_set = set(driven_sites)


# Computational density-matrix indices (same construction as the C++ helper).
def computational_indices():
    stride = [d ** (N - 1 - i) for i in range(N)]
    ket = []
    for a in range(1 << NQ):
        h = 0
        for qi in range(NQ):
            bit = (a >> (NQ - 1 - qi)) & 1
            h += bit * stride[qi]
        ket.append(h)
    return [ka * D + kb for ka in ket for kb in ket]


IDX = computational_indices()


def delta_omega(t):
    if t <= 0.0 or t >= T_GATE:
        return 0.0
    return -A * np.sin(np.pi * t / T_GATE)


def omega_t(k, t):
    return omega[k] + (delta_omega(t) if k in driven_set else 0.0)


_H = None
_C = None


def _build():
    a = [qt.tensor([qt.destroy(d) if j == i else qt.qeye(d) for j in range(N)])
         for i in range(N)]
    H0 = sum(omega[i] * a[i].dag() * a[i]
             + (alpha[i] / 2) * a[i].dag() * a[i].dag() * a[i] * a[i]
             for i in range(N))

    def op_cpl(l, m):
        return (a[l].dag() * a[m] + a[l] * a[m].dag()
                - a[l].dag() * a[m].dag() - a[l] * a[m])

    Hc = 0 * H0
    for l in range(N):
        for m in range(l + 1, N):
            Hc = Hc + g * op_cpl(l, m)

    H = [H0 + Hc]
    # (1) diagonal frequency-modulation drives
    drive_coeff = lambda t, args=None: delta_omega(t)
    for k in driven_sites:
        H.append([a[k].dag() * a[k], drive_coeff])
    # (2) induced coupling modulation on edges incident to a driven element
    def make_g_coeff(i, j):
        wi0, wj0 = omega[i], omega[j]
        def f(t, args=None):
            wi = omega_t(i, t)
            wj = omega_t(j, t)
            return g * (np.sqrt(wi * wj / (wi0 * wj0)) - 1.0)
        return f
    for i in range(N):
        for j in range(i + 1, N):
            if i in driven_set or j in driven_set:
                H.append([op_cpl(i, j), make_g_coeff(i, j)])

    c = []
    for i in range(N):
        c.append(np.sqrt(1.0 / T1[i]) * a[i])
        c.append(np.sqrt(1.0 / Tphi[i]) * (a[i].dag() * a[i]))
    return H, c


def _init():
    global _H, _C
    _H, _C = _build()


def _work(pos):
    s = IDX[pos]
    row, col = s // D, s % D
    M = np.zeros((D, D), dtype=complex)
    M[row, col] = 1.0
    dims = [[d] * N, [d] * N]
    res = qt.mesolve(_H, qt.Qobj(M, dims=dims), [0.0, TF], c_ops=_C, options=OPTS)
    return pos, np.asarray(res.states[-1].full(), dtype=complex)


if __name__ == "__main__":
    os.makedirs("propcheck", exist_ok=True)
    M = len(IDX)
    print("QuTiP DRIVEN ref: %s  NQ=%d NC=%d d=%d -> N=%d D=%d, %d computational states, "
          "driven_sites=%s (%d procs)"
          % (tag, NQ, NC, d, N, D, M, driven_sites, NPROC), flush=True)
    arr = np.zeros((M, D, D), dtype=complex)
    with Pool(processes=min(NPROC, M), initializer=_init) as pool:
        done = 0
        for pos, m in pool.imap_unordered(_work, range(M), chunksize=1):
            arr[pos] = m
            done += 1
            print("  %s: %d/%d computational states evolved" % (tag, done, M), flush=True)
    np.save("propcheck/%s_qutip_comp.npy" % tag, arr)
    print("saved propcheck/%s_qutip_comp.npy  shape=%s" % (tag, arr.shape))
