"""
QuTiP reference for the N-qubit / M-coupler TUNABLE-COUPLER model.

Builds the SAME static model as the C++ build_tunable_coupler_model and evolves
every one of the density_dim basis density matrices |row><col| via independent
mesolve (vern9, tight tol), saving propcheck/<tag>_qutip_full.npy with shape
(density_dim, D, D).

Model (N_total = num_qubits + num_couplers modes, each truncated to d levels):
  bare:      H_l = omega_l a_l^dag a_l + (alpha_l/2) a_l^dag a_l^dag a_l a_l
  ALL-TO-ALL capacitive coupling (uniform g, benchmark):
             H_lm = g (a_l^dag a_m + a_l a_m^dag - a_l^dag a_m^dag - a_l a_m)   for every pair l<m
  dissipators: sqrt(1/T1_l) a_l   and   sqrt(1/Tphi_l) (a_l^dag a_l)

Conventions match qutip_reference.py / the C++ state layout:
  - site-0-major tensor order: tensor([site0, site1, ...])
  - basis-state index s -> row=s//D, col=s%D, M[row,col]=1 (|row><col|)
  - dense D x D states line up entry-by-entry with the C++ row-major dump.

Usage:  python tunable_coupler_qutip.py <tag> <num_qubits> <num_couplers> <d> [nproc]
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
DENS = D * D
dims = [[d] * N, [d] * N]
TF = 40.0

# Must match cli_main.cpp make_tunable_coupler_benchmark_config exactly.
omega = [5.0 + 0.05 * i for i in range(N)]
alpha = [-0.3] * N
g = 0.02
T1 = [20.0] * N
Tphi = [30.0] * N
OPTS = {"method": "vern9", "atol": 1e-12, "rtol": 1e-10, "nsteps": 10_000_000}

_H = None
_C = None


def _build():
    a = [qt.tensor([qt.destroy(d) if j == i else qt.qeye(d) for j in range(N)])
         for i in range(N)]
    H0 = sum(omega[i] * a[i].dag() * a[i]
             + (alpha[i] / 2) * a[i].dag() * a[i].dag() * a[i] * a[i]
             for i in range(N))
    # ALL-TO-ALL full capacitive coupling (every pair l<m), uniform g.
    Hc = 0 * H0
    for l in range(N):
        for m in range(l + 1, N):
            Hc = Hc + g * (a[l].dag() * a[m] + a[l] * a[m].dag()
                           - a[l].dag() * a[m].dag() - a[l] * a[m])
    H = H0 + Hc
    c = []
    for i in range(N):
        c.append(np.sqrt(1.0 / T1[i]) * a[i])
        c.append(np.sqrt(1.0 / Tphi[i]) * (a[i].dag() * a[i]))
    return H, c


def _init():
    global _H, _C
    _H, _C = _build()


def _work(s):
    row, col = s // D, s % D
    M = np.zeros((D, D), dtype=complex)
    M[row, col] = 1.0
    res = qt.mesolve(_H, qt.Qobj(M, dims=dims), [0.0, TF], c_ops=_C, options=OPTS)
    return s, np.asarray(res.states[-1].full(), dtype=complex)


if __name__ == "__main__":
    os.makedirs("propcheck", exist_ok=True)
    print("QuTiP reference: %s  NQ=%d NC=%d d=%d -> N=%d D=%d DENS=%d (%d procs)"
          % (tag, NQ, NC, d, N, D, DENS, NPROC), flush=True)
    arr = np.zeros((DENS, D, D), dtype=complex)
    with Pool(processes=min(NPROC, DENS), initializer=_init) as pool:
        done = 0
        for s, m in pool.imap_unordered(_work, range(DENS), chunksize=1):
            arr[s] = m
            done += 1
            if done % 64 == 0 or done == DENS:
                print("  %s: %d/%d states evolved" % (tag, done, DENS), flush=True)
    np.save("propcheck/%s_qutip_full.npy" % tag, arr)
    print("saved propcheck/%s_qutip_full.npy  shape=%s" % (tag, arr.shape))
