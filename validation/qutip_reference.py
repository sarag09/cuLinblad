"""
Compute the FULL QuTiP reference propagator for a system: evolve every one of the
density_dim basis density matrices |row><col| via independent mesolve (vern9, tight
tol) and save to propcheck/<tag>_qutip_full.npy with shape (density_dim, D, D).
Parallelized across cores so even N=4 d=2 (256 columns) is quick.

Usage:  python qutip_reference.py <tag> <N> <d> [nproc]
"""
import sys
import os
import numpy as np
import qutip as qt
from multiprocessing import Pool

tag = sys.argv[1]
N = int(sys.argv[2])
d = int(sys.argv[3])
NPROC = int(sys.argv[4]) if len(sys.argv) > 4 else min(32, os.cpu_count() or 8)

D = d ** N
DENS = D * D
dims = [[d] * N, [d] * N]
TF = 40.0

omega = [5.0 + 0.05 * i for i in range(N)]
alpha = [-0.3] * N
g = [0.02] * (N - 1)
T1 = [20.0] * N
Tphi = [30.0] * N
driven_sites = list(range(N // 2)) or [0]
OPTS = {"method": "vern9", "atol": 1e-12, "rtol": 1e-10, "nsteps": 10_000_000}


def gauss(t):
    return 0.1 * np.exp(-0.5 * ((t - 20.0) / 10.0) ** 2)


_H = None
_C = None


def _build():
    a = [qt.tensor([qt.destroy(d) if j == i else qt.qeye(d) for j in range(N)])
         for i in range(N)]
    H0 = sum(omega[i] * a[i].dag() * a[i]
             + (alpha[i] / 2) * a[i].dag() * a[i].dag() * a[i] * a[i]
             for i in range(N))
    Hc = sum(g[i] * (a[i].dag() * a[i + 1] + a[i] * a[i + 1].dag())
             for i in range(N - 1)) if N > 1 else 0 * H0
    Htd = []
    for i in driven_sites:
        Htd.append([a[i], (lambda t, args=None, wd=omega[i]: gauss(t) * np.exp(1j * wd * t))])
        Htd.append([a[i].dag(), (lambda t, args=None, wd=omega[i]: np.conj(gauss(t) * np.exp(1j * wd * t)))])
    H = [H0 + Hc] + Htd
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
    arr = np.zeros((DENS, D, D), dtype=complex)
    with Pool(processes=min(NPROC, DENS), initializer=_init) as pool:
        done = 0
        for s, m in pool.imap_unordered(_work, range(DENS), chunksize=1):
            arr[s] = m
            done += 1
            if done % 32 == 0 or done == DENS:
                print("  %s: %d/%d states evolved" % (tag, done, DENS), flush=True)
    np.save("propcheck/%s_qutip_full.npy" % tag, arr)
    print("saved propcheck/%s_qutip_full.npy  shape=%s" % (tag, arr.shape))
