"""
QuTiP sesolve reference for the CLOSED tunable-coupler (Schrodinger) solver. Builds the same
driven Hamiltonian as tunable_coupler_qutip_driven.py (NO collapse operators), evolves each
computational basis KET |a> with sesolve, and compares to the C++ -tc_schrodinger dump.

Usage:  python verify_schrodinger.py <cpp_dump_file> <NQ> <NC> <d> [--rotating]
The C++ run that produced the dump must be the DEFAULT config (matches the params below):
  ./run_sim -dump_tc -tc_schrodinger -tc_num_qubits NQ -tc_num_couplers NC \
            -dump_cutoff_dim d -tc_drives 1 -tc_computational -dump_steps <N> [-tc_rotating_frame 1]
"""
import sys
import numpy as np
import qutip as qt

dump_file = sys.argv[1]
NQ = int(sys.argv[2]); NC = int(sys.argv[3]); d = int(sys.argv[4])
rotating = "--rotating" in sys.argv[5:]
N = NQ + NC
D = d ** N
TF = 40.0
A = 0.5
T_GATE = TF
omega = [5.0 + 0.05 * i for i in range(N)]
alpha = [-0.3] * N
g = 0.02
OPTS = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 10_000_000}

driven_sites = list(range(NQ // 2)) + [NQ + c for c in range(NC // 2)]
driven_set = set(driven_sites)


def comp_kets():
    stride = [d ** (N - 1 - i) for i in range(N)]
    ket = []
    for a in range(1 << NQ):
        h = 0
        for qi in range(NQ):
            h += ((a >> (NQ - 1 - qi)) & 1) * stride[qi]
        ket.append(h)
    return ket


def delta_omega(t):
    return 0.0 if (t <= 0.0 or t >= T_GATE) else -A * np.sin(np.pi * t / T_GATE)


def omega_t(k, t):
    return omega[k] + (delta_omega(t) if k in driven_set else 0.0)


def build_H():
    a = [qt.tensor([qt.destroy(d) if j == i else qt.qeye(d) for j in range(N)]) for i in range(N)]
    H0 = sum(omega[i] * a[i].dag() * a[i]
             + (alpha[i] / 2) * a[i].dag() * a[i].dag() * a[i] * a[i] for i in range(N))

    def op_cpl(l, m):
        return (a[l].dag() * a[m] + a[l] * a[m].dag() - a[l].dag() * a[m].dag() - a[l] * a[m])

    Hc = 0 * H0
    for l in range(N):
        for m in range(l + 1, N):
            Hc = Hc + g * op_cpl(l, m)
    H = [H0 + Hc]
    for k in driven_sites:
        H.append([a[k].dag() * a[k], lambda t, args=None: delta_omega(t)])

    def make_g(i, j):
        wi0, wj0 = omega[i], omega[j]
        return lambda t, args=None: g * (np.sqrt(omega_t(i, t) * omega_t(j, t) / (wi0 * wj0)) - 1.0)
    for i in range(N):
        for j in range(i + 1, N):
            if i in driven_set or j in driven_set:
                H.append([op_cpl(i, j), make_g(i, j)])
    return H


def parse_cpp(path):
    blk = {}; s = None; ent = None
    for line in open(path):
        line = line.strip()
        if line.startswith("STATE"):
            if s is not None:
                blk[s] = np.array(ent, dtype=complex)
            s = int(line.split()[1]); ent = []
        elif line.startswith("PROPAGATOR_DUMP_END"):
            if s is not None:
                blk[s] = np.array(ent, dtype=complex); s = None
        elif line and s is not None and not line[0].isalpha() and line[0] not in "P":
            r, im = line.split(); ent.append(complex(float(r), float(im)))
    return [blk[k] for k in sorted(blk)]


def back_transform(psi_R, tf):
    # rotating -> lab: psi_lab[k] = exp(-i E_k tf) psi_R[k],  E_k = sum_i omega_i occ_i(k)
    occ = np.zeros((D, N), dtype=int)
    stride = [d ** (N - 1 - i) for i in range(N)]
    for k in range(D):
        for i in range(N):
            occ[k, i] = (k // stride[i]) % d
    E = occ @ np.array(omega)
    return np.exp(-1j * E * tf) * psi_R


H = build_H()
kets = comp_kets()
cpp = parse_cpp(dump_file)
worst = 0.0
for pos, kidx in enumerate(kets):
    psi0 = qt.basis([d] * N, [(kidx // (d ** (N - 1 - i))) % d for i in range(N)])
    res = qt.sesolve(H, psi0, [0.0, TF], options=OPTS)
    psi_q = np.asarray(res.states[-1].full(), dtype=complex).ravel()
    psi_c = cpp[pos]
    if rotating:
        psi_c = back_transform(psi_c, TF)
    worst = max(worst, np.abs(psi_c - psi_q).max())
print("Schrodinger C++ vs QuTiP sesolve: worst |psi_C - psi_Q| over all comp states = %.3e" % worst)
