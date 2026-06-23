"""
Verify the ROTATING-FRAME C++ tunable-coupler propagator against the lab-frame QuTiP
reference, and measure the step-count reduction vs the lab-frame solver.

The rotating frame evolves rho_R = U_R^dag rho U_R with U_R = exp(-i H_R t),
H_R = sum_i omega_i n_i. We recover the lab-frame state by the exact diagonal phase
   rho_lab[k,l] = exp(-i (E_k - E_l) t_f) * rho_R[k,l],   E_k = sum_i omega_i * occ_i(k),
then compare entry-by-entry to the same QuTiP reference used for the lab-frame check.

Because both lab and rotating-frame solvers are fixed-step RK4 (error ~ C/N^4), the
step-reduction gain at matched accuracy is (err_lab@N / err_rot@N)^(1/4).

Usage: python verify_rotating_frame.py <tag> <num_qubits> <num_couplers> <d> [t_final]
       (reads propcheck/<tag>_rf_comp<STEPS>.txt for STEPS in a fixed list)
"""
import sys
import os
import numpy as np

tag = sys.argv[1]
NQ = int(sys.argv[2])
NC = int(sys.argv[3])
d = int(sys.argv[4])
t_final = float(sys.argv[5]) if len(sys.argv) > 5 else 40.0
N = NQ + NC
D = d ** N
M = (1 << NQ) ** 2

QREF = np.load("propcheck/%s_qutip_comp.npy" % tag)  # (M, D, D), lab frame
assert QREF.shape == (M, D, D), "QuTiP ref shape %s != %s" % (QREF.shape, (M, D, D))

# Reference (idle) angular frequencies, matching make_tunable_coupler_benchmark_config:
#   omega[i] = 5.0 + 0.05*i  for i in 0..N-1   (site 0 = qubit 0 = most-significant digit)
omega = np.array([5.0 + 0.05 * i for i in range(N)])

# E_k = sum_i omega_i * occ_i(k), with site-0-major mixed-radix unflatten.
strides = [d ** (N - 1 - i) for i in range(N)]
E = np.zeros(D)
for k in range(D):
    r = k
    for i in range(N):
        occ = r // strides[i]
        r = r % strides[i]
        E[k] += omega[i] * occ
phase = np.exp(-1j * E * t_final)            # (D,)  U_R diagonal
backshift = np.outer(phase, np.conj(phase))  # (D,D) rho_lab = backshift * rho_R


def parse(path):
    blocks = {}
    cur = None
    s = None
    ent = None

    def flush():
        if s is not None and cur is not None:
            cur[s] = np.array(ent, dtype=complex).reshape(D, D)

    for line in open(path):
        line = line.strip()
        if line.startswith("PROPAGATOR_DUMP_BEGIN"):
            t = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            key = (t.get("backend", "?"), int(t.get("batch_size", "0")))
            cur = {}
            s = None
        elif line.startswith("PROPAGATOR_DUMP_END"):
            flush()
            blocks[key] = cur
            cur = None
            s = None
        elif line.startswith("STATE"):
            flush()
            s = int(line.split()[1])
            ent = []
        elif line and s is not None:
            r, im = line.split()
            ent.append(complex(float(r), float(im)))
    return blocks


def stack_lab(block):
    # back-transform each rotating-frame state to the lab frame before comparison
    return np.stack([backshift * block[s] for s in range(M)])


out = []
def P(m=""):
    out.append(m); print(m)

P("#" * 80)
P("ROTATING-FRAME VERIFY  %s  (NQ=%d NC=%d d=%d -> N=%d, D=%d, t_f=%.3g)"
  % (tag, NQ, NC, d, N, D, t_final))
P("rho_R -> rho_lab via exp(-i(E_k-E_l)t_f), then entry-by-entry vs lab-frame QuTiP.")
P("#" * 80)

STEPS = [125, 250, 500, 1000, 2000, 16000]
err = {}
gpu_by_steps = {}
for steps in STEPS:
    path = "propcheck/%s_rf_comp%d.txt" % (tag, steps)
    if not os.path.exists(path):
        continue
    fb = parse(path)
    cand = [k for k in fb if k[0] == "gpu"]
    if not cand:
        P("\n[skip] no gpu block in %s" % path); continue
    gpu = fb[cand[0]]
    gpu_by_steps[steps] = gpu
    lab = stack_lab(gpu)
    diff = np.abs(lab - QREF)
    dmax = float(diff.max())
    err[steps] = dmax
    P("\n=== rotating frame @ %d steps (dt=%.5g) : back-transformed vs QuTiP ===" %
      (steps, t_final / steps))
    P("  max|rho_lab - QuTiP| = %.3e" % dmax)
    ccand = [k for k in fb if k[0] == "cpu"]
    if ccand:
        labc = stack_lab(fb[ccand[0]])
        P("  GPU vs CPU backend (same rotating-frame model) max|diff| = %.3e"
          % float(np.max(np.abs(lab - labc))))

# Invariants on the lab-frame-recovered best result.
best = max(gpu_by_steps) if gpu_by_steps else None
if best is not None:
    U = stack_lab(gpu_by_steps[best])
    nq2 = 1 << NQ
    tr = max(abs(np.trace(U[a * nq2 + b]) - (1.0 if a == b else 0.0))
             for a in range(nq2) for b in range(nq2))
    herm = max(float(np.max(np.abs(U[a * nq2 + b] - U[b * nq2 + a].conj().T)))
               for a in range(nq2) for b in range(nq2))
    P("\n=== invariants on lab-recovered state @ %d steps ===" % best)
    P("  trace preservation worst |Tr - delta| = %.3e" % tr)
    P("  Hermiticity        worst |U_ab - U_ba^dag| = %.3e" % herm)

# Gain vs lab frame: compare to the known lab-frame errors (err ~ C/N^4) if a
# companion lab file exists, else just report.
P("\n" + "=" * 80)
LAB_REF = {1000: 2.002e-03}  # known lab-frame max|C++-QuTiP| @1000 steps (2Q1C driven)
for steps in sorted(err):
    if steps in LAB_REF and err[steps] > 0:
        gain = (LAB_REF[steps] / err[steps]) ** 0.25
        P("step-reduction gain @ %d steps: lab err %.3e / rot err %.3e -> ~%.2fx fewer steps"
          % (steps, LAB_REF[steps], err[steps], gain))
ok = best is not None and min(err.values()) < 1e-3
P("=> %s" % ("ROTATING FRAME OK (matches QuTiP; see gain above)"
            if ok else "REVIEW: no rotating-frame result < 1e-3 vs QuTiP"))
P("=" * 80)

with open("propcheck/%s_rotating_frame_verify.txt" % tag, "w") as f:
    f.write("\n".join(out) + "\n")
sys.exit(0 if ok else 1)
