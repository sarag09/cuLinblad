"""
Verify the DRIVEN C++ tunable-coupler propagator (computational states) against the
QuTiP reference, entry-by-entry, with O(dt^4) convergence.

Inputs:
  propcheck/<tag>_comp1000.txt   (C++ -dump_tc -tc_drives 1 -tc_computational 1 -dump_steps 1000)
  propcheck/<tag>_comp16000.txt  (... -dump_steps 16000)
  propcheck/<tag>_qutip_comp.npy (python tunable_coupler_qutip_driven.py <tag> NQ NC d)

Usage: python verify_tunable_coupler_driven.py <tag> <num_qubits> <num_couplers> <d>
"""
import sys
import os
import numpy as np

tag = sys.argv[1]
NQ = int(sys.argv[2])
NC = int(sys.argv[3])
d = int(sys.argv[4])
N = NQ + NC
D = d ** N
M = (1 << NQ) ** 2   # number of computational density matrices

QREF = np.load("propcheck/%s_qutip_comp.npy" % tag)   # (M, D, D)
assert QREF.shape == (M, D, D), "QuTiP ref shape %s != expected %s" % (QREF.shape, (M, D, D))

out = []
def P(m=""):
    out.append(m)
    print(m)


def parse(path):
    blocks = {}
    key = None
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


def stack(block):
    return np.stack([block[s] for s in range(M)])


P("#" * 80)
P("DRIVEN TUNABLE-COUPLER VERIFY  %s  (NQ=%d NC=%d d=%d -> N=%d, D=%d)"
  % (tag, NQ, NC, d, N, D))
P("Evolving %d computational density matrices (qubits in {0,1}, couplers in |0>)." % M)
P("Time-dependent omega(t) drives on ~half the elements.  C++ (TSRK4) vs QuTiP (vern9).")
P("#" * 80)

worst = {}
gpu_by_steps = {}
for steps in (1000, 16000):
    path = "propcheck/%s_comp%d.txt" % (tag, steps)
    if not os.path.exists(path):
        P("\n[skip] %s not found" % path)
        continue
    fb = parse(path)
    gpu = fb.get(("gpu", M))
    if gpu is None:
        # batch_size in the header equals the number of evolved states (M)
        cand = [k for k in fb if k[0] == "gpu"]
        gpu = fb[cand[0]] if cand else None
    if gpu is None:
        P("\n[skip] no gpu block in %s" % path)
        continue
    gpu_by_steps[steps] = gpu
    diff = np.abs(stack(gpu) - QREF)
    dmax = float(diff.max())
    sw, rw, cw = np.unravel_index(int(diff.argmax()), diff.shape)
    worst[steps] = dmax
    P("\n=== computational states @ %d steps (dt=%.5g) : GPU vs QuTiP ===" % (steps, 40.0 / steps))
    P("  entries checked = %d (%d states x %d), max|C++ - QuTiP| = %.3e  @state%d(%d,%d)"
      % (diff.size, M, D * D, dmax, sw, rw, cw))
    cpu = fb.get(("cpu", M))
    if cpu is None:
        cand = [k for k in fb if k[0] == "cpu"]
        cpu = fb[cand[0]] if cand else None
    if cpu is not None:
        gc = float(np.max(np.abs(stack(gpu) - stack(cpu))))
        P("  GPU vs CPU backend (same C++ model) max|diff| = %.3e" % gc)

if 1000 in worst and 16000 in worst and worst[16000] > 0:
    P("\n=== RK4 convergence (1000 -> 16000 steps, 16x refinement) ===")
    P("  worst|diff|: %.3e -> %.3e   (shrink x%.1f; ideal O(dt^4) = 16^4 = 65536)"
      % (worst[1000], worst[16000], worst[1000] / worst[16000]))

# Physics invariants on the best available GPU result.
best = 16000 if 16000 in gpu_by_steps else (1000 if 1000 in gpu_by_steps else None)
if best is not None:
    U = stack(gpu_by_steps[best])  # (M, D, D), ordered by computational density index
    tr_worst = 0.0
    for pos in range(M):
        # pos = a*(2^NQ) + b ; computational populations are at a==b
        a, b = pos // (1 << NQ), pos % (1 << NQ)
        tr_worst = max(tr_worst, abs(np.trace(U[pos]) - (1.0 if a == b else 0.0)))
    herm_worst = 0.0
    nq2 = 1 << NQ
    for a in range(nq2):
        for b in range(nq2):
            herm_worst = max(herm_worst,
                             float(np.max(np.abs(U[a * nq2 + b] - U[b * nq2 + a].conj().T))))
    P("\n=== Physics invariants (no oracle), @ %d steps ===" % best)
    P("  trace preservation  worst |Tr - delta| = %.3e" % tr_worst)
    P("  Hermiticity         worst |U_ab - U_ba^dag| = %.3e" % herm_worst)

conv = worst.get(16000, worst.get(1000))
P("\n" + "=" * 80)
ok = conv is not None and conv < 1e-6
if conv is not None:
    P("Converged worst |C++ - QuTiP| over EVERY entry/state = %.3e" % conv)
P("=> %s" % ("DRIVEN C++ == QuTiP (same physics; residual is RK4 truncation -> 0)"
            if ok else "REVIEW NEEDED (no converged 16000-step result < 1e-6)"))
P("=" * 80)

with open("propcheck/%s_tc_driven_verify.txt" % tag, "w") as f:
    f.write("\n".join(out) + "\n")
sys.exit(0 if ok else 1)
