"""
Verify the C++ tunable-coupler propagator against the QuTiP reference, entry-by-entry.

The C++ solver uses fixed-step RK4 (TSRK4 + TSADAPTNONE); QuTiP uses adaptive vern9.
They cannot agree bit-for-bit at finite step count, so the proof that they compute the
SAME open-system propagator is that the residual falls as O(dt^4) under step refinement
toward QuTiP's accuracy floor (~1e-8). We compare the full propagator (all density_dim
basis states) at 1000 and 16000 steps, plus oracle-free physics invariants.

Inputs (generate first):
  propcheck/<tag>_full1000.txt   (C++ -dump_tc ... -dump_steps 1000;  gpu [+ cpu] blocks)
  propcheck/<tag>_full16000.txt  (C++ -dump_tc ... -dump_steps 16000)
  propcheck/<tag>_qutip_full.npy (python tunable_coupler_qutip.py <tag> NQ NC d)

Usage: python verify_tunable_coupler.py <tag> <num_qubits> <num_couplers> <d>
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
DENS = D * D

QREF = np.load("propcheck/%s_qutip_full.npy" % tag)   # (DENS, D, D)

out = []
def P(m=""):
    out.append(m)
    print(m)


def parse(path):
    """Parse PROPAGATOR_DUMP blocks -> {(backend, batch_size): {state: DxD ndarray}}."""
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
    return np.stack([block[s] for s in range(DENS)])


P("#" * 80)
P("TUNABLE-COUPLER VERIFY  %s  (NQ=%d NC=%d d=%d -> N=%d, D=%d, density_dim=%d)"
  % (tag, NQ, NC, d, N, D, DENS))
P("Model: all-to-all full capacitive coupling, static.  C++ (TSRK4) vs QuTiP (vern9).")
P("#" * 80)

worst = {}
gpu_by_steps = {}
for steps in (1000, 16000):
    path = "propcheck/%s_full%d.txt" % (tag, steps)
    if not os.path.exists(path):
        P("\n[skip] %s not found" % path)
        continue
    fb = parse(path)
    gpu = fb.get(("gpu", DENS))
    if gpu is None:
        P("\n[skip] no gpu full block in %s" % path)
        continue
    gpu_by_steps[steps] = gpu
    diff = np.abs(stack(gpu) - QREF)
    dmax = float(diff.max())
    sw, rw, cw = np.unravel_index(int(diff.argmax()), diff.shape)
    worst[steps] = dmax
    P("\n=== FULL propagator @ %d steps (dt=%.5g) : GPU vs QuTiP ===" % (steps, 40.0 / steps))
    P("  entries checked = %d (%d states x %d), max|C++ - QuTiP| = %.3e  @state%d(%d,%d)"
      % (diff.size, DENS, D * D, dmax, sw, rw, cw))
    # GPU vs CPU cross-check (same backend, no oracle)
    cpu = fb.get(("cpu", DENS))
    if cpu is not None:
        gc = float(np.max(np.abs(stack(gpu) - stack(cpu))))
        P("  GPU vs CPU backend (same C++ model) max|diff| = %.3e" % gc)

# O(dt^4) convergence
if 1000 in worst and 16000 in worst and worst[16000] > 0:
    ratio = worst[1000] / worst[16000]
    P("\n=== RK4 convergence (1000 -> 16000 steps, 16x refinement) ===")
    P("  worst|diff|: %.3e -> %.3e   (shrink x%.1f; ideal O(dt^4) = 16^4 = 65536)"
      % (worst[1000], worst[16000], ratio))

# Oracle-free physics invariants on the converged (or best available) GPU propagator.
best_steps = 16000 if 16000 in gpu_by_steps else (1000 if 1000 in gpu_by_steps else None)
if best_steps is not None:
    gpu = gpu_by_steps[best_steps]
    U = stack(gpu)  # (DENS, D, D); U[s] = evolved |row><col|, s = row*D+col
    # Trace preservation: Tr(U[|j><k|]) == delta_jk
    tr_worst = 0.0
    for s in range(DENS):
        row, col = s // D, s % D
        expect = 1.0 if row == col else 0.0
        tr_worst = max(tr_worst, abs(np.trace(U[s]) - expect))
    # Hermiticity: U(|j><k|) == U(|k><j|)^dag
    herm_worst = 0.0
    for row in range(D):
        for col in range(D):
            s = row * D + col
            sp = col * D + row
            herm_worst = max(herm_worst, float(np.max(np.abs(U[s] - U[sp].conj().T))))
    P("\n=== Physics invariants (no oracle), @ %d steps ===" % best_steps)
    P("  trace preservation  worst |Tr(U|j><k|) - delta_jk| = %.3e" % tr_worst)
    P("  Hermiticity         worst |U(|j><k|) - U(|k><j|)^dag| = %.3e" % herm_worst)

# Verdict
conv = worst.get(16000, worst.get(1000))
P("\n" + "=" * 80)
ok = conv is not None and conv < 1e-6
if conv is not None:
    P("Converged worst |C++ - QuTiP| over EVERY entry/state = %.3e" % conv)
P("=> %s" % ("C++ == QuTiP propagator (same physics; residual is RK4 truncation -> 0)"
            if ok else "REVIEW NEEDED (no converged 16000-step result < 1e-6)"))
P("=" * 80)

with open("propcheck/%s_tc_verify.txt" % tag, "w") as f:
    f.write("\n".join(out) + "\n")
sys.exit(0 if ok else 1)
