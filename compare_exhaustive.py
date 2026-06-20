"""
EXHAUSTIVE entry-by-entry comparison: for every system, every GPU batch size, and
every basis state, compare ALL D*D density-matrix entries against the full QuTiP
reference (propcheck/<tag>_qutip_full.npy). Reports the worst entry and the exact
count of complex entries checked, at the 1000-step config AND at the converged
16000-step config.

Usage: python compare_exhaustive.py <tag> <N> <d>
"""
import sys
import numpy as np

tag = sys.argv[1]
N = int(sys.argv[2])
d = int(sys.argv[3])
D = d ** N
DENS = D * D

QREF = np.load("propcheck/%s_qutip_full.npy" % tag)   # (DENS, D, D)


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


out = []
def P(m=""):
    out.append(m)
    print(m)


P("#" * 80)
P("EXHAUSTIVE ENTRY-BY-ENTRY CHECK  %s  (N=%d d=%d, D=%d, density_dim=%d)"
  % (tag, N, d, D, DENS))
P("QuTiP reference: full propagator, all %d basis states, mesolve/vern9" % DENS)
P("#" * 80)

total_entries = 0
worst_overall_conv = 0.0

for steps, sweep_path in [(1000, "propcheck/%s_sweep1000.txt" % tag),
                          (16000, "propcheck/%s_sweep16000.txt" % tag)]:
    sweep = parse(sweep_path)
    bsizes = sorted(bs for (bk, bs) in sweep if bk == "gpu")
    P("\n=== batch sweep @ %d steps : batch sizes %s ===" % (steps, bsizes))
    # cross-batch identity (every entry, every shared state)
    cb_worst = 0.0
    for s in range(max(bsizes)):
        mats = [sweep[("gpu", bs)][s] for bs in bsizes if bs > s]
        for i in range(1, len(mats)):
            cb_worst = max(cb_worst, float(np.max(np.abs(mats[0] - mats[i]))))
    P("  cross-batch identity (C++ vs C++): worst |diff| over all shared states/entries = %.3e"
      % cb_worst)
    # each batch size vs QuTiP, ALL entries of ALL its states
    for bs in bsizes:
        dmax = 0.0
        wi = None
        n_ent = 0
        for s in range(bs):
            diff = np.abs(sweep[("gpu", bs)][s] - QREF[s])
            n_ent += diff.size
            m = float(diff.max())
            if m > dmax:
                dmax = m
                r, c = np.unravel_index(int(diff.argmax()), diff.shape)
                wi = (s, r, c)
        total_entries += n_ent
        P("  batch=%d vs QuTiP: %4d entries checked (%d states x %d), max|diff| = %.3e  @state%d(%d,%d)"
          % (bs, n_ent, bs, D * D, dmax, wi[0], wi[1], wi[2]))
        if steps == 16000:
            worst_overall_conv = max(worst_overall_conv, dmax)

# FULL propagator (all density_dim states) vs QuTiP, every entry
P("\n=== FULL propagator (ALL %d states) vs QuTiP ===" % DENS)
for steps, full_path in [(1000, "propcheck/%s_full1000.txt" % tag),
                         (16000, "propcheck/%s_full16000.txt" % tag)]:
    fb = parse(full_path)
    gpu = fb.get(("gpu", DENS))
    if gpu is None:
        continue
    diff = np.abs(np.stack([gpu[s] for s in range(DENS)]) - QREF)
    dmax = float(diff.max())
    s_w, r_w, c_w = np.unravel_index(int(diff.argmax()), diff.shape)
    n_ent = diff.size
    total_entries += n_ent
    P("  @%5d steps: %6d entries checked (%d states x %d), max|diff| = %.3e  @state%d(%d,%d)"
      % (steps, n_ent, DENS, D * D, dmax, s_w, r_w, c_w))
    if steps == 16000:
        worst_overall_conv = max(worst_overall_conv, dmax)
    # tie batch results to the full propagator: batch-8 states == full first-8
    sweep = parse("propcheck/%s_sweep%d.txt" % (tag, steps))
    if ("gpu", 8) in sweep:
        tie = max(float(np.max(np.abs(sweep[("gpu", 8)][s] - gpu[s]))) for s in range(8))
        P("           (batch=8 block == full-dump first 8 states to %.1e)" % tie)

P("\n" + "=" * 80)
P("TOTAL complex entries compared C++ vs QuTiP for %s : %d" % (tag, total_entries))
P("Converged (16000-step) worst |C++ - QuTiP| over EVERY entry/state/batch = %.3e"
  % worst_overall_conv)
P("=> %s" % ("SAME PROPAGATOR entry-by-entry to QuTiP's accuracy floor (~1e-7)"
            if worst_overall_conv < 1e-6 else "REVIEW NEEDED"))
P("=" * 80)

with open("propcheck/%s_exhaustive.txt" % tag, "w") as f:
    f.write("\n".join(out) + "\n")
sys.exit(0 if worst_overall_conv < 1e-6 else 1)
