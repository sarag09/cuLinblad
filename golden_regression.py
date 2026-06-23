"""
GPU optimization-regression gate: compare an optimized C++ propagator dump to a frozen
golden master (the pre-optimization, QuTiP-validated GPU output), entry-by-entry.

Because the solver runs fixed-step TSRK4 (bit-reproducible), an optimization that only
changes HOW the GPU computes (kernels/caching/routing) should reproduce the golden to
bit / rounding level. A physics-level difference (>=1e-6) is a bug.

Usage: python golden_regression.py <current_dump.txt> <golden_dump.txt> <D>
Exit 0 if max|current - golden| < 1e-6 (not physics drift), else 1.
"""
import sys
import numpy as np

cur_path, gold_path, D = sys.argv[1], sys.argv[2], int(sys.argv[3])


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


cur = parse(cur_path)
gold = parse(gold_path)

keys = sorted(set(cur) & set(gold))
if not keys:
    print("golden_regression: NO matching (backend,batch) blocks between dumps")
    sys.exit(1)

worst = 0.0
per_block = []
for k in keys:
    states = set(cur[k]) & set(gold[k])
    bw = 0.0
    for s in states:
        bw = max(bw, float(np.max(np.abs(cur[k][s] - gold[k][s]))))
    per_block.append((k, len(states), bw))
    worst = max(worst, bw)

for (backend, bs), nst, bw in per_block:
    print("  backend=%s batch=%d: %d states, worst|opt - golden| = %.3e" % (backend, bs, nst, bw))

cls = ("BIT-IDENTICAL" if worst == 0.0 else
       "rounding-level" if worst < 1e-11 else
       "small (accumulated rounding)" if worst < 1e-6 else
       "PHYSICS DRIFT")
print("golden_regression: worst over all blocks = %.3e  [%s]" % (worst, cls))
sys.exit(0 if worst < 1e-6 else 1)
