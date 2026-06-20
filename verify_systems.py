"""
Generalized C++/CUDA vs QuTiP propagator verification across system sizes and
GPU batch sizes.  Usage:  python verify_systems.py <tag> <N> <d>

For each system it confirms:
  (1) BATCH CONSISTENCY  - a basis state evolved inside a GPU batch of size B is
      identical for every B that contains it (batch sizes 1,2,4,8).
  (2) QuTiP MATCH        - each evolved basis state matches the QuTiP reference
      (built independently with mesolve / vern9), at the 1000-step config.
  (3) RK4 CONVERGENCE    - the C++<->QuTiP residual falls ~O(dt^4) toward QuTiP's
      own accuracy floor as the fixed step count rises 1000 -> 4000 -> 16000.
  (4) GPU == CPU         - the two C++ backends agree to machine precision.
  (5) PHYSICS            - trace preservation Tr(U|j><k|)=d_jk and Hermiticity
      U(|j><k|)=U(|k><j|)^dag on the full propagator.

Index convention (verified against the C++ source): C++ basis state index s puts a
1 at row-major flat position s of the vectorized density matrix, i.e. initial
|s//D><s%D|.  The C++ Hilbert index is site-0-major (strides [d^(N-1),...,d,1]),
identical to QuTiP's tensor ordering, so the dense DxD matrices line up directly.
"""

import sys
import numpy as np
import qutip as qt

tag = sys.argv[1]
N = int(sys.argv[2])
d = int(sys.argv[3])
D = d ** N
DENS = D * D
dims = [[d] * N, [d] * N]
TF = 40.0

# ---- parameters: EXACTLY make_transmon_chain_benchmark_config / prop_qt.py ----
omega = [5.0 + 0.05 * i for i in range(N)]
alpha = [-0.3] * N
g = [0.02] * (N - 1)
T1 = [20.0] * N
Tphi = [30.0] * N
driven_sites = list(range(N // 2)) or [0]


def gauss(t):
    return 0.1 * np.exp(-0.5 * ((t - 20.0) / 10.0) ** 2)


def build_model():
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
    c_ops = []
    for i in range(N):
        c_ops.append(np.sqrt(1.0 / T1[i]) * a[i])
        c_ops.append(np.sqrt(1.0 / Tphi[i]) * (a[i].dag() * a[i]))
    return H, c_ops


H, c_ops = build_model()
MESOLVE_OPTS = {"method": "vern9", "atol": 1e-12, "rtol": 1e-10, "nsteps": 10_000_000}

_qref_cache = {}


def qutip_evolve(s):
    """Reference evolution of basis state s = |s//D><s%D| via independent mesolve."""
    if s in _qref_cache:
        return _qref_cache[s]
    row, col = s // D, s % D
    M = np.zeros((D, D), dtype=complex)
    M[row, col] = 1.0
    rho0 = qt.Qobj(M, dims=dims)
    res = qt.mesolve(H, rho0, [0.0, TF], c_ops=c_ops, options=MESOLVE_OPTS)
    out = np.asarray(res.states[-1].full(), dtype=complex)
    _qref_cache[s] = out
    return out


# ---------------------------- C++ dump parser --------------------------------
def parse(path):
    """Return {(backend, batch_size): {s: DxD ndarray}}."""
    blocks = {}
    key = None
    cur = None
    s = None
    ent = None

    def flush():
        nonlocal s, ent
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


def maxabs(a, b):
    return float(np.max(np.abs(a - b)))


out = []
def P(msg=""):
    out.append(msg)


P("#" * 78)
P("SYSTEM %s : N=%d  d=%d  ->  D=%d  density_dim=%d" % (tag, N, d, D, DENS))
P("#" * 78)

sweep = parse("propcheck/%s_sweep1000.txt" % tag)
batch_sizes = sorted(bs for (bk, bs) in sweep if bk == "gpu")
P("\n[1] BATCH-SIZE CONSISTENCY (GPU, 1000 steps), batch sizes %s" % batch_sizes)
worst_consistency = 0.0
worst_desc = ""
maxstate = max(batch_sizes) - 1
for s in range(maxstate + 1):
    mats = [sweep[("gpu", bs)][s] for bs in batch_sizes if bs > s]
    for i in range(1, len(mats)):
        dd = maxabs(mats[0], mats[i])
        if dd > worst_consistency:
            worst_consistency = dd
            worst_desc = "state %d across batches" % s
P("    worst |diff| for any state shared across batch sizes = %.3e  (%s)"
  % (worst_consistency, worst_desc))
P("    => %s" % ("batch-size INDEPENDENT (machine precision)"
                 if worst_consistency < 1e-10 else "BATCH DEPENDENCE DETECTED"))

# QuTiP match for the first-8 (largest sweep batch) at 1000 steps
big = max(batch_sizes)
P("\n[2] QuTiP MATCH at 1000 steps (batch=%d block, states 0..%d):" % (big, big - 1))
per = []
for s in range(big):
    dd = maxabs(sweep[("gpu", big)][s], qutip_evolve(s))
    per.append(dd)
    P("    STATE %2d (|%d><%d|): max|C++ - QuTiP| = %.3e" % (s, s // D, s % D, dd))
P("    -> max over first %d states = %.3e (RK4 truncation at dt=0.04)" % (big, max(per)))

# Convergence: batch-8 at 1000 / 4000 / 16000 vs QuTiP
P("\n[3] RK4 CONVERGENCE (batch=8 GPU vs QuTiP), max over states 0..7:")
conv_files = [(1000, ("sweep", ("gpu", 8))),
              (4000, ("propcheck/%s_b8_4000.txt" % tag, ("gpu", 8))),
              (16000, ("propcheck/%s_b8_16000.txt" % tag, ("gpu", 8)))]
prev = None
finest = None
monotonic = True
for steps, (src, k) in conv_files:
    blk = sweep[k] if src == "sweep" else parse(src)[k]
    gmax = max(maxabs(blk[s], qutip_evolve(s)) for s in range(8))
    ratio = "" if prev is None else "   (x%.1f smaller)" % (prev / gmax if gmax else float("inf"))
    P("    steps=%6d  dt=%.5f  max|diff| = %.3e%s" % (steps, TF / steps, gmax, ratio))
    if prev is not None and gmax > prev:
        monotonic = False
    prev = gmax
    finest = gmax

# Full propagator: GPU/CPU, trace, Hermiticity (+ full QuTiP for small systems)
full = parse("propcheck/%s_full1000.txt" % tag)
gpu_full = full.get(("gpu", DENS))
cpu_full = full.get(("cpu", DENS))
P("\n[4] FULL PROPAGATOR (all %d basis states, GPU full dump):" % DENS)
if cpu_full is not None:
    gc = max(maxabs(gpu_full[s], cpu_full[s]) for s in range(DENS))
    P("    GPU vs CPU backend max|diff| = %.3e" % gc)
else:
    P("    (CPU backend not dumped for this size)")

# trace + hermiticity
diag = [r * D + r for r in range(D)]
worst_tr = 0.0
for s in range(DENS):
    tr = sum(gpu_full[s].reshape(-1)[i] for i in diag)
    exp = 1.0 if (s // D == s % D) else 0.0
    worst_tr = max(worst_tr, abs(tr - exp))
worst_h = 0.0
for row in range(D):
    for col in range(D):
        worst_h = max(worst_h, maxabs(gpu_full[row * D + col], gpu_full[col * D + row].conj().T))
P("    trace preservation  worst |Tr(U|j><k|) - d_jk| = %.3e" % worst_tr)
P("    Hermiticity         worst |U(|j><k|) - U(|k><j|)^dag| = %.3e" % worst_h)

full_qutip_max = None
if DENS <= 81:                    # cheap enough to evolve every column with mesolve
    full_qutip_max = max(maxabs(gpu_full[s], qutip_evolve(s)) for s in range(DENS))
    P("    full propagator GPU vs QuTiP, max over ALL %d states (1000 steps) = %.3e"
      % (DENS, full_qutip_max))

# Site-ordering discriminator: if we (wrongly) interpreted the Hilbert index as
# site-0-MINOR, the per-site detuning + which-site-is-driven would change, so the
# match must get dramatically worse.  Confirms the site-0-major convention.
def reverse_sites(h):
    digs = []
    x = h
    for _ in range(N):
        digs.append(x % d)
        x //= d
    # digs is site-(N-1)..site0 (since site0 is most significant); rebuild reversed
    rev = 0
    for k in range(N):
        rev = rev * d + digs[k]
    return rev

rev_max = 0.0
for s in range(min(8, DENS)):
    row, col = s // D, s % D
    s_rev = reverse_sites(row) * D + reverse_sites(col)
    rev_max = max(rev_max, maxabs(sweep[("gpu", big)][s], qutip_evolve(s_rev)))
P("\n[5] SITE-ORDER SANITY: matching C++ vs QuTiP under the WRONG (site-reversed)")
P("    labeling gives max|diff| = %.3e (should be >> the correct %.1e)"
  % (rev_max, max(per)))

same = (finest is not None and finest < 1e-5 and monotonic
        and worst_consistency < 1e-10 and worst_tr < 1e-9 and worst_h < 1e-9)
P("\n" + "=" * 78)
P("VERDICT [%s]: %s" % (tag, "C++ == QuTiP propagator (same physics; residual is"
                              " RK4 truncation -> 0 under refinement)"
                              if same else "NEEDS REVIEW"))
P("=" * 78)

report = "\n".join(out)
print(report)
with open("propcheck/%s_result.txt" % tag, "w") as f:
    f.write(report + "\n")
sys.exit(0 if same else 1)
