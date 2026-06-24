"""
Julia-free, analytic end-to-end validation of the -load_model handoff path (deserialize term-list
JSON -> matrix-free GPU solver -> dump). Exercises closed (Schrodinger), open (Lindblad), and the
time-dependent pulse path, each against an exact/analytic result. Proves the C++ half of the Julia
handoff is correct independent of the Julia exporter.

Usage:  source setup_env.sh && python validate_load_model.py [path-to-run_sim]
"""
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

RUN_SIM = sys.argv[1] if len(sys.argv) > 1 else "./build_local/run_sim"
PI = np.pi


def run(doc):
    md = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(doc, md); md.close()
    out = md.name + ".out"
    subprocess.run([RUN_SIM, "-tc_integrator", "1", "-load_model", md.name, "-results_out", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    blk = {}; s = None; ent = []
    for line in open(out):
        line = line.strip()
        if line.startswith("STATE"):
            if s is not None: blk[s] = np.array(ent, dtype=complex)
            s = int(line.split()[1]); ent = []
        elif line.startswith("PROPAGATOR_DUMP_END"):
            if s is not None: blk[s] = np.array(ent, dtype=complex); s = None
        elif s is not None and line and line[0] != "P":
            r, i = line.split(); ent.append(complex(float(r), float(i)))
    os.unlink(md.name); os.unlink(out)
    return [blk[k] for k in sorted(blk)]


def mat(real, imag=None):
    return {"real": real, "imag": imag if imag is not None else [0.0] * len(real)}


fails = 0
def check(ok, msg):
    global fails
    print(("  PASS " if ok else "  FAIL ") + msg)
    if not ok: fails += 1


# 1) CLOSED, static diagonal H = pi*n0 on 2 qubits: |10>,|11> get phase -1 at t=1; norms 1.
psis = run({
    "device": {"local_dims": [2, 2], "n_qubits": 2, "n_couplers": 0}, "system": "closed",
    "hamiltonian_terms": [{"name": "pn0", "sites": [0], "row_dim": 2, "col_dim": 2,
                           "matrix": mat([0, 0, 0, PI])}],
    "dissipator_terms": [], "time_dependent_terms": [],
    "evolution": {"tfinal": 1.0, "num_steps": 2000}, "initial_states": {"num_states": 4}})
check(len(psis) == 4 and all(abs(np.linalg.norm(p) - 1) < 1e-9 for p in psis)
      and abs(psis[0][0] - 1) < 1e-6 and abs(psis[2][2] + 1) < 1e-6, "closed static diagonal propagator")

# 2) OPEN, T1 jump a on qubit 0: |00><00| stays trace 1; 16 density columns of length 16.
rhos = run({
    "device": {"local_dims": [2, 2], "n_qubits": 2, "n_couplers": 0}, "system": "open",
    "hamiltonian_terms": [{"name": "pn0", "sites": [0], "row_dim": 2, "col_dim": 2, "matrix": mat([0, 0, 0, 1.0])}],
    "dissipator_terms": [{"name": "T1", "sites": [0], "row_dim": 2, "col_dim": 2, "matrix": mat([0, 1.0, 0, 0])}],
    "time_dependent_terms": [], "evolution": {"tfinal": 0.2, "num_steps": 500}, "initial_states": {"num_states": 4}})
tr0 = sum(rhos[0][i * 4 + i] for i in range(4))
check(len(rhos) == 16 and len(rhos[0]) == 16 and abs(tr0 - 1) < 1e-9, "open Lindblad trace-preserving")

# 3) TIME-DEPENDENT: H(t) = sine(a,b,T)*n0 on 1 qubit -> excited phase exp(-i*Integral f dt).
a, b, T = 2.0, 1.0, 1.0
Phi = (a - b) * 2 * T / PI + b * T
psis = run({
    "device": {"local_dims": [2], "n_qubits": 1, "n_couplers": 0}, "system": "closed",
    "hamiltonian_terms": [], "dissipator_terms": [],
    "time_dependent_terms": [{"name": "drv", "sites": [0], "row_dim": 2, "col_dim": 2, "matrix": mat([0, 0, 0, 1.0]),
                              "coefficient": {"kind": "envelope", "key": "d"}}],
    "envelopes": {"d": {"shape": "sine", "a": a, "b": b, "T": T}},
    "evolution": {"tfinal": T, "num_steps": 8000}, "initial_states": {"num_states": 2}})
check(abs(psis[0][0] - 1) < 1e-6 and abs(psis[1][1] - np.exp(-1j * Phi)) < 1e-7,
      "time-dependent sine drive vs analytic exp(-i.integral f)")

# 4) EXPLICIT closed: inject (|0>+|1>)/sqrt2 into H=pi*n0 -> (|0>-|1>)/sqrt2 at t=1 (|1> phases by -1).
sq = 1.0 / np.sqrt(2.0)
psis = run({
    "device": {"local_dims": [2], "n_qubits": 1, "n_couplers": 0}, "system": "closed",
    "hamiltonian_terms": [{"name": "pn", "sites": [0], "row_dim": 2, "col_dim": 2, "matrix": mat([0, 0, 0, PI])}],
    "dissipator_terms": [], "time_dependent_terms": [],
    "evolution": {"tfinal": 1.0, "num_steps": 2000},
    "initial_states": {"encoding": "explicit", "states": {"real": [[sq, sq]], "imag": [[0.0, 0.0]]}}})
check(len(psis) == 1 and abs(psis[0][0] - sq) < 1e-6 and abs(psis[0][1] + sq) < 1e-6,
      "explicit closed: (|0>+|1>)/sqrt2 under pi*n0 -> (|0>-|1>)/sqrt2")

# 5) EXPLICIT open cross-check preview: same ket as |+><+|, tiny dissipator -> rho ~ |psi_closed><psi_closed|.
# closed final = [sq, -sq] -> |psi><psi| (row-major D^2=4) = [0.5, -0.5, -0.5, 0.5].
rhos = run({
    "device": {"local_dims": [2], "n_qubits": 1, "n_couplers": 0}, "system": "open",
    "hamiltonian_terms": [{"name": "pn", "sites": [0], "row_dim": 2, "col_dim": 2, "matrix": mat([0, 0, 0, PI])}],
    "dissipator_terms": [{"name": "tiny", "sites": [0], "row_dim": 2, "col_dim": 2, "matrix": mat([0, 1e-4, 0, 0])}],
    "time_dependent_terms": [], "evolution": {"tfinal": 1.0, "num_steps": 2000},
    "initial_states": {"encoding": "explicit", "states": {"real": [[sq, sq]], "imag": [[0.0, 0.0]]}}})
r = rhos[0]
exp_rho = np.array([0.5, -0.5, -0.5, 0.5], dtype=complex)
check(len(rhos) == 1 and abs((r[0] + r[3]) - 1) < 1e-3 and np.max(np.abs(r - exp_rho)) < 1e-3,
      "explicit open |+><+| ~ closed |psi><psi| (tiny dissipator, cross-check preview)")

print("validate_load_model: %s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(0 if fails == 0 else 1)
