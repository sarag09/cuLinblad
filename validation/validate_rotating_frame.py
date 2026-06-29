"""Segment-1 rotating-frame validation (no Julia). A 2-qubit capacitive-coupled model evolved in C++
two ways: (a) LAB frame [diag w0 n0 + w1 n1 + g(H1+H3)]; (b) ROTATING frame [diag removed by the frame,
coupling -> g(cos(Dt)H1 + sin(Dt)H2 + cos(St)H3 + sin(St)H4) via the new 'rf' coefficient, D=w0-w1,
S=w0+w1], result_frame='rotating' so the C++ applies the e^{-iE_k T} back-transform. The two lab-frame
final states must agree to ~RK4-convergence. Validates the rf coefficient + the back-transform end-to-end."""
import json, subprocess, sys
import numpy as np

w0, w1, g, T, N = 1.0, 0.7, 0.15, 8.0, 40000
Df, Sf = w0 - w1, w0 + w1
RS = sys.argv[1] if len(sys.argv) > 1 else "./build_local/run_sim"  # run from repo root, or pass a path

def m4(entries):
    re = [0.0]*16; im = [0.0]*16
    for (r, c), v in entries.items():
        re[r*4+c] = float(np.real(v)); im[r*4+c] = float(np.imag(v))
    return {"real": re, "imag": im}
def m2(entries):
    re = [0.0]*4; im = [0.0]*4
    for (r, c), v in entries.items():
        re[r*2+c] = float(np.real(v)); im[r*2+c] = float(np.imag(v))
    return {"real": re, "imag": im}

# 4 computational basis kets as explicit initial states (frame is identity at t=0, so same both frames)
kets_re = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
init = {"encoding": "explicit", "states": {"real": kets_re, "imag": [[0.0]*4 for _ in range(4)]}}
device = {"local_dims": [2, 2], "n_qubits": 2, "n_couplers": 0}

lab = {"schema_version": 1, "system": "closed", "device": device, "envelopes": {},
  "hamiltonian_terms": [
    {"name": "w0n0", "sites": [0], "row_dim": 2, "col_dim": 2, "matrix": m2({(1,1): w0})},
    {"name": "w1n1", "sites": [1], "row_dim": 2, "col_dim": 2, "matrix": m2({(1,1): w1})},
    {"name": "coup", "sites": [0,1], "row_dim": 4, "col_dim": 4, "matrix": m4({(2,1): g, (1,2): g, (3,0): g, (0,3): g})}],
  "dissipator_terms": [], "time_dependent_terms": [],
  "evolution": {"t0": 0.0, "tfinal": T, "num_steps": N, "rotating_frame": False, "result_frame": "lab"},
  "initial_states": init}

def rf(base_g, freq, phase, sign=1.0):
    return {"kind": "rf", "base": {"kind": "constant", "value": base_g}, "freq": freq, "phase": phase, "sign": sign}
rot = {"schema_version": 1, "system": "closed", "device": device, "envelopes": {},
  "hamiltonian_terms": [], "dissipator_terms": [],
  "time_dependent_terms": [
    {"name": "h1", "sites": [0,1], "row_dim": 4, "col_dim": 4, "matrix": m4({(2,1): 1, (1,2): 1}),    "coefficient": rf(g, Df, "cos")},
    {"name": "h2", "sites": [0,1], "row_dim": 4, "col_dim": 4, "matrix": m4({(2,1): 1j, (1,2): -1j}),  "coefficient": rf(g, Df, "sin")},
    {"name": "h3", "sites": [0,1], "row_dim": 4, "col_dim": 4, "matrix": m4({(3,0): 1, (0,3): 1}),    "coefficient": rf(g, Sf, "cos")},
    {"name": "h4", "sites": [0,1], "row_dim": 4, "col_dim": 4, "matrix": m4({(3,0): 1j, (0,3): -1j}),  "coefficient": rf(g, Sf, "sin")}],
  "evolution": {"t0": 0.0, "tfinal": T, "num_steps": N, "rotating_frame": True,
                "frame_omegas": [w0, w1], "result_frame": "rotating"},
  "initial_states": init}

json.dump(lab, open("/tmp/rf_lab.json", "w")); json.dump(rot, open("/tmp/rf_rot.json", "w"))
for tag, jf in [("lab", "/tmp/rf_lab.json"), ("rot", "/tmp/rf_rot.json")]:
    subprocess.run([RS, "-tc_integrator", "1", "-load_model", jf, "-results_out", f"/tmp/rf_{tag}.out"], check=True)

def parse(p):
    blk = []; ent = None
    for line in open(p):
        line = line.strip()
        if line.startswith("STATE"):
            if ent is not None: blk.append(np.array(ent, dtype=complex))
            ent = []
        elif line.startswith("PROPAGATOR_DUMP_END"):
            if ent is not None: blk.append(np.array(ent, dtype=complex)); ent = None
        elif ent is not None and line and line[0] != "P":
            r, i = line.split(); ent.append(complex(float(r), float(i)))
    return blk

labs, rots = parse("/tmp/rf_lab.out"), parse("/tmp/rf_rot.out")
err = max(np.max(np.abs(labs[s] - rots[s])) for s in range(len(labs)))
nrm = max(abs(np.linalg.norm(labs[s]) - 1.0) for s in range(len(labs)))
print(f"states={len(labs)}  lab |norm-1|={nrm:.2e}  max|lab - rotating(back-transformed)| = {err:.3e}")
print("SEGMENT-1 (rf coeff + back-transform)", "PASS" if err < 1e-7 else "FAIL")
sys.exit(0 if err < 1e-7 else 1)
