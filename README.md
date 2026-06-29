# cuLindblad

A **matrix-free GPU solver for open-quantum-system (Lindblad) and closed-system (Schrödinger)
dynamics** of tunable-coupler superconducting devices. Instead of assembling the `D²×D²`
Liouvillian, it evaluates the right-hand side directly on the density matrix via grouped local
tensor contractions, integrated with a fixed-step RK4 (PETSc `TS`), and runs on one or many GPUs.

This repository is the **open-system engine** behind a paper that walks up a ladder of gate
complexity on a ring of tunable-coupler transmons — iSWAP/CZ on 2Q1C → single 2Q on 4Q4C →
simultaneous 2Q → CCZ (3Q) → ring CCZ / CCCZ (4Q) — and for each rung reports the
**open-system gate fidelity at T1 = T2 = 100 µs**. The gate models are constructed in a separate
Julia repository and handed to this solver as a JSON term-list (see **Julia bridge** below); the
paper anchor is the 4Q4C device (8 modes, `n_basis = 3`, Hilbert dim `D = 3⁸ = 6561`).

## Status

- Open-system propagator validated entry-by-entry against QuTiP `mesolve` (worst entry **1.8e-8**
  at n4d2, via O(dt⁴) convergence to QuTiP's integrator floor — see `docs/PROPAGATOR_VERIFICATION.md`).
- Exact rotating-frame evolution + lab back-transform validated to **8.7e-15**.
- Julia↔C++ handoff (`-load_model`) validated field-by-field and end-to-end.
- Multi-GPU MPI state-sharding of the independent density matrices, with automatic node-local
  GPU binding (`docs/MULTINODE.md`).

## Repository layout

```
cpp/          solver: apps/ (CLI) · backends/{cpu,cuda}/ · core/ · io/ · petsc/ · include/culindblad/
validation/   QuTiP cross-checks + golden-master regression + the Julia-free handoff validator
cluster/      HTCondor submit files + MPI/multi-node launch scripts
docs/         design notes, the propagator-verification protocol, the Julia↔C++ schema, multinode
profiling/    nsys profiling notes / bottleneck write-ups
propcheck/    tracked golden validation data (regenerated run outputs are gitignored)
archive/      original course-project report + notebook (provenance)
CMakeLists.txt · build.sh · setup_env.sh.example · LICENSE
```

## Dependencies

- **PETSc** built from source with **complex scalars + CUDA** (`--with-scalar-type=complex
  --with-cuda`; `PETSC_ARCH=arch-cuda-complex`). PETSc `TS` (`TSRK`/`TSRK4`/`TSADAPTNONE`) is the
  validated fixed-step integrator backbone.
- **CUDA Toolkit + cuTENSOR + cuBLAS** (an NVIDIA HPC SDK works; the build expects CUDA core +
  math_libs + cuTENSOR co-located).
- **MPI** (for one-rank-per-GPU state-sharding).
- **CMake ≥ 3.14**, a **C++17** compiler.
- **QuTiP** (`pip install qutip==5.0.4`) for the validation scripts only.

The compiled binary is **not portable across clusters** — rebuild against your site's
MPI/PETSc/cuTENSOR and GPU arch.

## Build

```bash
cp setup_env.sh.example setup_env.sh   # then edit paths: CUDA, PETSC_DIR/ARCH, CULINDBLAD_CUDA_ARCH
source setup_env.sh
./build.sh                             # configures into build_local/ and makes -> build_local/run_sim
                                       # ./build.sh clean  wipes build_local first
```

`CULINDBLAD_CUDA_ARCH` is the GPU compute capability with no `sm_` prefix (e.g. `90` = H100,
`120` = RTX PRO 6000 / Blackwell).

## Run — open-system Lindblad (Julia-handoff path)

The Julia exporter writes `model.json` (`system: "open"`, single-site `√(rate)·L` dissipator
blocks with **T1 = T2 = 100 µs baked into the magnitudes**, rotating-frame couplings,
`frame_omegas`, `result_frame: "rotating"`). Then:

```bash
source setup_env.sh
./build_local/run_sim -tc_integrator 1 -load_model model.json -results_out results.txt
```

Everything physical — dissipators, T1/T2, the rotating frame, step count, and the lab
back-transform — comes from the JSON; the only CLI flags `-load_model` reads are `-load_model`
and `-results_out`. `-tc_integrator`: `0` = PETSc TSRK4 (QuTiP-validated reference), **`1` = custom
RK4 (recommended, ~46% less GPU memory)**, `2` = Taylor. Optional 4Q4C fast path (additive):
`-tc_fused_diagonal 1 -tc_fuse_coupling 1 -tc_fused_diag_prepass 1`.

## Run — multi-GPU state-sharding

The 256 independent density matrices of the 4Q4C open anchor shard across GPUs; node-local GPU
binding is automatic by hostname (no flag). Launch from the repo root:

```bash
mpiexec -n $((NODES*GPUS_PER_NODE)) [-ppn GPUS_PER_NODE] \
  ./build_local/run_sim -dump_tc 1 -tc_computational 1 -tc_num_qubits 4 -tc_num_couplers 4 \
  -dump_cutoff_dim 3 -tc_rotating_frame 1 -tc_integrator 1 -dump_steps 1000 -dump_tag prod4q4c
python validation/merge_ranks.py propcheck/prod4q4c propcheck/prod4q4c_merged.txt \
  $((NODES*GPUS_PER_NODE)) 256
```

Condor wrappers are in `cluster/` (submit from the repo root, e.g.
`condor_submit cluster/mpi_shard.sub`; edit `requirements`/`request_gpus` for your site).

## Validation

Run from the repo root with the QuTiP venv active.

- **Open propagator vs QuTiP** (headline cross-check), per `docs/PROPAGATOR_VERIFICATION.md`:
  ```bash
  ./build_local/run_sim -dump_propagator 1 -dump_num_transmons 4 -dump_cutoff_dim 2 -dump_steps 16000 \
    > propcheck/n4d2_full16000.txt
  python validation/qutip_reference.py n4d2 4 2
  python validation/compare_exhaustive.py n4d2 4 2     # worst |C++ − QuTiP| over all entries → 1.8e-8
  ```
- **Rotating frame:** `validation/verify_rotating_frame.py` (vs lab QuTiP); self-contained,
  Julia-free, QuTiP-free check: `validation/validate_rotating_frame.py [path-to-run_sim]`.
- **Closed / Schrödinger vs QuTiP `sesolve`:** `validation/verify_schrodinger.py`.
- **Driven regression gate** (golden master + QuTiP): `cluster/validate_gpu.sh <tag> <NQ> <NC> <d>`.
- **Julia-free handoff validator** (proves the C++ half of the bridge against analytic answers):
  `python validation/validate_load_model.py ./build_local/run_sim`.

No optimization is accepted on runtime alone: the golden fixture must still pass, batch
consistency must hold, and the physical result must stay within tolerance.

## Julia bridge

The 4Q4C Julia repo's `opensystem/gpu_export.jl` decomposes the model into **local k-site dense
blocks** (re-walked from `destroy(d)` primitives, never the assembled `6561×6561` operator),
classifies pulse envelopes into shape + params, encodes the T1=T2 collapse operators as
single-site `√(rate)·L` dissipator blocks, and writes `model.json`. It shells out to
`run_sim -load_model`, parses the lab-frame `PROPAGATOR_DUMP`, and applies the dressed projection
Julia-side. The two repos stay separate, communicating only through the JSON term-list. Schema:
`docs/JULIA_CPP_HANDOFF_SCHEMA.md` §2; rotating-frame contract: `docs/ROTATING_FRAME_HANDOFF_DESIGN.md`.

## Why matrix-free

For local dimension `d` and `N` subsystems, `D = dᴺ` and the density matrix has `D²` entries, so
explicit Liouvillian assembly scales as `D⁴`. cuLindblad instead applies the RHS directly,

```
dρ/dt = −i[H, ρ] + Σ_k ( L_k ρ L_k† − ½{L_k†L_k, ρ} )
```

as structured local grouped-tensor contractions, with scratch-buffer reuse and per-layout
executor reuse to keep GPU residency low.

## Limitations

- **`-load_model` is currently single-process.** MPI multi-GPU state-sharding is implemented only
  on the internal `-dump_tc -tc_computational` path. The full Julia-driven 4Q4C open anchor
  therefore cannot yet be sharded across GPUs through `-load_model` alone — adding sharding to the
  `run_loaded_model` path is **planned work**.
- The full-stack export to reduced process matrices / effective error channels (for downstream
  resource-estimation / QEC tools) is not yet built.

## License

MIT — see [LICENSE](LICENSE). Built on PETSc, CUDA/cuTENSOR, MPI, and validated against QuTiP.
