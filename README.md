# cuLindblad

A **matrix-free GPU solver for open-quantum-system (Lindblad) and closed-system (Schrödinger)
dynamics** of tunable-coupler superconducting devices. Instead of assembling the `D²×D²`
Liouvillian, it evaluates the right-hand side directly on the density matrix via grouped local
tensor contractions, integrated with fixed-step classic RK4 (PETSc `TS` or a custom lower-memory
stepper), and runs on one or many GPUs.

This solver computes the direct open-system (Lindblad) results in *Gate errors and multiqubit
operations in a tunable-coupler qubit ring* (S. Kunnumbrath and M. Otten, in preparation): the
single-pair iSWAP and CZ channels at 40, 50 and 60 ns on the full four-qubit, four-coupler ring
(4Q4C: 8 three-level modes, Hilbert dim `D = 3⁸ = 6561`) at **T1 = Tφ = 100 µs**, and the 40 ns
gates at T1 = Tφ = 50 µs. The paper uses these direct solves to check a first-order decoherence
estimate that it then applies to its CCZ and C3Z gates. The gate models are built in a separate,
private Julia repository and handed to this solver as a JSON term-list (see **Julia bridge** below).
See **Reproducing the paper's open-system runs** for the exact build and flags.

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
docs/         design notes, the propagator-verification protocol, the Julia↔C++ schema, multinode;
              anchors/ = log + result of one of the paper's open-system runs
profiling/    nsys profiling notes / bottleneck write-ups
propcheck/    tracked golden validation data (regenerated run outputs are gitignored)
archive/      original course-project report + notebook (provenance)
CMakeLists.txt · build.sh · build_fp32.sh · setup_env.sh.example · LICENSE
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
./build_fp32.sh                        # single-precision build -> build_local_fp32/run_sim
                                       # (needs a single-precision complex PETSc arch,
                                       #  PETSC_ARCH=arch-cuda-complex-single)
```

`CULINDBLAD_CUDA_ARCH` is the GPU compute capability with no `sm_` prefix (e.g. `90` = H100,
`120` = RTX PRO 6000 / Blackwell).

## Run — open-system Lindblad (Julia-handoff path)

The Julia exporter writes `model.json` (`system: "open"`; single-site dissipator blocks
`L_rel = b/√T1` and `L_dep = √(2/Tφ)·n` on every mode, with **T1 and Tφ baked into the
magnitudes**; rotating-frame couplings, `frame_omegas`, `result_frame: "rotating"`). Tφ is the
pure-dephasing time, so the paper's T1 = Tφ = 100 µs gives T2 = [(2T1)⁻¹ + Tφ⁻¹]⁻¹ = 66.7 µs. Then:

```bash
source setup_env.sh
./build_local/run_sim -tc_integrator 1 -load_model model.json -results_out results.txt
```

The JSON fixes the physics: dissipators, T1/Tφ, the rotating frame, step count, and the lab
back-transform. Command-line flags only choose how the calculation runs and where results are saved:
`-results_out` (single-process output file), `-dump_tag` (per-rank output names under MPI),
`-tc_checkpoint_every N -tc_checkpoint_dir DIR` (checkpoint/restart), `-tc_integrator`, and the
fast-path flags. `-tc_integrator`: `0` = PETSc TSRK4 (default; QuTiP-validated reference), **`1` =
custom RK4 (recommended: ~46% less GPU memory, agrees with TSRK4 to rounding level, 3.2e-14 in the
rotating frame; required for checkpointing and for accurate fp32 open runs)**, `2` = Taylor.
Optional 4Q4C fast path (additive): `-tc_fused_diagonal 1 -tc_fuse_coupling 1 -tc_fused_diag_prepass 1`.

## Reproducing the paper's open-system runs

The paper's open-system channels were computed with the single-precision build, the custom RK4,
the 4Q4C fast path, checkpointing, and the independent density matrices split across GPUs. The run
recorded in `docs/anchors/` used:

```bash
./build_fp32.sh
mpiexec -n 7 ./build_local_fp32/run_sim -tc_integrator 1 -use_gpu_aware_mpi 0 \
  -tc_fused_diagonal 1 -tc_fuse_coupling 1 -tc_fused_diag_prepass 1 \
  -load_model model_open.json -dump_tag prod_cz_T40_fp32 \
  -tc_checkpoint_every 1000 -tc_checkpoint_dir ckpt
python validation/merge_ranks.py propcheck/prod_cz_T40_fp32 merged.txt 7 256
```

- The model propagates all 256 dressed computational matrix units (16 × 16 inputs) in the full
  `D = 6561` space. RK4 steps are about 3.3 ps (12000 steps for a 40 ns gate).
- In the fp32 build, the open path divides every density matrix with Re Tr ρ > 0.5 by its real trace
  after each step. This cancels single-precision trace drift; the near-zero-trace off-diagonal inputs
  are left alone. It is automatic (no flag) and is not compiled into the fp64 build.
- `docs/anchors/` holds the log and result of the 40 ns CZ at T1 = Tφ = 50 µs: F_closed = 0.99903
  and F_open = 0.99551, matching the paper's Table I and Appendix F.

## Run — multi-GPU state-sharding

The independent density matrices (256 for a 4Q4C open channel) shard across GPUs; node-local GPU
binding is automatic by hostname (no flag). This self-contained route builds the 4Q4C model in C++
(no `model.json`); the `-load_model` route shards the same way. Launch from the repo root:

```bash
mpiexec -n $((NODES*GPUS_PER_NODE)) [-ppn GPUS_PER_NODE] \
  ./build_local/run_sim -dump_tc 1 -tc_computational 1 -tc_num_qubits 4 -tc_num_couplers 4 \
  -dump_cutoff_dim 3 -tc_rotating_frame 1 -tc_integrator 1 -dump_steps 1000 -dump_tag prod4q4c
python validation/merge_ranks.py propcheck/prod4q4c propcheck/prod4q4c_merged.txt \
  $((NODES*GPUS_PER_NODE)) 256
```

Condor wrappers are in `cluster/` (submit from the repo root, e.g.
`condor_submit cluster/mpi_shard.sub`; edit `requirements`/`request_gpus` for your site).

### Closed system (Schrödinger |ψ⟩) multi-GPU

The closed path shards the same way, but over the **2ᴺQ computational kets** instead of the density
columns — each rank evolves a disjoint slice as its batch (no inter-rank comms). It works on both the
self-contained `-dump_tc -tc_schrodinger` route and the `-load_model` route, in fp32 and fp64. The
convenience launcher (parametric, no `model.json` needed) is `cluster/run_closed_shard.sh`:

```bash
# 4Q4C closed, 4 GPUs, fp64, self-verifying (single-process == 4-rank merged, bit-identical):
cluster/run_closed_shard.sh 4 4 4 200 fp64 validate
# 4Q4C closed gate, 8 GPUs, fp32:
cluster/run_closed_shard.sh 8 4 4 12000 fp32 run
```

`condor_submit cluster/closed_shard.sub` runs it on the cluster (edit `arguments`/`request_gpus`).
**Sizing:** ket-sharding uses at most `2ᴺQ` ranks (16 for 4Q4C). Use `-tc_integrator 1` for its
lower memory.

For long runs, add checkpoint/restart on the `-load_model` closed path (blip-proof, auto-resume):
`-tc_checkpoint_every N -tc_checkpoint_dir DIR` (requires `-tc_integrator 1`; per-rank `.ckpt`, strict
header guard). The same flags work for open runs; the paper's open runs used them. The open fp32
checkpoint test builds the fp32 binary, checks that a resumed run matches an uninterrupted one bit for
bit, and checks that a checkpoint from a different step count is refused:
`cluster/ckpt_build_validate.sh <open_model_200_steps.json> <same_model_400_steps.json>`
(HTCondor wrapper: `cluster/ckpt_validate.sub`).

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
- **Closed multi-GPU sharding** (single-process == N-rank merged, bit-identical; fp32 & fp64):
  `validation/validate_closed_multigpu.sh <run_sim> <NQ> <NC> <d> <steps> <nranks> [flags]`.
- **Closed checkpoint/restart** (resume is bit-identical; open or closed model): first
  `source setup_env.sh`, then
  `validation/validate_closed_checkpoint.sh <run_sim> <closed_model.json> <ckpt_every> [nranks]`.
- **Driven regression gate** (golden master + QuTiP): `cluster/validate_gpu.sh <tag> <NQ> <NC> <d>`.
- **Julia-free handoff validator** (proves the C++ half of the bridge against analytic answers):
  `python validation/validate_load_model.py ./build_local/run_sim`.

No optimization is accepted on runtime alone: the golden fixture must still pass, batch
consistency must hold, and the physical result must stay within tolerance.

## Julia bridge

The 4Q4C Julia repository is private (available on request). Its `opensystem/gpu_export.jl`
decomposes the model into **local k-site dense
blocks** (re-walked from `destroy(d)` primitives, never the assembled `6561×6561` operator),
classifies pulse envelopes into shape + params, encodes the T1/Tφ collapse operators as
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

- **State-sharding parallelizes across states, not within one.** Each rank evolves whole density
  matrices / kets; a *single* state vector or density matrix is not split across GPUs. So the useful
  rank count is capped at the number of independent states — `(2ᴺQ)²` density columns (open) or `2ᴺQ`
  kets (closed). A problem needing a single state too large for one GPU would require intra-state
  (domain) decomposition, which is not implemented.
- **Single precision is checked only on a small system.** The paper's open runs use the fp32 build.
  fp32 and fp64 were compared only on the reduced two-qubit, one-coupler cell (`D = 27`), where they
  agree to 8.6e-7 in process fidelity; this does not certify fp32 precision on the full ring.
- The full-stack export to reduced process matrices / effective error channels (for downstream
  resource-estimation / QEC tools) is not yet built.

## License

MIT — see [LICENSE](LICENSE). Built on PETSc, CUDA/cuTENSOR, MPI, and validated against QuTiP.
