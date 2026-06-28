# Multi-node multi-GPU runs (open Lindblad state-sharding)

The 256 computational density matrices are **independent** (same H(t), no interaction), so multi-node is
embarrassingly parallel: each MPI rank evolves a disjoint contiguous shard, binds **one GPU**, writes its
own result file to shared storage, and a post-step merge stitches the global-ordered dump. There is **no
inter-rank communication during evolution** — MPI only provides rank/size + per-node GPU binding.

## What the code does (cli_main.cpp `main()`)
- **GPU binding is automatic and node-local.** Each rank computes its node-local rank by hostname
  (`MPI_Allgather` of hostnames; local_rank = number of same-host ranks with a lower global rank) and binds
  `cudaSetDevice(local_rank % gpus_on_node)`. PETSc's VECCUDA is steered to the same device. This is the
  ONLY `cudaSetDevice` in the process, so it governs every integrator (PETSc TSRK4, custom RK4
  `-tc_integrator 1`, Taylor `2`), closed and open.
- **Sharding uses the global rank/size** — correct across any number of nodes; nothing to configure.
- **Per-rank device log** (printed when size>1) lets you verify distinct GPUs:
  `[rank g/N] host=... local_rank=L gpus/node=G -> intended GPU i, bound GPU d of ndev real (PCI ...)`.
  On a correct multi-node run, ranks on the same host show distinct `local_rank`/PCI; different hosts repeat.

## Build for Argonne A100 (sm_80)
The GPU arch is a build knob (`CULINDBLAD_CUDA_ARCH`, consumed by `build.sh` ->
`-DCMAKE_CUDA_ARCHITECTURES`). On the A100 system, against that cluster's PETSc + cuTENSOR + MPI modules:

    module load <petsc> <cuda> <cutensor> <cray-mpich-or-equivalent>   # cluster-specific
    export CULINDBLAD_CUDA_ARCH=80           # A100 = sm_80   (qis4 Blackwell = 120)
    # point setup_env.sh / build.sh at the ALCF CUDA+cuTENSOR (CUDA_MERGED) and PETSc on that system
    ./build.sh

(The binary is NOT portable across clusters — rebuild on the target against its MPI/PETSc.)

## Launch (one rank per GPU, spanning nodes)
Under the scheduler (PBS/`mpiexec` on Polaris-style A100, or srun), request `NODES` nodes with
`GPUS_PER_NODE` GPUs each and launch `NODES*GPUS_PER_NODE` ranks:

    # inside the job allocation:
    mpiexec -n $((NODES*GPUS_PER_NODE)) [-ppn GPUS_PER_NODE] \
        ./build_local/run_sim \
        -dump_tc 1 -tc_computational 1 -tc_num_qubits 4 -tc_num_couplers 4 \
        -tc_rotating_frame 1 -tc_integrator 1 \
        -tc_fused_diagonal 1 -tc_fuse_coupling 1 -tc_fused_diag_prepass 1 \
        -dump_steps 12000 -dump_tag prod4q4c
    # each rank -> propcheck/prod4q4c_rank<g>.txt on the SHARED filesystem
    python merge_ranks.py propcheck/prod4q4c propcheck/prod4q4c_merged.txt \
        $((NODES*GPUS_PER_NODE)) 256        # 256 = (2^4)^2 computational columns

Requirements: (1) `propcheck/` on storage shared by all nodes; (2) `request_gpus`/`--gpus` ==
ranks-per-node so `gpus_on_node` matches; (3) use `-tc_integrator 1` (custom RK4, ~46% less GPU memory than
PETSc TSRK4). Verify the per-rank log shows distinct `(host, PCI)` before trusting a long run.

## Develop / validate on a single GPU
`run_multinode_test.sh` builds and runs: a 2-rank shared-GPU shard (bit-identical vs single-process) and a
**simulated multi-node** run via the test knobs:

    TC_FAKE_RANKS_PER_NODE=2 TC_FAKE_GPUS_PER_NODE=2  -> 4 ranks look like 2 nodes x 2 GPUs;
    the log shows the intended GPU split (0,1,0,1) while all bind the one real GPU; merge stays bit-identical.

Validated on qis4 (1-GPU interactive): build + both shards == single-process bit-identical (0.000e+00),
local ranks 0,1 (real) and 0,1,0,1 (faked 2-node). The real 2-physical-GPU distinct binding is confirmed by
the per-rank PCI log on the first true 2-GPU run (Argonne, or a local `request_gpus=2` batch job).

## Local gotcha (UW HEP / qis4)
BATCH GPU jobs on qis4 currently die in Condor's GPU wrapper before the executable runs
(`condor_gpu_job_wrapper` -> empty `taskset` list from a missing `/sys/.../local_cpulist`, driver 13.0).
Use an **interactive** qis4 session for local GPU runs until that node issue is fixed.
