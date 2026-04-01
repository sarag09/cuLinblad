# Large-Target Profiling Notes

Date: 2026-04-01

Scope:
- Preserve the validated `N=2`, `d=2` checkpoint.
- Use the larger profiling target to identify the next scalable GPU bottleneck.
- Do not change the model, integrator, trusted baseline, or current batched cuTENSOR path.

Build used:
- `build_local/run_sim`
- Reason: `build/run_sim` was an older binary that still executed the deprecated full-state sweep; `build_local/run_sim` matches the current `cpp/src/apps/cli_main.cpp` batch-only `1/2/4` validation flow.

Validation command:

```bash
cd /home/sarag/cuLinblad_final/build_local
./run_sim -use_gpu_aware_mpi 0
```

Validation results:
- `B=1` GPU selected-state time: `10.4963 s`
- `B=2` GPU selected-state time: `10.3281 s`
- `B=4` GPU selected-state time: `9.63831 s`
- `State 0` batch(1) vs batch(2) max abs diff: `0`
- `State 0` batch(1) vs trusted baseline max abs diff: `4.63704e-07`
- `State 0` batch(2) vs trusted baseline max abs diff: `4.63704e-07`
- `State 0 (0,0)`: `(0.618768,0)`

Larger-target timing command:

```bash
cd /home/sarag/cuLinblad_final/build_local
./run_sim -use_gpu_aware_mpi 0 \
  -profile_large_target 1 \
  -profile_num_transmons 6 \
  -profile_cutoff_dim 2 \
  -profile_batched_num_steps 250
```

Larger-target timing results:
- Target: `N=6`, `d=2`, Hilbert dimension `64`, density dimension `4096`, batched TS steps `250`
- `B=1` GPU selected-state time: `7.33971 s`
- `B=2` GPU selected-state time: `9.78859 s`
- `B=4` GPU selected-state time: `14.7008 s`
- Throughput rises from `0.136245` states/s at `B=1` to `0.272095` states/s at `B=4`, but not linearly with batch size.

Nsight Systems command:

```bash
cd /home/sarag/cuLinblad_final/build_local
nsys profile --force-overwrite=true --sample=none --trace=cuda,nvtx,osrt --stats=true \
  -o nsys_large_target \
  ./run_sim -use_gpu_aware_mpi 0 \
  -profile_large_target 1 \
  -profile_num_transmons 6 \
  -profile_cutoff_dim 2 \
  -profile_batched_num_steps 250
```

Nsight Systems artifacts:
- `build_local/nsys_large_target.nsys-rep`
- `build_local/nsys_large_target.sqlite`

Nsight Systems observations:
- NVTX range time is concentrated in PETSc step/function ranges:
  - `:TSStep`: `48.0%`, `74.716 s`, `3750` instances
  - `:TSFunctionEval`: `43.6%`, `67.810 s`, `22506` instances
- CUDA API time is dominated by launch and synchronization overhead:
  - `cudaLaunchKernel`: `30.1%`, `13.463 s`, `1,163,352` calls
  - `cudaLaunchKernelExC_v11060`: `23.5%`, `10.499 s`, `951,592` calls
  - `cudaMemsetAsync`: `22.8%`, `10.215 s`, `973,973` calls
  - `cudaStreamSynchronize`: `10.4%`, `4.653 s`, `22,440` calls
  - `cudaMemcpyAsync`: `7.8%`, `3.497 s`, `162,206` calls
- GPU kernel time is dominated by cuTENSOR contraction kernels:
  - top cuTENSOR kernel: `45.0%`, `9.056 s`, `463,794` launches
  - second cuTENSOR kernel: `42.3%`, `8.516 s`, `403,784` launches
  - third cuTENSOR kernel: `6.1%`, `1.227 s`, `84,014` launches
- Grouped-layout and accumulation kernels are visible but secondary:
  - `vector_accumulate_kernel`: `1.5%`
  - `commutator_combine_kernel`: `1.3%`
  - `dissipator_combine_kernel`: `0.9%`
  - `grouped_to_flat_batched_kernel`: `0.6%`
  - `flat_to_grouped_batched_kernel`: `0.6%`
  - `vector_add_kernel`: `0.6%`

Conclusion:
- The next scalable GPU bottleneck is still cuTENSOR launch count and the surrounding per-step CUDA launch/synchronization overhead, not grouped-layout staging.
- PETSc TS work is visible at the range level, but the trace shows that most of that time is spent inside frequent RHS evaluations that trigger very large numbers of CUDA launches and stream synchronizations.
- The next milestone should target reducing contraction launch count per RHS/step or fusing more work around the existing batched cuTENSOR path before revisiting grouped-layout staging.
