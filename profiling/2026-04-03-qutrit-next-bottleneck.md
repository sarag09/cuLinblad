# 2026-04-03 next qutrit bottleneck after aggregation wins

Local GPU:

- `NVIDIA GeForce RTX 4050 Laptop GPU`
- `6141 MiB`

Validation command:

```bash
./run_sim -use_gpu_aware_mpi 0
```

Validation results:

- `N=2`, `d=2`, `B=1`: `6.23252 s`
- `N=2`, `d=2`, `B=2`: `6.29927 s`
- `N=2`, `d=2`, `B=4`: `6.1993 s`
- `State 0 batch(1) vs batch(2) max abs diff = 0`
- `State 0 batch(1) vs trusted baseline max abs diff = 4.63704e-07`
- `State 0 batch(2) vs trusted baseline max abs diff = 4.63704e-07`
- `State 0 element (0,0) = (0.618768,0)`
- Trusted State 0 matrix still matches the required baseline within tolerance.

Stress commands run:

```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 16
```

```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 8,16
```

Clean sequential stress results:

| Batch size | Time (s) | States/s |
| --- | ---: | ---: |
| 8 | 58.5832 | 0.136558 |
| 16 | 124.771 | 0.128235 |
| 16 (`8,16` sweep) | 124.371 | 0.128647 |

Representative `nsys` command:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --stats=true \
  -o profiling/2026-04-03-qutrit-next-bottleneck \
  ./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 16
```

Profiled case:

- `N=5`, `d=3`, `B=16`, `250` batched TS steps
- Profiled timing: `125.042 s`
- Profile outputs:
  - `profiling/2026-04-03-qutrit-next-bottleneck.nsys-rep`
  - `profiling/2026-04-03-qutrit-next-bottleneck.sqlite`

Main `nsys` observations:

- CUDA API time is overwhelmingly `cudaStreamSynchronize`: `119.68 s` across `1511` calls (`97.6%` of CUDA API time).
- NVTX ranges show the wall time is almost entirely inside PETSc stepping and RHS evaluation:
  - `:TSStep`: `124.98 s` (`49.9%`)
  - `:TSFunctionEval`: `124.08 s` (`49.5%`)
- GPU kernel time is still dominated by cuTENSOR contraction kernels:
  - `39.26 s` (`32.8%`)
  - `32.35 s` (`27.0%`)
  - `24.63 s` (`20.6%`)
- Those top three cuTENSOR kernels account for about `80.4%` of total GPU kernel time.
- Internal non-cuTENSOR kernels are secondary:
  - `vector_accumulate_kernel`: `8.51 s` (`7.1%`)
  - `vector_add_kernel`: `3.24 s` (`2.7%`)
  - `commutator_combine_kernel`: `2.91 s` (`2.4%`)
  - `flat_to_grouped_batched_kernel`: `2.36 s` (`2.0%`)
  - `grouped_to_flat_batched_kernel`: `1.93 s` (`1.6%`)
  - `anti_commutator_combine_kernel`: `1.61 s` (`1.3%`)
  - `batched_grouped_diagonal_dissipator_jump_kernel`: `1.22 s` (`1.0%`)
- GPU memory traffic is present but not the main limiter:
  - device-to-device memcpy: `2.70 s` total for `351535 MB`
  - memset: `0.048 s`
  - host-to-device memcpy: `0.025 s`
- PETSc GPU `MAXPY` kernels appear, but each variant is small compared with the cuTENSOR kernels.

Conclusion:

- The next dominant cost in the `N=5`, `d=3` qutrit stress path is still contraction-heavy GPU compute, not memory capacity, not bulk memcpy time, and not standalone PETSc TS kernel overhead.
- The high `cudaStreamSynchronize` share is consistent with the host blocking on long contraction kernels rather than with launch overhead being the primary problem.
- After the current aggregation and diagonal-jump wins, the next optimization direction should target reducing the number or cost of the remaining dominant cuTENSOR contractions in each RHS/TS step.
- If a synchronization change is considered next, it should be justified only insofar as it reduces blocking around the contraction-heavy path without undoing the currently validated cleanup.
