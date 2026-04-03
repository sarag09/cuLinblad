# 2026-04-03 local qutrit boundary check

Local GPU:

- `NVIDIA GeForce RTX 4050 Laptop GPU`
- `6141 MiB`
- driver `591.86`

Validation command:

```bash
./run_sim -use_gpu_aware_mpi 0
```

Validation results:

- `N=2`, `d=2`, `B=1`: `7.96313 s`
- `N=2`, `d=2`, `B=2`: `7.87651 s`
- `N=2`, `d=2`, `B=4`: `7.32423 s`
- `State 0 batch(1) vs batch(2) max abs diff = 0`
- `State 0 batch(1) vs trusted baseline max abs diff = 4.63704e-07`
- `State 0 element (0,0) = (0.618768,0)`

Stress commands run:

```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 1,2,4,8,16
```

```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 24,32
```

Stress results for `N=5`, `d=3`, `250` batched TS steps:

| Batch size | Time (s) | States/s |
| --- | ---: | ---: |
| 1 | 22.9914 | 0.0434946 |
| 2 | 29.2139 | 0.0684606 |
| 4 | 51.0691 | 0.0783253 |
| 8 | 98.1150 | 0.0815370 |
| 16 | 203.525 | 0.0786144 |
| 24 | 307.577 | 0.0780292 |
| 32 | 404.995 | 0.0790133 |

Interpretation:

- No failure was observed through `B=32`.
- The first boundary reached locally is practical runtime, not fit-to-memory.
- `B=16` is the last clearly practical local checkpoint at roughly `3.4 min`.
- `B=24` and `B=32` still fit, but they cost roughly `5.1 min` and `6.75 min`, so they are poor iteration points on this GPU.

Representative `nsys` command:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  -o profiling/qutrit_n5d3_b16_250 \
  ./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 16
```

Profiled result:

- `B=16`: `201.973 s`, `0.0792186 states/s`

Key `nsys` observations:

- CUDA API time is dominated by `cudaStreamSynchronize`: `195.62 s` across `1511` calls (`98.2%` of CUDA API time).
- GPU time is dominated by cuTENSOR contraction kernels:
  - `81.20 s` (`40.7%`)
  - `66.87 s` (`33.5%`)
  - `22.47 s` (`11.3%`)
- Those three cuTENSOR kernels account for about `85.5%` of GPU time.
- Internal combine/accumulate/format-conversion kernels are secondary:
  - `vector_accumulate_kernel`: `7.63 s` (`3.8%`)
  - `commutator_combine_kernel`: `5.37 s` (`2.7%`)
  - `dissipator_combine_kernel`: `4.37 s` (`2.2%`)
  - `flat_to_grouped_batched_kernel`: `2.14 s` (`1.1%`)
  - `grouped_to_flat_batched_kernel`: `1.77 s` (`0.9%`)
- GPU memory ops are small relative to kernel time:
  - device-to-device memcpy: `1.70 s`
  - host-to-device memcpy: `0.048 s`
  - memset: `0.044 s`
- PETSc GPU `MAXPY` kernels are present but individually minor, each well under `1 s` total.

Conclusion:

- The next blocker in the near-boundary qutrit regime is not GPU memory footprint.
- The dominant cost is cuTENSOR contraction work, with the host repeatedly blocking on stream completion.
- Raw CUDA launch API overhead is not the main issue by itself; the bigger problem is the serialized compute-and-sync cadence around expensive contraction kernels.

Next optimization direction:

- Do not spend the next iteration on more scratch-memory reduction.
- Focus next on reducing the number or cost of the dominant cuTENSOR contractions per RHS/TS step, or on removing unnecessary synchronization around that contraction-heavy path if measurement shows it is safe.
