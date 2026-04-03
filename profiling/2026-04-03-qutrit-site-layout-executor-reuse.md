## Scope

Measured whether commit `2a641fa` ("Reduce: reuse executors per site layout to trim GPU residency")
improves practical qutrit memory headroom on the local RTX 4050 laptop GPU without regressing
the validated `N=2`, `d=2` path.

Comparison baseline commit:

- `63ba256` `Fix: keep grouped scratch buffers while restoring executor hot-path reuse`

Both trees were built and run locally with the same commands.

## Validation

Command:

```bash
./run_sim -use_gpu_aware_mpi 0
```

`HEAD` (`2a641fa`) result:

- batch size 1: `7.21278 s`
- batch size 2: `6.94581 s`
- batch size 4: `7.4261 s`
- `batch(1)` vs `batch(2)` max abs diff: `0`
- trusted baseline max abs diff: `4.63704e-07`
- State 0 `(0,0)`: `(0.618768,0)`

Trusted State 0 matrix remained:

```text
(0.618768,0) (-0.011736,-0.00655564) (0.0358747,0.0189598) (0.0114891,0.0038102)
(-0.011736,0.00655564) (0.037414,0) (-0.0355817,-0.0742448) (0.00936175,0.0117862)
(0.0358747,-0.0189598) (-0.0355817,0.0742448) (0.334468,0) (-0.0370064,0.00583516)
(0.0114891,-0.0038102) (0.00936175,-0.0117862) (-0.0370064,-0.00583516) (0.00935067,0)
```

Baseline commit `63ba256` also passed the same validation:

- batch size 1: `7.17731 s`
- batch size 2: `6.67893 s`
- batch size 4: `6.1512 s`
- `batch(1)` vs `batch(2)` max abs diff: `0`
- trusted baseline max abs diff: `4.63704e-07`
- State 0 `(0,0)`: `(0.618768,0)`

## Stress Commands

Required command:

```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 1,2,4,8
```

Additional stressing command:

```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 16
```

For the baseline comparison the same commands were run from the `63ba256` worktree using its
separately built `run_sim`.

## Results

`HEAD` (`2a641fa`) qutrit timings:

- batch size 1: `23.4165 s`
- batch size 2: `29.1981 s`
- batch size 4: `50.3633 s`
- batch size 8: `96.4205 s`
- batch size 16: `200.728 s`

Baseline `63ba256` qutrit timings:

- batch size 1: `22.9559 s`
- batch size 2: `29.0191 s`
- batch size 4: `50.1328 s`
- batch size 8: `96.6083 s`
- batch size 16: `203.718 s`

Measured whole-GPU memory above idle while each run executed:

| Commit | Batch | Peak delta vs idle |
| --- | --- | --- |
| `2a641fa` | 8 | `681 MiB` |
| `63ba256` | 8 | `1017 MiB` |
| `2a641fa` | 16 | `1153 MiB` |
| `63ba256` | 16 | `1793 MiB` |

Observed fit behavior on the practical stress cases:

- `2a641fa` fit batches `1,2,4,8,16`
- `63ba256` fit batches `1,2,4,8,16`
- no larger tested batch fit on `HEAD` that failed on `63ba256`, because both fit through `16`

## Conclusion

Commit `2a641fa` provides real memory headroom on the tested qutrit stress cases:

- about `336 MiB` lower peak GPU residency at batch `8`
- about `640 MiB` lower peak GPU residency at batch `16`

Runtime cost was acceptable:

- no material slowdown in the required `N=5`, `d=3` GPU-only stress cases
- batch `16` was slightly faster on `HEAD` (`200.728 s` vs `203.718 s`)

Decision for this milestone:

- keep the site-layout executor reuse change
- it improves memory headroom materially on practical qutrit cases
- it does so without breaking validation and without unacceptable runtime cost
