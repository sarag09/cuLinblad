# cuLindblad ↔ QuTiP Propagator Verification

This document records an exhaustive verification that the cuLindblad C++/CUDA
solver and the QuTiP reference (`prop_qt.py`) compute the **same Lindblad
propagator** for the transmon-chain model, across multiple system sizes and GPU
batch sizes, checked **entry-by-entry over every basis state**.

## 1. What was verified

The open-system propagator is the superoperator `U` with `ρ(t) = U[ρ(0)]`. For a
system of Hilbert dimension `D = d^N` it has `D²` columns, one per basis density
matrix `|row⟩⟨col|`. We reconstruct it column-by-column by **evolving all `D²`
basis density matrices** through both engines and comparing the resulting `D×D`
matrices entry-by-entry.

Systems checked (parameters identical to `prop_qt.py`'s `N=2,d=2` fixture, scaled
to each size):

| tag  | N | d | D = d^N | density_dim = D² |
|------|---|---|---------|------------------|
| n2d2 | 2 | 2 | 4       | 16               |
| n2d3 | 2 | 3 | 9       | 81               |
| n4d2 | 4 | 2 | 16      | 256              |

GPU batch sizes swept: **1, 2, 4, 8** (plus the full `density_dim` batch).

## 2. Model equivalence (analytic)

The C++ `build_transmon_chain_model` and the QuTiP `simulate_transmons` are the
same model term-for-term:

- `H₀ = Σᵢ ωᵢ aᵢ†aᵢ + (αᵢ/2) aᵢ†aᵢ†aᵢaᵢ`  (the `aᵢ†aᵢ†aᵢaᵢ = nᵢ(nᵢ−1)` term is the C++ `0.5·n(n−1)` operator scaled by α)
- coupling `Σᵢ gᵢ (aᵢ†aᵢ₊₁ + aᵢaᵢ₊₁†)`
- dissipators `√(1/T1ᵢ)·aᵢ` (relaxation) and `√(1/Tφᵢ)·nᵢ` (pure dephasing)
- drive on each driven site: `Ω(t)[(a+a†)cos(ωₐt) + i(a−a†)sin(ωₐt)]`, which is
  algebraically identical to QuTiP's `a·Ω(t)e^{iωₐt} + a†·Ω(t)e^{−iωₐt}` for real `Ω(t)`.

Parameters: `ωᵢ = 5.0 + 0.05·i`, `αᵢ = −0.3`, `gᵢ = 0.02`, `T1 = 20`, `Tφ = 30`,
Gaussian envelope `Ω(t) = 0.1·exp(−½((t−20)/10)²)`, `ωₐ = ωᵢ`, driven sites
`range(N//2)`, evolution `t ∈ [0, 40]`.

**Index/ordering convention** (verified against the C++ source): basis-state index
`s` puts a `1` at row-major flat position `s` of the vectorized density matrix,
i.e. initial `|s//D⟩⟨s%D|`; the final state is dumped row-major. The C++ Hilbert
index is **site-0-major** (`compute_strides` → `[d^(N-1),…,d,1]`), identical to
QuTiP's `tensor([site0,…,siteN-1])` ordering, so the dense `D×D` matrices line up
directly.

## 3. Method

- **C++:** new non-invasive CLI flags on `run_sim` (the trusted validation path is
  untouched and still passes). `-dump_propagator` evolves all `density_dim` basis
  states; `-dump_batch_sizes "1,2,4,8"` evolves the first `B` states in a batched
  GPU solve of size `B`; `-dump_num_transmons/-dump_cutoff_dim/-dump_steps` set the
  system and the fixed RK4 step count. Each final state is printed at 17-digit
  precision between `PROPAGATOR_DUMP_BEGIN/END` markers.
- **QuTiP:** `qutip_reference.py` evolves every basis density matrix independently
  with `mesolve` (vern9, atol 1e-12, rtol 1e-10) and saves the full reference.
- **Compare:** `compare_exhaustive.py` loads both and reports the worst entry, the
  exact count of complex entries compared, and the convergence trend.

The C++ benchmark integrates with **fixed-step RK4** (`TSRK`/`TSRK4`/`TSADAPTNONE`);
QuTiP uses **adaptive vern9**. They therefore cannot agree bit-for-bit at any finite
step count — the proof that they are the *same propagator* is that the C++↔QuTiP
residual falls as **O(dt⁴)** under step refinement down to QuTiP's own accuracy
floor (~1e-8, the `mesolve`-vs-`propagator` self-consistency level).

## 4. Results — exhaustive, entry-by-entry

For every system, every batch size 1/2/4/8, and the full propagator, we compared
**every** density-matrix entry of **every** evolved state to the QuTiP reference.

### Converged agreement (worst over EVERY entry / state / batch)

| System | full propagator | entries compared | worst \|C++ − QuTiP\| @ 1000 steps | worst @ 16000 steps |
|--------|-----------------|------------------|------------------------------------|---------------------|
| n2d2   | 16 × 16         | 992              | 9.247e-04                          | **1.418e-08**       |
| n2d3   | 81 × 81         | 15,552           | 8.697e-04                          | **1.272e-08**       |
| n4d2   | 256 × 256 (65,536 entries) | 138,752 | 1.210e-03                  | **1.809e-08**       |

Total: **~155,000 complex-entry comparisons**, all passing.

### Batch-size sweep vs QuTiP (max over all states in each batch)

| batch | n2d2 @1k → @16k | n2d3 @1k → @16k | n4d2 @1k → @16k |
|-------|-----------------|-----------------|-----------------|
| 1     | 1.89e-4 → 2.86e-9 | 5.85e-4 → 8.31e-9 | 2.93e-4 → 6.48e-9 |
| 2     | 6.72e-4 → 1.03e-8 | 7.79e-4 → 1.11e-8 | 5.14e-4 → 8.10e-9 |
| 4     | 7.23e-4 → 1.11e-8 | 7.79e-4 → 1.11e-8 | 1.21e-3 → 1.81e-8 |
| 8     | 7.23e-4 → 1.11e-8 | 8.25e-4 → 1.23e-8 | 1.21e-3 → 1.81e-8 |

### RK4 step-refinement convergence (textbook O(dt⁴))

`max|C++ − QuTiP|` over states 0–7 as the fixed step count rises:

| steps | dt     | n2d2     | n2d3     | n4d2     |
|-------|--------|----------|----------|----------|
| 1000  | 0.040  | 7.23e-4  | 8.25e-4  | 1.21e-3  |
| 4000  | 0.010  | 2.85e-6  | 3.18e-6  | 4.65e-6  |
| 16000 | 0.0025 | 1.11e-8  | 1.23e-8  | 1.81e-8  |

Each 4× refinement shrinks the error ~256× = 4⁴ (the exact RK4 rate) until it hits
QuTiP's ~1e-8 floor. This is the definitive proof the two engines compute the same
propagator.

## 5. Independent cross-checks

| Check | n2d2 | n2d3 | n4d2 |
|-------|------|------|------|
| **Batch-size independence** (C++ vs C++, state s across all B) | 0 (bit-identical) | 4.4e-16 | 4.4e-16 |
| batch=8 block ≡ full-dump first-8 states | 0 | 0 | 0 |
| **GPU vs CPU** backend (full propagator) | 2.55e-15 | 2.22e-15 | (GPU-only at D=16) |
| **Trace preservation** worst \|Tr(U\|j⟩⟨k\|) − δⱼₖ\| | 1.55e-15 | 2.67e-15 | 4.44e-15 |
| **Hermiticity** worst \|U(\|j⟩⟨k\|) − U(\|k⟩⟨j\|)†\| | 0 | 3.90e-16 | 0 |
| **QuTiP propagator vs independent mesolve** (self-consistency) | 7.7e-8 | — | — |
| **Site-order sanity** (wrong reversed labeling) | 0.133 ≫ | 0.093 ≫ | 0.071 ≫ |

The site-order sanity matches each C++ result against QuTiP under a *wrong*
(site-reversed) labeling and confirms the diff blows up ~100×, proving the
site-0-major convention is the correct one.

## 6. Honest framing of "exact"

- **C++ vs C++ across batch sizes: bit-identical** (0 to ~4e-16). The batched GPU
  path is provably batch-size independent.
- **C++ vs QuTiP: exact to QuTiP's own integration accuracy (~1e-8), not
  bit-identical** — two different integrators never agree bit-for-bit. The ~1e-3
  residual at the validation's default 1000-step config is *pure RK4 truncation*,
  uniform across every entry, and vanishes as O(dt⁴) under refinement.

## 7. Reproduce

```bash
# environment (UW-HEP qis* GPU node)
source setup_env.sh                      # PETSc + CUDA + sm_120
./build.sh                               # builds build_local/run_sim
source /nfs_scratch/kunnumbrath/qutip-venv/bin/activate   # qutip 5.0.4

# per system <tag N d>: generate C++ dumps (GPU), QuTiP reference, and compare
run() { ./build_local/run_sim -use_gpu_aware_mpi 0 "$@"; }
for s in "n2d2 2 2" "n2d3 2 3" "n4d2 4 2"; do set -- $s
  run -dump_propagator 1 -dump_num_transmons $2 -dump_cutoff_dim $3 -dump_batch_sizes 1,2,4,8 -dump_steps 1000  > propcheck/${1}_sweep1000.txt
  run -dump_propagator 1 -dump_num_transmons $2 -dump_cutoff_dim $3 -dump_batch_sizes 1,2,4,8 -dump_steps 16000 > propcheck/${1}_sweep16000.txt
  run -dump_propagator 1 -dump_num_transmons $2 -dump_cutoff_dim $3 -dump_steps 1000  > propcheck/${1}_full1000.txt
  run -dump_propagator 1 -dump_num_transmons $2 -dump_cutoff_dim $3 -dump_steps 16000 > propcheck/${1}_full16000.txt
  python qutip_reference.py   $1 $2 $3
  python compare_exhaustive.py $1 $2 $3
done
```

## 9. Adaptive time-stepping and batch-size dependence (expected, not a bug)

**Observation:** with an *adaptive* integrator (e.g. Dormand–Prince `-ts_type rk
-ts_rk_type 5dp -ts_adapt_type basic`), the same basis state evolved in different
GPU batch sizes gives slightly different results. **This is expected** — it is a
property of batched adaptive stepping, not a solver bug.

**Mechanism:** all `B` states are packed into one PETSc `Vec` and integrated by one
`TS` with one adaptive step-size controller. PETSc forms a *single* weighted error
norm over the *whole combined vector* and picks *one* Δt for the entire batch. That
error norm — and therefore the step sequence — depends on the batch contents, so the
same state follows a different timeline at different batch sizes.

Direct evidence (n2d2, rtol 1e-6, atol 1e-8): the adaptive solver takes **1082
steps for batch=1** but **1235 steps for batch=8** (the 8-state aggregate error
forces smaller steps); the chosen Δt already diverges by step 3.

**The cross-batch difference is bounded by, and scales linearly with, the
tolerance** — every batch's result is correct to within `atol`/`rtol`; the
discrepancy is tolerance-level truncation, not a physical error. (The RHS correctly
decouples the states — proven by the fixed-step path being bit-identical across
batches and matching QuTiP.)

| rtol / atol     | cross-batch max\|diff\| (state 0, B=1..8) | each batch vs QuTiP |
|-----------------|-------------------------------------------|---------------------|
| 1e-6  / 1e-8    | 2.5e-07                                   | ~1e-6 … 4e-6        |
| 1e-10 / 1e-12   | 2.5e-11                                   | ~6e-11 … 4e-10      |

(10⁴ tighter tolerance → 10⁴ smaller cross-batch difference.)

**Rectification — to get batch-independent results:**

1. **Fixed-step (recommended; the validated default):** `TSRK` + `TSRK4` +
   `TSADAPTNONE`. Identical predetermined steps for every batch ⇒ **bit-identical**
   results (0–4e-16). This is exactly why the validated batched path uses
   fixed-step.
2. **Tighten tolerance:** the cross-batch diff scales with tol, so rtol/atol
   1e-10/1e-12 drives it to ~1e-11 (negligible, though still not bit-identical).
3. **Unbatched adaptive:** evolve one state per solve (`-dump_batch_sizes 1`) so each
   state gets its own controller. A *single batched adaptive* solve cannot produce
   bit-identical per-state results across batch sizes — sharing one timeline is
   intrinsic to batching.

There is no way to make one batched adaptive solve reproduce per-state results
across batch sizes; for reproducible batching use fixed-step (option 1).

## 8. Files

| file | role |
|------|------|
| `cpp/src/apps/cli_main.cpp` | adds `-dump_propagator`, `-dump_batch_sizes`, `-dump_num_transmons`, `-dump_cutoff_dim`, `-dump_steps` (additive; trusted validation path untouched) |
| `verify_propagator.py` | first single-system (n2d2) propagator + convergence + mesolve self-check |
| `verify_systems.py` | per-system summary verifier (`<tag> <N> <d>`) |
| `qutip_reference.py` | full QuTiP reference, parallel `mesolve` → `propcheck/<tag>_qutip_full.npy` |
| `compare_exhaustive.py` | entry-by-entry C++ vs QuTiP over all states/batches → `propcheck/<tag>_exhaustive.txt` |
| `propcheck/<tag>_exhaustive.txt`, `propcheck/<tag>_result.txt` | recorded numeric results |

Large intermediate dumps (`propcheck/*_sweep*.txt`, `*_full*.txt`,
`*_qutip_full.npy`, root `*_dump*.txt`/`*.npy`) are regenerated by the commands
above and are git-ignored.
