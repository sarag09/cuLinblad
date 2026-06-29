# Design Doc — MPI multi-GPU state-sharding for the `-load_model` path

Status: design only (no implementation). Branch context: `julia-handoff`. Target: mirror the proven `-dump_tc` shard onto `run_loaded_model` so the 256 independent 4Q4C density matrices evolve disjointly across N GPUs.

Review status: adversarially reviewed — verdict **SOUND-WITH-FIXES**. The four required fixes (capture `N` before sharding; scope the golden gate to lindblad; state the 4Q4C anchor is self-consistency-validated not bit-identical; re-cite the SELF-communicator proof) are folded in below, marked **Risk found in review**.

---

## 1. Problem & current behavior

The `-dump_tc` benchmark path already shards across MPI ranks/GPUs. The `-load_model` path (the production Julia→C++ handoff path) does **not** — it is single-process.

- Dispatch: `cli_main.cpp:1323-1333`. The `-load_model` branch calls `run_loaded_model(model_path, results_path)` at `:1328-1330` and passes **only** `(model_path, results_path)` — no `mpi_rank`/`mpi_size`, no `-dump_tag`.
- `run_loaded_model` (`cli_main.cpp:979-1031`) is MPI-oblivious. It builds the **full** state list and evolves **all** of it:
  - open: `idx = computational_density_indices_for_dims(lm.model.local_dims, lm.num_qubits)` at `:1006-1008` (256 for 4Q4C), or `explicit_states = lm.initial_states.explicit_states` at `:999`;
  - closed: `computational_state_indices_for_dims(...)` at `:1016-1018`, or `explicit_states`.
  - evolve: `evolve_loaded_model_liouvillian(...)` `:1009-1010` / `evolve_loaded_model_schrodinger(...)` `:1019-1020`.
  - dump: `dump_loaded_model_result(*out, lm, timing, mode)` `:1014` / `:1024`, where `out` is `-results_out` file or `std::cout` (`:984-993`).

Consequence under `mpiexec -n N ... -load_model`: every rank evolves the **entire** state set (no speedup) and every rank writes the **same** `-results_out` file / interleaves stdout (corrupt output). Note that the MPI/GPU plumbing already runs for this path: node-local GPU binding by hostname + `cudaSetDevice` (`cli_main.cpp:1190-1297`) and global `mpi_rank`/`mpi_size` on `PETSC_COMM_WORLD` (`:1284-1287`) are all computed **before** the `:1323` dispatch. **Each rank is already bound to its own GPU; only the work-partition and per-rank output are missing.**

---

## 2. Shardable unit & slicing

**The shardable unit is the list of independent initial states**, evolved as a *single batched* cuTENSOR `evolve_state_batch_cuda_ts` over the whole batch (`tunable_coupler_benchmark.cpp:1011-1012`, batch size = `initial.size()`). It is **not** a per-state loop. So the correct insertion point is the **state list, before it is handed to the evolve function** — shard the input list and the batched evolve naturally runs only the rank's slice with zero cross-rank communication.

**No-deadlock proof (corrected citation — Risk found in review).** Every evolve-side PETSc object lives on `PETSC_COMM_SELF`, so each rank runs an independent `TSSolve` with **zero** MPI collectives inside evolve. The proof is the actual object creation — `TSCreate(PETSC_COMM_SELF, …)` / `VecCreate(PETSC_COMM_SELF, …)` at `batch_evolution.cpp:727,730` and `:826,829`, plus `tunable_coupler_benchmark.cpp:248,255` — and every `TSSolve`/`Vec*` op runs on those SELF objects (`batch_evolution.cpp:1060`, `tunable_coupler_benchmark.cpp:320`). (The earlier draft cited `tunable_coupler_benchmark.cpp:996-997`/`:956-957`; those are only `PetscCallAbort(PETSC_COMM_SELF, create_cuda_..._context(...))` *error-abort* wrappers, **not** proof the TS/Vec live on SELF — do not rely on them for the claim.) The only collectives in the whole run are the startup `MPI_Allgather` of hostnames (`cli_main.cpp:1231`, all ranks, fixed size) and the collective `PetscFinalize` at `:1331` (all ranks reach it). No deadlock.

Reuse `shard_indices` (`cli_main.cpp:540-553`) unchanged. It is a **contiguous block partition**, identity for `size<=1`, remainder spread to **low** ranks:
```
base = n/size; rem = n%size;
begin = r*base + min(r, rem);  cnt = base + (r < rem ? 1 : 0)
```
Union is exact, gap/overlap-free, and **global order is preserved by contiguity** (this is the property `merge_ranks.py` relies on).

Count `n` per path:
- **computational, open**: `n = (2^n_q)^2` density-column indices from `computational_density_indices_for_dims` (`tunable_coupler_benchmark.cpp:918-932`) = 256 for 4Q4C. `shard_indices(idx, rank, size)` applies directly (it is a `vector<Index>`).
- **computational, closed**: `n = 2^n_q` state indices from `computational_state_indices_for_dims`. Same direct use.
- **explicit**: `explicit_states` is `vector<vector<Complex>>` (`model_io.cpp:221-243`), **not** a `vector<Index>`, so `shard_indices` cannot be applied directly. Compute `begin`/`cnt` with the **same arithmetic** (replicate the 4 lines, or add a small templated overload) and pass `explicit_states[begin .. begin+cnt)` to the evolver. `n = explicit_states.size()`.

Each rank's `final_states` then come back numbered **local** `0..cnt-1` (the evolvers always index `0..initial.size()-1`; `evolve_loaded_model_liouvillian` `:969-1016`, `evolve_loaded_model_schrodinger` `:937-967`).

---

## 3. Code changes (file-by-file, concrete)

### 3a. `cli_main.cpp` — thread rank/size + tag into the dispatch (`:1323-1333`)
After the existing `-results_out` read (`:1326-1327`), read `-dump_tag` exactly as the dump_tc path does at `:1420-1423` (default e.g. `"loadmodel_shard"`), and pass `mpi_rank`, `mpi_size`, `tag` into `run_loaded_model`:
```cpp
char lm_tag_buf[128] = "loadmodel_shard";
PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetString(
    nullptr, nullptr, "-dump_tag", lm_tag_buf, sizeof(lm_tag_buf), nullptr));
const int rc = run_loaded_model(
    std::string(model_path_buf),
    results_set == PETSC_TRUE ? std::string(results_buf) : std::string(),
    static_cast<int>(mpi_rank), static_cast<int>(mpi_size), std::string(lm_tag_buf));
```
`mpi_rank`/`mpi_size` are already in scope from `:1284-1287`.

### 3b. `cli_main.cpp` — `run_loaded_model` signature + sharding (`:979-1031`)
Change the signature to
```cpp
int run_loaded_model(const std::string& model_path, const std::string& results_path,
                     int mpi_rank = 0, int mpi_size = 1, const std::string& tag = "loadmodel_shard");
```
(Note — Risk found in review, INFO: `run_loaded_model` has exactly **one** caller (`:1328`) and no header declaration, so the default args do not actually keep "other callers" compiling — there are none. They are kept only to make `size==1` behavior obviously identical and the call site readable; the "other caller" justification is moot.)

**Capture `N`, guard, THEN slice — order is load-bearing (Risk found in review, the one genuine correctness trap).**
The computational branch overwrites `idx` in place via `idx = shard_indices(idx, …)`. After that overwrite, `idx.size()` is the **local** count, so an empty-shard guard written against `idx.size()` would compare against the post-shard size and be wrong. You must snapshot `N` **before** the `shard_indices` call — exactly as the `-dump_tc` template does, guarding on `selected_state_indices.size()` (`:1460-1469`) *before* the `shard_indices` at `:1474`. The explicit branch is naturally safe (it slices into a fresh `states_slice` and never overwrites `explicit_states`), but follow the same `N`-first pattern for symmetry.

Open branch (`:1005-1014`):
```cpp
std::vector<Index> idx = use_explicit
    ? std::vector<Index>{}
    : computational_density_indices_for_dims(lm.model.local_dims, lm.num_qubits);

// Capture N BEFORE any shard_indices overwrite (Risk #1).
const std::size_t N = use_explicit ? explicit_states.size() : idx.size();
if (mpi_size > 1 && static_cast<std::size_t>(mpi_size) > N) {        // empty-shard guard, fires collectively
    if (mpi_rank == 0)
        std::cerr << "[load_model] mpi_size=" << mpi_size << " > N=" << N
                  << " would create empty shards; abort." << std::endl;
    MPI_Abort(PETSC_COMM_WORLD, 1);
}

std::vector<std::vector<Complex>> states_slice;
const std::vector<std::vector<Complex>>* states_ptr = &explicit_states;
if (use_explicit) {
    // replicate shard_indices arithmetic on the N explicit kets
    const std::size_t base = N / mpi_size, rem = N % mpi_size, r = mpi_rank;
    const std::size_t b = r*base + std::min(r,rem), c = base + (r<rem?1:0);
    states_slice.assign(explicit_states.begin()+b, explicit_states.begin()+b+c);
    states_ptr = &states_slice;
} else {
    idx = shard_indices(idx, mpi_rank, mpi_size);   // identity for size<=1; safe AFTER N captured
}
const auto timing = evolve_loaded_model_liouvillian(
    lm.model, idx, *states_ptr, lm.evolution.t0, lm.evolution.tfinal, num_steps);
```
Closed branch (`:1015-1024`) is identical with `computational_state_indices_for_dims` + `evolve_loaded_model_schrodinger`. **The evolve_* functions are unchanged** — they size the batch to the passed list (`tunable_coupler_benchmark.cpp:946-948`, `:976-994`).

Because `N` and `mpi_size` are model-driven and identical on every rank, the guard fires **collectively** — safe (no partial-abort hang).

### 3c. `cli_main.cpp` — per-rank output
Replace the single `out`/`-results_out` selection (`:984-993`) so that **when `mpi_size>1`** it writes a per-rank file and ignores `-results_out`, mirroring `tc_dump_states` (`:574-590`):
```cpp
std::ofstream fout;
std::ostream* out = &std::cout;
if (mpi_size > 1) {
    std::error_code ec; std::filesystem::create_directories("propcheck", ec);
    const std::string fname = "propcheck/" + tag + "_rank" + std::to_string(mpi_rank) + ".txt";
    fout.open(fname);
    if (!fout.is_open()) {                           // missing shard ⇒ corrupt merge ⇒ hard abort
        std::cerr << "[rank " << mpi_rank << "] cannot open " << fname << std::endl;
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }
    out = &fout;
} else if (!results_path.empty()) {                  // size==1: byte-identical to today
    fout.open(results_path);
    if (!fout.is_open()) { /* existing :988-991 error */ return 1; }
    out = &fout;
}
```
`dump_loaded_model_result` (`:914-975`) is **unchanged** — it already takes `std::ostream&`, numbers `STATE` by the **local** loop index `s` (`:953-954`), writes `batch_size`/`num_states` = the **local** `final_states.size()` (`:930-931`), and applies the rotating-frame lab back-transform per `(k,l)` inside the loop (`:935-971`), so it shards with zero change.

### 3d. Guards (mirror `:1448-1469`)
- **Too-many-ranks / empty shard**: implemented in 3b using the **pre-shard** `N` (Risk #1). Mirrors `:1460-1469`. An empty batch makes `create_cuda_batch_execution_context` choke (open) or silently misbehave (closed), so this must hard-abort.
- **File-open failure**: the `MPI_Abort` in 3c (mirrors `:580-584`). A silently dropped shard would feed a wrong merged anchor, and there is **no golden compare in production `run` mode**.
- Note: unlike `-dump_tc` there is no `-tc_computational` flag to require; the encoding (computational vs explicit) is model-driven (`lm.is_open`, `use_explicit`). Both encodings shard, so no "require computational" abort is needed — but see Open Question Q3.

### 3e. Backward compatibility (load-bearing for the validation oracle)
For `mpi_size==1`: `shard_indices` is identity (`:543-544`), the explicit slice arithmetic gives `base=N, rem=0, b=0, c=N` (full list), no `propcheck/` file is created, and output goes to `-results_out`/stdout exactly as `:984-993` today. The single-process dump must remain **byte-identical**, because it is the golden reference the bit-identical gate compares against — **but note the reach limit in §5/Risk #3: a single-process golden is only feasible for the small (3Q2C) model, not for the 4Q4C N=256 anchor.**

---

## 4. Output & merge

**Per-rank file**: `propcheck/<tag>_rank<r>.txt`, `r` = **global** `mpi_rank`, `tag` from `-dump_tag` (default `loadmodel_shard`). One self-contained `PROPAGATOR_DUMP_BEGIN … STATE … PROPAGATOR_DUMP_END` block per file (`dump_loaded_model_result`, `:920-974`).

**Index ordering** (explicit, load-bearing): each rank evolves a **contiguous, global-ordered** slice (Section 2) and numbers `STATE` **locally 0..cnt-1**. The global 0..N-1 order is reconstructed **purely from rank-order + contiguity** — no global index is ever written. Do **not** switch to interleaved partitioning or the merged order silently scrambles vs the single-process golden.

**Merge — reuse `validation/merge_ranks.py` UNCHANGED.** Argv (verified `:26-29`):
```
python validation/merge_ranks.py <tag_prefix> <merged_out.txt> [expected_nranks] [expected_N]
```
The 4th arg is the **total merged STATE count** (`expected_N`), not a column width. `merge_ranks.py`:
- globs `<tag>_rank*.txt`, sorts by rank int, requires the contiguous set `{0..k-1}` (`:52-55`) and exactly `expected_nranks` if given (`:56-57`);
- requires each rank file to have its own `BEGIN`+`END`;
- requires all rank `BEGIN` headers identical **except** `batch_size`/`num_states` (so `result_frame`, `density_dim`, `mode`, etc. must match — they are model-driven, hence identical);
- validates each `STATE` block length == header `density_dim` (lindblad) **or** `state_len` (schrodinger) — i.e. `hdr_len = density_dim or state_len` (`merge_ranks.py:103`), so **merge handles both encodings**;
- renumbers `STATE` globally `0..N-1` in rank-then-local order and rewrites merged `batch_size`/`num_states` to the total (`:116`).

`dump_loaded_model_result` already emits exactly this structure. Its `PROPAGATOR_DUMP_END backend=gpu mode=…` line has no `batch_size` token, so merge's `setkv` on END is a harmless no-op. **No change to `merge_ranks.py` is required.**

**Risk found in review (INFO — `steps=` header quirk):** `dump_loaded_model_result` writes `steps=lm.evolution.num_steps` (`:928`), which can be `0` when the JSON omits `num_steps`, while the evolve actually used the local `num_steps` defaulted to `1000` (`:983`). This mislabels the header but is **identical on every rank** (model-driven), so the header-agreement gate (`merge_ranks.py:72-77`) passes and sharding is unaffected. Pre-existing; flagged so a future author does not "fix" it **per-rank** and thereby break merge.

`expected_N` per path: `(2^n_q)^2` (open computational, 256 for 4Q4C), `2^n_q` (closed computational), or `M` (explicit). `D = prod(local_dims) = 6561` for 4Q4C.

---

## 5. Validation plan

Reuse the `-dump_tc` acceptance harness pattern (`cluster/run_mpi_shard.sh:43-48`): multi-rank shard → `merge_ranks.py` → single-process reference → `golden_regression.py` must report **BIT-IDENTICAL** (worst `|merged−single| == 0.0`; gate exits 0 iff `< 1e-6`, `golden_regression.py:69-77`). Bit-identicality holds because both runs are fixed-step `TSRK4`+`TSADAPTNONE` (`tunable_coupler_benchmark.cpp:998-1002`) on the same GPU kernels over disjoint contiguous slices; the merged `batch_size` is rewritten to N so its `(gpu, N)` key matches the single-process `(gpu, N)` (`golden_regression.py:33`).

**Risk found in review (Risk #2 — golden gate is lindblad-only):** `golden_regression.py:27` hardcodes `np.array(ent).reshape(D, D)` for **every** STATE block. A schrödinger/closed dump writes `state_len = D` entries per STATE (`cli_main.cpp:924`, `:959-962`), not `D*D`, so `reshape(D,D)` on a length-`D` block raises `ValueError`. Therefore the **bit-identical gate is valid only for `mode=lindblad`** (`merge_ranks.py` itself handles both — only the *gate* is restricted). Any closed/Schrödinger shard test must either (a) extend `golden_regression.py` to read `state_len` for the schrodinger mode, or (b) fall back to merge self-consistency only. The 4Q4C deliverable is open (lindblad, `D*D` per STATE → `reshape(6561,6561)` works), so it is unaffected — but the §7 "3Q2C fast" check **must use an open model** to exercise the golden gate.

**Risk found in review (Risk #3 — the 4Q4C N=256 anchor cannot be bit-identical-validated):** a single-process reference holds the **entire** batch in one ctx (`tunable_coupler_benchmark.cpp:996-997`, batch sized to `initial.size()`). One 4Q4C density matrix is ~688 MB (§6) and 256 of them ≈ **176 GB**, which exceeds a single RTX PRO 6000 (~96 GB). So a single-process golden for N=256 is **infeasible** on the target hardware. State plainly: **the real anchor (4Q4C, N=256) is validated only by merge self-consistency (contiguity / END / expected-`N` / header-agreement gates) + a sub-shard cross-check; it is never bit-identical-validated.** Bit-identicality is demonstrable only on the small 3Q2C model. This is an acceptable strategy, but the oracle does **not** reach the actual deliverable — do not frame the single-process run as the 4Q4C golden.

Add a `-load_model` analogue of `run_mpi_shard.sh` (new script, e.g. `cluster/run_mpi_shard_loadmodel.sh`) that runs, for a chosen model JSON:
```bash
# 1) multi-rank sharded evolve (N ranks = N GPUs)
mpiexec -n $NRANKS ./build_local/run_sim -load_model $MODEL -dump_tag $TAG
# 2) merge (UNCHANGED script). N = (2^NQ)^2 open, or num_states for explicit
$PY validation/merge_ranks.py "propcheck/$TAG" "propcheck/${TAG}_merged.txt" "$NRANKS" "$N"
# 3) single-process golden  -- ONLY for the small (3Q2C) model; infeasible at 4Q4C N=256 (Risk #3)
./build_local/run_sim -load_model $MODEL -results_out "propcheck/${TAG}_single.txt"
# 4) bit-identical gate (lindblad only, Risk #2; D = prod(local_dims) = 6561 for 4Q4C)
$PY validation/golden_regression.py "propcheck/${TAG}_merged.txt" "propcheck/${TAG}_single.txt" "$D"
```
Sequence:
1. **3Q2C fast, OPEN model** (small D, 2 GPUs/ranks): quick confirmation that slice+merge+local-numbering reproduces single-process bit-identically. Must be an **open/lindblad** model so step 4 is exercisable (Risk #2). Use a 3Q2C open model JSON (`config_2Q1C_cz.jl`-style export scaled, or the existing 3Q2C model).
2. **4Q4C anchor** (the real target, 256 density matrices, `N=256`, `D=6561`): run on the available GPU count (memory permitting — see §6), merge to 256 states, then **self-consistency validation only** (merge contiguity / expected-`N` / header-agreement gates + a sub-shard cross-check, e.g. compare a 2-rank vs 4-rank merge over the same model). No single-process golden at this size (Risk #3).

Keep `merge` and `golden_regression` **out of the MPI job** (separate Python invocations post-run), matching current `-dump_tc` practice — each rank calls `PetscFinalize` independently (`:1331`) after its file write completes.

---

## 6. Edge cases & non-goals

- **Rotating-frame lab back-transform**: per-`(k,l)` inside `dump_loaded_model_result` (`:935-971`), depends only on `E_k, E_l, T` from a global `Efr` table derived from `frame_omegas`+`local_dims` (identical on every rank). No cross-state coupling → shards trivially, no change. `result_frame` is identical across ranks so it passes merge's header-agreement gate.
- **Dressed-state projection / construction**: done Julia-side before export; the C++ path only consumes `explicit_states`/term-list. Unaffected by sharding.
- **Per-rank GPU memory (drives min rank count, not just speed)**: one 4Q4C density matrix is `D*D = 6561² ≈ 4.3e7` complex128 ≈ **688 MB**; a rank's batched ctx holds `shard_size` of them simultaneously (`create_cuda_batch_execution_context`, `tunable_coupler_benchmark.cpp:997`). 256 states ≈ 176 GB total. **Choose enough ranks that `shard_size * 688 MB` + solver workspace fits each GPU.** This is also why a single-process N=256 reference is infeasible (§5, Risk #3). Deployment constraint, not a code change.
- **Solver replication**: `make_solver` (`tunable_coupler_benchmark.cpp:974`) runs on every rank (full D superoperator context, on `PETSC_COMM_SELF` per §2). Redundant but correct (matrix-free), acceptable.
- **`num_steps` consistency**: model-driven (`lm.evolution.num_steps`, default 1000 at `:983`), identical across ranks → identical integration grid. Worth an assert but not a risk. (See the related `steps=` header-label quirk in §4.)
- **Out of scope**: changing the dump format; changing `merge_ranks.py`; cross-rank communication / `Allreduce`; the full-propagator (`evolve_all_states`) and batch-sweep dumps (those are `-dump_tc`-only); GPU-aware MPI (states are independent, no device-to-device transfer); the closed/Schrodinger explicit-state physics. Merge stays a separate post-step.

---

## 7. Phased implementation plan

Each phase is independently testable.

- **(a) Plumb + slice + per-rank file.** §3a–3e. **Capture `N` before `shard_indices` (Risk #1).** Build; run `mpiexec -n 1` and confirm output is byte-identical to current single-process (identity shard, no `propcheck/` file). Run `mpiexec -n 2` on a small **open** 3Q2C model and confirm two `propcheck/<tag>_rank{0,1}.txt` files appear with disjoint local-numbered STATE blocks summing to N.
- **(b) Merge support.** Confirm `merge_ranks.py propcheck/<tag> merged.txt 2 N` succeeds **unchanged** (contiguity/END/expected-N/header-agreement gates pass). If any header field is rank-varying, fix the dump rather than the merge script.
- **(c) Validate 3Q2C bit-identical.** Run the new `run_mpi_shard_loadmodel.sh` validate flow on an **open/lindblad** 3Q2C model (Risk #2); `golden_regression.py` must print BIT-IDENTICAL (worst 0.0). This is the **only** phase that achieves bit-identicality.
- **(d) 4Q4C anchor.** Run on the production GPU count (memory-sized per §6) with `N=256`, `D=6561`; merge to 256 states; **self-consistency validation only** — contiguity / expected-N / header-agreement gates + sub-shard cross-check. No single-process golden at this size (Risk #3). This is the deliverable anchor.

---

## 8. Open questions for the author

1. **Auto-enable vs explicit flag.** Should sharding auto-engage whenever `mpi_size>1` on `-load_model` (proposed, mirrors `-dump_tc`), or gate behind an explicit `-shard_load_model`? Auto is simpler and matches the dump_tc precedent; a flag is more defensive against accidental `mpiexec` launches. **Recommendation: auto-enable** (the empty-shard guard + per-rank files make accidental multi-rank safe and observable).
2. **`-results_out` semantics under `mpi_size>1`.** Proposed: ignore `-results_out` and always write `propcheck/<tag>_rank<r>.txt`, then merge into the user's `-results_out` via a wrapper. Alternative: treat `-results_out` as a prefix for per-rank files. **Recommendation: ignore + wrapper-merge** (keeps the dump_tc pattern and size==1 backward-compat clean). Confirm the wrapper should `cp propcheck/<tag>_merged.txt $results_out`.
3. **Explicit path under multi-rank.** Unlike `-dump_tc` (which requires `-tc_computational`), the explicit-states encoding *can* shard. Do we allow multi-rank explicit, or restrict multi-rank to the computational encoding (simpler, avoids the `vector<vector<Complex>>` slicing edge)? **Recommendation: allow it**, but only via the replicated begin/cnt slice in §3b (do not over-engineer a generic template) — and verify `expected_N` in the acceptance script switches to `M` for explicit. Note: if the explicit model is **closed**, the bit-identical gate is unavailable (Risk #2) — fall back to merge self-consistency.
4. **`-dump_tag` collision with the dump_tc default.** The dump_tc path defaults `-dump_tag` to `tc_shard` (`:1421`). Proposed load_model default is `loadmodel_shard`. Confirm distinct defaults are acceptable (prevents accidental cross-path file mixing in `propcheck/`).
5. **Header field `mode`/`result_frame`/`steps` stability.** Merge requires all non-`batch_size`/`num_states` header keys identical across ranks. These are model-driven today (safe), including the `steps=` field even when mislabeled 0 (§4, Risk #6/INFO). Flag for future authors: any rank-dependent header field added later (e.g. per-rank timing) **must not** go in the `BEGIN` line or it breaks the merge — and do not "fix" the `steps=0` label per-rank.

---

Relevant files: `/nfs_scratch/kunnumbrath/cuLinblad/cpp/src/apps/cli_main.cpp` (`run_loaded_model` 979-1031, `dump_loaded_model_result` 914-975, `shard_indices` 540-553, dump_tc shard/guards template 557-601 / 1444-1495, dispatch 1323-1333); `/nfs_scratch/kunnumbrath/cuLinblad/cpp/src/core/tunable_coupler_benchmark.cpp` (`evolve_loaded_model_*` 937-1017, `computational_density_indices_for_dims` 918-932); `/nfs_scratch/kunnumbrath/cuLinblad/cpp/src/core/batch_evolution.cpp` (727/730/826/829 — `PETSC_COMM_SELF` TS/Vec creation, the actual no-deadlock proof); `/nfs_scratch/kunnumbrath/cuLinblad/cpp/src/io/model_io.cpp` (118-135 open-detect, 217-246 explicit initial states); `/nfs_scratch/kunnumbrath/cuLinblad/validation/merge_ranks.py` (handles both `density_dim`/`state_len`); `/nfs_scratch/kunnumbrath/cuLinblad/validation/golden_regression.py` (line 27 hardcodes `reshape(D,D)` → lindblad-only gate); `/nfs_scratch/kunnumbrath/cuLinblad/cluster/run_mpi_shard.sh` (template for the new `run_mpi_shard_loadmodel.sh`).