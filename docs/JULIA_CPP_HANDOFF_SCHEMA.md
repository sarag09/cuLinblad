# Julia ⇄ C++ GPU-solver handoff — term-list schema & `use_gpu` routing (DRAFT)

Goal: when a Julia 4Q4C config sets `use_gpu = true`, build the model in Julia as usual but route the
*evolution* to the cuLindblad matrix-free GPU solver (closed |ψ⟩ Schrödinger **and** open ρ Lindblad),
then read results back into the Julia scoring/optimizer — instead of calling Julia `sesolve`/`mesolve`.

The two repos stay **separate**; they communicate via a serialized **local term-list** file (never the
assembled 6561×6561 operator). New code: a Julia exporter + a C++ JSON importer + a closed-form pulse
factory. Solver core unchanged.

---

## 1. Architecture / routing (`use_gpu = true`)

Julia branch point: top of `compute!` (gate_sim.jl:421), after the `H_parts` assert. `use_gpu` is read
in `load_prms_qcc!` (gate_sim.jl:398) exactly like `use_computational`. Branching in `compute!` (not
`run!`) routes **every** optimizer evaluation through the GPU with no extra plumbing.

```
compute!(qcc):
  if qcc.use_gpu:  qcc.Uro = run_gpu_channel(qcc); return   # else existing Julia propagator path
```

`run_gpu_channel(qcc)` (new `gpu_export.jl`):
1. Decompose `qcc.H_parts (H0,Hs,gs)` + `qcc.sys` into **local k-site blocks** (re-walk the per-site
   Duffing + per-pair coupling construction — do NOT export the assembled H0), transpose-flatten each
   block to row-major real/imag, classify each tuned-key envelope into `shape + params`. Write
   `model_<runid>.json`.
2. Synchronously shell out to the C++ binary:
   `run_sim -load_model model.json -results_out results.txt -tc_rotating_frame 1 -tc_integrator 1
   -tc_fused_diagonal 1 -tc_fuse_coupling 1 -tc_fused_diag_prepass 1 [-tc_schrodinger for closed]`.
3. Parse the `PROPAGATOR_DUMP` result (the M length-D **lab-frame** |ψ_a(T)⟩).
4. Apply the **dressed projection Julia-side**: `U_comp[i,a] = ⟨dressed_i | ψ_a⟩` (helper.jl:162-209),
   wrap as the same `QuantumObject` `_propagator_unitary` returns — so the optimizer sees no difference.

Synchronous file-based shell-out per evaluation (the optimizer is a low-dim pattern search, tens of
evals). A persistent server is a later optimization, not needed for validation.

---

## 2. Term-list JSON schema (v1) — what Julia exports, C++ reads

```jsonc
{
  "schema_version": 1,
  "device": {
    "site_labels": ["Q1","Q2","C1"],     // qubits first, then couplers; site-0 = first tensor factor = MSB
    "local_dims": [3,3,3],               // -> Model.local_dims; may be NON-uniform (e.g. [3,3,2,2,3,2,2,2])
    "n_qubits": 2, "n_couplers": 1
  },
  "system": "closed",                    // "closed" => |ψ⟩ path; "open" => ρ path. MUST equal isempty(c_ops)

  // STATIC H as LOCAL k-site dense blocks (re-walked from primitives, NOT the assembled operator).
  // Tuned modes contribute ZERO static frequency (their full ω(t) lives in time_dependent_terms).
  "hamiltonian_terms": [
    { "name":"duffing_Q1", "sites":[0], "row_dim":3, "col_dim":3,    // diag(ω·n + ½α·n(n-1)), rad/s
      "matrix": { "real":[...9 row-major...], "imag":[...9...] } },
    { "name":"coupling_Q1_Q2", "sites":[0,1], "row_dim":9, "col_dim":9,   // g(a†b+ab†−a†b†−ab) if CRT
      "matrix": { "real":[...81...], "imag":[...81...] } }
  ],

  "dissipator_terms": [],                // single-site √rate·block; EMPTY today (closed). Open = future.

  // H(t) = H0 + Σ coeff_i(t)·H_i   (mirrors H_parts.Hs / gs)
  "time_dependent_terms": [
    { "name":"HO_omega_C1", "sites":[2], "row_dim":3, "col_dim":3,        // n operator on tuned site
      "matrix": {"real":[0,1,2 row-major],"imag":[0...]},
      "coefficient": { "kind":"envelope", "key":"omega_C1" } },           // coeff = full ω_C1(t)
    { "name":"coupling_Q1_C1", "sites":[0,2], "row_dim":9, "col_dim":9,   // g computed at ω_tuned=1
      "matrix": {...},
      "coefficient": { "kind":"product_sqrt", "keys":["omega_C1"] } }     // coeff = ∏ √ω_key(t)
  ],

  "envelopes": {                         // named library; C++ compiles each into a TimeScalarFunction
    "omega_C1": { "shape":"sine", "a":2.63e10, "b":3.26e10, "T":5.0e-8 }  // peak, idle, gate time (s)
    // flattop adds "tr"; flat_top_gaussian adds "t0","tau","sigma" (use a_eff renorm);
    // shaped adds "harmonics":[...]; ccz_slepian adds "lambda","theta_i","theta_knots","omega_knots"
  },

  "evolution": {
    "t0":0.0, "tfinal":5.0e-8,
    "num_steps":4000,                    // fixed-step count (Julia is adaptive — exporter must choose; OPEN #1)
    "rotating_frame":true,               // -> -tc_rotating_frame 1
    "frame_omegas":[2.51e10,2.51e10,3.26e10],  // bare ω per site (rad/s), site-0-major — the frame contract
    "drop_counter_rotating":false,
    "result_frame":"lab"                 // C++ back-transforms to lab BEFORE returning
  },

  "initial_states": { "encoding":"computational", "num_states":4 }  // 2^n_q kets, Q1=MSB, couplers |0>
}
```

**Complex encoding:** each block is row-major `matrix[row*col_dim+col]`, length `row_dim*col_dim`, as two
parallel float64 arrays `{"real":[...], "imag":[...]}` (exact complex128 round-trip; matches the existing
`verify_*.py` parse style).

---

## 3. The four design pillars (with the *why*)

- **Convention (the #1 risk) — fixed entirely in the Julia exporter, C++ stays canonical row-major.**
  Julia `.data` is column-major; `vec(permutedims(B.data))` yields the exact C++ row-major order. Tensor
  order already matches (Julia `tensor` first factor = MSB = C++ site-0-major). The local block is **NOT**
  `op.data` (that's full-space) — rebuild it from `destroy(d)` primitives per site/pair. A cheap Julia
  **unit-check** (reassemble the full operator from the emitted blocks, assert == `hamiltonian_static` to
  1e-12) catches every row/col-major or site-order bug *before* any GPU run.

- **Time-dependence — parameterized pulse library (shape_id + params), not a sampled grid.** The shape set
  is small/closed (sine, flattop, flat_top_gaussian, slepian, ccz_slepian, shaped); the optimizer varies
  only peak freqs / λ-vectors / a few harmonics, never free waveforms. Closed-form replay matches the
  adaptive Julia solver to machine precision at any t (a grid would need sub-ns density + an interp floor).
  C++ compiles `{shape,params}` into a `std::function<double(double)>` (parallels the existing
  `make_cosine_time_scalar` factory). HO coeff = `env[key](t)`; coupling coeff = `∏ √env[key](t)` (the
  operator already absorbed g at ω=1). `ccz_slepian` is the one table-driven escape hatch (export its
  exact-diagonalization (θ,ω) knots).

- **Closed vs open — by the `system` field, derived from `isempty(c_ops)`.** closed → |ψ⟩ Schrödinger
  path (`-tc_schrodinger`), evolve the 2^n_q computational kets. open → ρ Lindblad path; since Julia's
  open path hard-errors on `use_computational` (helper.jl:255), the open oracle is a full-space mesolve
  and C++ should evolve the full superoperator. Closed is validated first; open waits for real T1/Tφ.

- **Results — C++ returns raw full-length lab-frame |ψ_a(T)⟩; Julia owns the dressed projection.** Slicing
  a bare computational block in C++ would book ~4% fake leakage (BASIS_ARTIFACT.md). Returning the full
  length-D state preserves the dressing tails. C++ must back-transform out of the rotating frame first
  (`ψ_lab[k] = e^{-i E_k t_f} ψ_R[k]`, frame_omegas per the contract). v1 reuses the existing
  `PROPAGATOR_DUMP` text format (Julia parser already exists in `verify_schrodinger.py`).

---

## 4. Build order (small, independently-committable, validate-then-commit)

| # | side | what | validation |
|---|---|---|---|
| 1 | cpp | JSON deserializer → `Model` (vendor `nlohmann/json.hpp`; static terms only) | unit test: tiny 2-site JSON → `OperatorTerm.matrix` element-exact |
| 2 | julia | local-block decomposer + transpose-flatten (no I/O) | reassemble full op from blocks == `hamiltonian_static` to 1e-12 (the convention gate) |
| 3 | both | **static-only** closed round-trip (envelopes pinned to idle) | C++ `U_comp` == Julia native to ~1e-8 (wiring, time-dep factored out) |
| 4 | cpp | parameterized envelope factory (`TimeScalarFunction`) | unit test each shape vs hardcoded gate_sim.jl values |
| 5 | julia | envelope classifier + full exporter + `run_gpu_channel` + `compute!` branch | **CASE 1**: 2Q1C sine round-trip == Julia == QuTiP ~1e-8 |
| 6 | both | 2-tuned-key (CZ) + 3Q2C + mixed-dim 4Q4C-as-2Q1C | CASES 2/3/4 ~1e-8; heterogeneous dims end-to-end |
| 7 | both | **open** (deferred): dissipators + ρ path | 2Q1C + one T1 == full-space mesolve ~1e-8 |

Validation is a **three-way oracle**: Julia-native == QuTiP/QuantumToolbox full-space == Julia-exported→C++→
dressed-projection. The QuantumToolbox engine is a *second independent oracle* on top of QuTiP.

---

## 5. Decisions you need to make

| # | decision | my recommendation |
|---|---|---|
| 1 | **num_steps** (Julia adaptive Vern9 vs C++ fixed-step) — who picks the step count? | A `num_steps` (+ `rotating_frame`) const per config module, floor set by propcheck convergence data |
| 2 | frame contract — C++ back-transforms to lab before returning? | **Yes** (lab return; the bare-frame + weight-1-Z gauge is the one forbidden combo) |
| 3 | dressed-projection owner | **Option A** — C++ returns raw lab |ψ⟩, Julia projects (zero new C++ math) |
| 4 | JSON library | **vendor `nlohmann/json.hpp`** (single header; Julia `JSON.jl` writes it natively) |
| 5 | **envelope coverage for v1** — which shapes? | sine + flattop + flat_top_gaussian + slepian. **Does 4Q4C production use `ccz_slepian`?** (needs the table) |
| 6 | **T1/Tφ values, units, per-site split** (open only) | not in code anywhere — need SI seconds + the √(2/Tφ)·n convention pinned to QuantumToolbox |
| 7 | dressed vs bare jump operators (open only) | bare single-site local blocks (accept ~0.6% (g/Δ)²) keeps matrix-free; revisit if needed |
| 8 | result format | v1 `PROPAGATOR_DUMP` text; move to results-JSON later if drift bites |
| 9 | **`dressed_number_state` collision bug** (greedy mis-assign at some CCCZ idle points) | **must be confirmed fixed Julia-side before any CCZ/CCCZ round-trip** — it corrupts U_comp regardless of projection owner |

**Hard prerequisites before the *open*-system round-trip (not before closed):** real T1/Tφ/coupler-loss
values (#6) and the dressed-jump decision (#7). Closed-system validation (CASES 1–4) needs none of these.
