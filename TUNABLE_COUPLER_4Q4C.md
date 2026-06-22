# 4Q4C Tunable-Coupler Gate Simulation — Project Specification

> Scientific/engineering goal that drives the cuLindblad performance work. This is the
> *target application*: simulate a tunable-coupler superconducting-qubit device under the
> Lindblad master equation, fast enough to optimize gate pulses by repeated time evolution.

## 1. Goal

Find the **highest-fidelity gate** on a tunable-coupler architecture by **optimizing the
control-pulse parameters**. Evaluating a candidate pulse requires a full open-system
(Lindblad) time evolution of a large qutrit system; the optimizer calls this evolution
**many times**, so each evolution must be fast. This is *why* cuLindblad needs to be
accelerated.

## 2. Physical system

A **tunable coupler** architecture (see superconducting-qubit literature on tunable
couplers) with:

- **4 transmon qubits** + **4 tunable couplers** = **8 elements**.
- Each element truncated to **3 levels (qutrit)**.
- Therefore an **8-qutrit** system: Hilbert dimension `D = 3^8 = 6561`,
  vectorized density-matrix dimension `density_dim = D^2 = 43,046,721`.
- The device is **open** (T1 relaxation, T2 dephasing) → dynamics governed by the
  **Lindblad master equation**.

Index the elements as `λ ∈ {q1, q2, q3, q4, c1, c2, c3, c4}`.

## 3. Hamiltonian

The full Hamiltonian decomposes into uncoupled (bare) modes plus their pairwise
interactions. Reference form for the 2-qubit / 1-coupler case (elements `{1, c, 2}`):

```
H = H_1 + H_c + H_2 + H_1c + H_2c + H_12
```

### 3.1 Bare modes (one per element)

```
H_λ = ω_λ b_λ† b_λ + (α_λ/2) b_λ† b_λ† b_λ b_λ
```

with circuit parameters

```
ω_λ = sqrt(8 E_Jλ E_Cλ) − E_Cλ ,    α_λ = − E_Cλ
```

Note `b_λ† b_λ† b_λ b_λ = n_λ (n_λ − 1)`, so the quartic term is `(α_λ/2) n_λ(n_λ−1)`.

### 3.2 Two-body interactions — ALL-TO-ALL

Capacitive coupling between every pair of elements (qubit–qubit, qubit–coupler, AND
coupler–coupler), i.e. up to `C(8,2) = 28` pair terms:

```
H_{λμ} = g_{λμ} ( b_λ† b_μ + b_λ b_μ† − b_λ† b_μ† − b_λ b_μ )
```

**Important:** this is the **full capacitive coupling**, algebraically equal to
`H_{λμ} = − g_{λμ} (b_λ† − b_λ)(b_μ† − b_μ)`, so it **includes the counter-rotating terms**
(`b†b†` and `bb`). This is **different from** the rotating-wave nearest-neighbor coupling
`g(a_i† a_{i+1} + a_i a_{i+1}†)` used in the current `transmon_chain_benchmark` model.
The interaction graph is **all-to-all**, not a 1-D chain.

**The coupling strengths are frequency-dependent.** From the circuit parameters:

```
g_jc = (1/2)        (C_jc / sqrt(C_j C_c)) sqrt(ω_j ω_c)      (qubit j – coupler c)
g_12 = (1/2)(1 + η) (C_12 / sqrt(C_1 C_2)) sqrt(ω_1 ω_2)      (qubit 1 – qubit 2)
```

i.e. in general `g_{λμ} = K_{λμ} · sqrt(ω_λ ω_μ)`, where `K_{λμ}` is a **constant**
capacitance ratio (geometry; includes the `(1+η)` factor for qubit–qubit). Only the
frequencies `ω_λ` carry time dependence — see §3.3.

### 3.3 Time-dependent control (gates)

A gate is performed by **modulating element frequencies `ω_λ(t)`** in time — typically
the coupler frequencies, sometimes qubits too. Example envelope: a **single-sided sine/dip**
— `ω_λ(t)` starts at its idle value, dips to a minimum, and returns to the same value over
the gate duration.

Modulating `ω_λ(t)` makes **two kinds** of terms time-dependent, because the couplings are
frequency-dependent (§3.2):

1. **The bare number term** of the tuned element — `ω_λ(t) · n_λ` — a DIAGONAL, single-site
   time-dependent Hamiltonian term.
2. **Every interaction edge incident to λ** — `g_{λμ}(t) = K_{λμ} · sqrt(ω_λ(t) · ω_μ)` for
   all neighbours μ — a 2-site time-dependent term (the same fixed 4-monomial operator,
   scaled by a time-dependent coefficient). If μ is *also* being tuned, the coefficient is
   `K_{λμ} · sqrt(ω_λ(t) · ω_μ(t))`.

What stays **constant**: the anharmonicities `α_λ`, the capacitance ratios `K_{λμ}`, and any
coupling edge whose **both** endpoints are untuned.

> Example: tuning only `ω_q1` → time-dependent terms are `H_q1`'s number term, `g_12` (q1–q2),
> and `g_1c` (q1–coupler) … i.e. every edge touching q1; `g_2c` stays constant.

In the all-to-all 4Q4C graph, tuning one element makes its bare term + all **7** incident
edges time-dependent. A gate may tune **multiple couplers and/or qubits at once**, each with
its own pulse envelope; an edge between two tuned elements depends on **both** envelopes.

**Implementation note:** each time-dependent term is `scalar(t) × (fixed operator)`, so the
operator can stay GPU-resident with its cuTENSOR plan/executor reused across steps — only the
scalar `ω_λ(t)` or `g_{λμ}(t)` need be re-evaluated each RHS call (apply it in the elementwise
combine, no per-step re-upload). The bare term is diagonal (diagonal time-dependent commutator
fast path); the coupling terms are 2-site (cuTENSOR / sparse path).

### 3.4 Dissipation (open system)

Per-element **T1 relaxation** and **T2 dephasing** as Lindblad collapse operators
(relaxation `∝ b_λ`, dephasing `∝ n_λ`; same convention family as the existing code's
`√(1/T1)·a` and `√(1/Tφ)·n`).

## 4. Simulation workload

To characterize a gate, evolve a complete basis of input density matrices through the noisy
dynamics and reconstruct the logical channel:

- **16 computational states** → **256 input density matrices**. A *computational state* is a
  basis ket with **all 4 couplers in `|0⟩`** and each of the **4 qubits in `|0⟩` or `|1⟩`**
  (16 such kets). The 256 density matrices are their outer products `|i⟩⟨j|`, `i,j = 1…16`
  (the basis needed for the 4-qubit process matrix). Couplers and the 3rd (leakage) levels
  are part of the full 8-qutrit dynamics but are traced out / projected to obtain the 4-qubit
  logical process matrix.
- Evolve all 256 under the time-dependent + dissipative generator, extract the reduced
  4-qubit process matrix / effective error channel, and compute **gate fidelity** vs the
  target unitary. (Connects to the README's long-term goal of exporting reduced channels to
  tools like Stim.)

## 5. Performance & optimization loop

- One 8-qutrit density matrix ≈ `43,046,721 × 16 B ≈ 688 MB`; the working set (grouped
  scratch, cuTENSOR workspace, RK4 stage vectors) is several GB per state. All 256 states
  cannot be batched at once (256 × 688 MB ≈ 176 GB > 96 GB VRAM on the RTX 6000 Pro), so the
  256 are evolved in **chunked batched GPU solves**.
- Pulse-parameter **optimization** requires **many** such full evolutions → the single-GPU,
  per-density-matrix evolution must be made fast; batching across states is then the
  throughput multiplier.

## 6. Implications for the cuLindblad codebase

What this target needs that the current code does **not** already provide:

1. **A new model builder** (the current `build_transmon_chain_model` is a 1-D
   nearest-neighbor chain). Needs: element types (qubit vs coupler), **all-to-all** 2-body
   topology, the **full capacitive coupling** form (4 monomials incl. counter-rotating), and
   per-element `E_J/E_C → ω/α` parameters.
2. **Time-dependent terms driven by `ω_λ(t)` envelopes** (configurable single-sided-sine /
   dip, several elements tunable at once). Two kinds, both `scalar(t) × fixed operator`: the
   diagonal bare term `ω_λ(t)·n_λ`, and **every incident coupling** `g_{λμ}(t) = K_{λμ}·√(ω_λ(t)ω_μ)`
   (2-site). The model builder should take the `ω_λ(t)` envelopes as primitives and *derive*
   both the bare scalars and the coupling scalars from them. Fits the existing
   `TimeDependentTerm` machinery (always a Hamiltonian commutator); keep operators GPU-resident
   and apply the time-dependent scalar per step (no re-upload).
3. **Per-element T1/T2 dissipators**.
4. **256-state chunked batched evolution**, **reduced 4-qubit process-matrix extraction**,
   **fidelity** computation, and a **pulse-parameter optimizer** loop wrapped around it.

## 7. Validation (writing fast code must not change the physics)

Because the target (8 qutrits, 43M-dim) is far beyond any independent oracle, use a layered
plan rather than a single fixture:

- **Oracle tier (QuTiP, occasional):** validate the *kernels* at the largest QuTiP-feasible
  size that exercises the target's code paths — **qutrit local dim + an interior, all-to-all
  coupled element**. (`n3d3` is the keystone small proxy; the existing exhaustive
  verification covers n2d2/n2d3/n4d2.) Correctness of the *local* k-site kernels transfers to
  large N because only the passive dimension grows.
- **Golden-master tier (every change, up to target size):** freeze full-state dumps from an
  oracle-validated build using **fixed-step TSRK4** (bit-exact) at sizes up to n8d3.
- **Invariant tier (every change, every size incl. target, no oracle):** trace preservation,
  Hermiticity, **batch-size independence**, and **optimized-path == validated-fallback**.
- **Scale landmine:** PETSc is (currently) built without `--with-64-bit-indices`; a batched
  `Vec` of `batch_size × density_dim` overflows 32-bit `PetscInt` once
  `batch_size ≳ 49` at n8d3. Rebuild PETSc with 64-bit indices (or cap batch) and audit
  in-kernel CUDA index types.

---

*Status: project specification / intent, confirmed with the user (2026-06-22). The 256 = 16
computational states (couplers `|0⟩`, qubits `|0⟩/|1⟩`) and their outer products; and tuning
`ω_λ(t)` makes both the bare term and every incident frequency-dependent coupling
`g_{λμ}(t) ∝ √(ω_λ ω_μ)` time-dependent.*
