# Rotating-Frame / Dressed-Basis Design Notes for the Matrix-Free GPU Lindblad Solver

**Purpose.** Handoff from a design Q&A session (local Claude) to the cluster Claude that is
writing the **matrix-free, multi-node multi-GPU Lindblad master-equation solver** (tensor
contraction; intended to run the open-system T1/T2 *anchor* for the 4Q4C gate program).
This captures the questions asked, the verified answers, an **empirical check that was actually
run against the real code**, and a concrete **action-item list** at the end.

Provenance tags used below:
- **[wf-verified]** = conclusion produced and adversarially cross-checked in a multi-agent research workflow (literature + repo).
- **[measured]** = computed numerically against the real 4Q4C Hamiltonian / eigenvectors in this session.
- **[derivation]** = analytic, not separately benchmarked.

System recap: 4 transmons + 4 tunable couplers = 8 coupled bosonic modes, `n_basis=3` (dim `3^8=6561`)
or `4` (`4^8=65536`). Diabatic flux-pulse gates (iSWAP/CZ/CCZ/CCCZ), T_gate 40–300 ns (native CCCZ ~1.8–2 µs).
T1≈T2≈50 µs (assumed; see action items — **not yet set in code**). Julia stack; QuantumToolbox + DifferentialEquations.

---

## TL;DR — decisions

1. **Rotating frame to cut step count: YES, correct & standard.** Necessary, not sufficient. Use a **per-mode** frame with a **constant park-frame generator**. [wf-verified]
2. **Do NOT use fixed-step RK4.** After the frame removes the GHz carrier, use exponential / Magnus / Krylov integrators (Arnoldi, not symmetric Lanczos, for the non-Hermitian Liouvillian). Monitor trace/positivity (no explicit RK is completely positive). [wf-verified]
3. **Frame × dressed-state evolution: no fundamental conflict.** The frame (generator) and dressed injection/readout (basis) are **orthogonal axes**. The *evolution and step reduction are exact and fine.* The ONLY catch is **scoring/gauge** with the **bare-mode frame** — avoid the single bad combination "bare-mode frame scored with the weight-1-Z gauge." [wf-verified + measured]
4. **Use the dressed-H₀ frame** (interaction picture w.r.t. the full idle Hamiltonian) **OR the hybrid** (bare-mode stepping + exact dressed projection + frame-rotated target). Both are clean. [wf-verified + measured]
5. **For the open-system metric, do NOT optimize against MCWF** (stochastic objective). The program needs an *anchor* (fixed-pulse evaluation), not optimization-in-the-loop. For an in-loop open-system term, use the **deterministic Abad first-order decoherence floor**. [wf-verified]
6. **Real prerequisite:** device `T1/T2/coupler-loss` are **unspecified** (`c_ops = QuantumObject[]` in every config). No anchor produces an actionable number until these are chosen. [wf-verified]
7. **Bug found** (independent of the solver): `System.dressed_number_state` mis-assigns `|1101⟩` and `|1111⟩` at the default CCCZ idle. May be corrupting CCCZ readout right now. [measured]

---

## Q1 — "Is the rotating frame the correct method to reduce step size for the matrix-free GPU Lindblad solver? (lab-frame dynamics are at GHz, forcing tiny RK4 steps.)"

**Answer: Yes — correct and standard. It's the right first move, with caveats.** [wf-verified]

### Why it works
- Lab frame: fastest phase ~ Σ mode frequencies (tens of GHz) → period ~tens–hundreds of ps → tiny steps.
- A **per-mode** frame `V = Πₖ exp(i ωₖ nₖ t)` cancels the bare diagonal `Σ ωₖ nₖ` exactly (via the `−i V V̇†` term). Residual scales: detunings, exchange, anharmonicity (~hundreds of MHz). **Must be per-mode** — a single global frame leaves ~1 GHz residual detunings on the off-frequency modes and defeats the purpose.

### Caveats that decide whether the win is real
1. **The frame alone does NOT make H static.** Co-rotating exchange → difference frequency `exp(i(ωᵢ−ωⱼ)t)` (slow, good); counter-rotating → **sum frequency `exp(i(ωᵢ+ωⱼ)t) ~ 10 GHz`** (faster than the carrier). The configs set `en_counter_rotating_terms = true`, so these survive. Either drop them (RWA) — but `g/Σ ~ 10⁻²` so the RWA error `~(g/Σ)² ~ 10⁻⁴` is **borderline at the program's precision; validate per gate** — or use a Magnus-class integrator that tolerates oscillatory H.
2. **Time-dependent flux ⇒ use a CONSTANT generator** (idle/park frequencies). Building the frame from the *instantaneous* flux-modulated frequencies `ωₖ(t)` adds a spurious `−t·(dωₖ/dt)·nₖ` Coriolis term that corrupts conditional phases. (Integral-phase `Θ=∫ω` is also valid but the constant park-frame is the safe default.)
3. **Not fixed-step RK4.** Once the stiff fast linear part is gone, exponential / commutator-free Magnus (CFQM) / Krylov action-of-exponential are the right tools (they resolve the residual oscillation per step and stay structure-preserving). For the **open-system Liouvillian** use **Arnoldi** (non-Hermitian generator), not symmetric Lanczos. **No explicit RK is completely positive** — monitor `Tr ρ`, Hermiticity, and min-eigenvalue; for hard CPTP guarantees use a Rouchon/Kraus-style step. Anharmonicity `αₖ nₖ(nₖ−1)/2` becomes a **constant** in the rotating frame → an exponential integrator absorbs it exactly (RK4 still has to resolve it).
4. **GPU note:** adaptive step-rejection needs a host-side error-norm reduction each step (serializes kernels at D=65536). A **fixed-step exponential/Magnus** with an offline-tuned `h` keeps a static kernel graph *and* is structure-preserving — usually better than fixed RK4 on both axes.

### Memory / architecture
- ρ is **0.69 GB (d=3) / 69 GB (d=4)** in complex128 (× ~5 for stage buffers). d=3 fits one GPU; **d=4 full-ρ forces the multi-node plan.**
- The *dense* Liouvillian superoperator is `D⁴` (~30 PB even at d=3) → **matrix-free tensor contraction is mandatory**, not optional. Cost of applying `L[ρ]` as reshaped few-body contractions is `~D²·dᵏ` per k-body term, not `D⁴`.
- **Fork: full-ρ vs MCWF (quantum trajectories).** MCWF evolves state vectors (memory **D not D²**, ~1 MB/state at d=4) and is embarrassingly parallel across nodes/GPUs — but its statistical error on `(1−F)` is weakest exactly in the `F > 0.999` regime (most trajectories have zero jumps). See Q2.

### Collapse operators in the rotating frame
- Single-mode T1 (`a`), pure-dephasing (`n`), and heating (`a†`) are **frame-invariant** under a per-mode diagonal frame (`D[e^{iφ}L]=D[L]`; `n` commutes with the generator). **Multi-mode/correlated** c_ops (e.g. coupler-mediated loss) are **not** — they gain `e^{±i(ωᵢ−ωⱼ)t}`. Assert each c_op equals itself up to a global phase under the frame.
- ⚠ Jump operators must be defined in the **dressed eigenbasis**, not bare modes (eigenoperators of the free propagator; dressed-state relaxation occurs at the dressed splitting ~2g — arXiv:2601.11316). Bare-`a` vs dressed-`a` is first-order on this strongly-dressed system.

### Numerical precision
- Accumulated phase `ωt` over ~100 ns is ~3000 rad. In float32 the ULP there is ~2×10⁻⁴ — at the 10⁻⁴ fidelity target. **Keep the frame reference phase in float64** even if states are float32.

### Coherence floor (decision-relevant)
- At T1=Tφ=50 µs the channel-summed floor is ~1.6×10⁻³ (40 ns) → ~1.2×10⁻² (300 ns), and **~4×10⁻² for the ~1.8 µs native CCCZ** vs ~1×10⁻² for an ~450–600 ns ancilla path. So the anchor can **flip native-vs-ancilla** — that is *why* it is owed.

---

## Q2 — "MCWF has problems: noisy evaluation, many trajectories per point, more samples near convergence, optimizer efficiency drops on a non-smooth objective. Total optimization wall time can be worse. Right?"

**Answer: Right — for MCWF *as an optimization objective*. But that's a use case this program doesn't have, and the better fix is a deterministic objective.** [wf-verified]

- **The objection is correct in-the-loop.** Finite-difference / Nelder-Mead degrade on a stochastic objective (FDSA converges `k^(-1/3)` vs `k^(-1/2)`). Resolving `(1−F)~ε` to 10% relative needs `N ~ 100/ε` trajectories **per point** — `~10⁵` at ε=10⁻³, `~10⁶` at ε=10⁻⁴ (rare-jump-dominated). Confirmed in the literature: trajectory-based open-system control (Abdelhafez & Schuster, PRA 99, 052327) needs SGD + improved sampling *precisely* to fight this noise.
- **But the program's open-system run is an ANCHOR** = evaluate an already-coherently-optimized pulse once, with error bars. At a fixed pulse there is no optimizer loop → the four concerns don't apply (throw a large fixed N at one point, report `F ± CI`).
- **Why design and anchor separate cleanly:** Abad et al. (PRL 129, 150504, 2022) — to first order in weak dissipation, the fidelity reduction is **operation-independent**, depending only on gate time and the dissipation. So re-optimizing pulse *shape* against T1/T2 at fixed duration buys ~nothing; decoherence adds a **duration penalty**, it does not relocate the coherent optimum. **Exception:** gates that leave the computational subspace (your leaky CCZ/CCCZ) — the coefficient then depends on the |2⟩-dwell via `L(t)=U†(t) L U(t)`, so decoherence *can* move the optimum there.
- **Best in-loop open-system objective = the deterministic Abad first-order floor:**
  `1 − F_open ≈ (1 − F_closed) + Σ_k Γ_k ∫₀^τ δF(t, L_k(t)) dt`, with `L_k(t)=U†(t)L_k U(t)` from the **closed-system propagator you already compute**. Smooth, deterministic, **memory D not D²**, error `O((Γτ)²) ~ 10⁻⁵–10⁻⁶` in this regime. Published, leakage-aware formulas exist for **CZ (Eq. 12), CCZS three-qubit (Eq. 17), simultaneous-CZ four-qubit (Eq. 19)** in arXiv:2302.13885 (Quantum 9, 1684, 2025) — adapt, don't re-derive.
  - **Correction to an earlier claim:** large transient leakage does **not** break first order; the small parameter is `Γτ`, not the leakage fraction (all coherent leakage is in `L(t)`). What leakage breaks is the *operation-independent universality* → leaky gates need the gate-specific `δF/δL`, not a flat `Γ·T_gate`.
- **Gradient-based open-system control** (if ever wanted) = deterministic adjoint/GRAPE/Krotov on `mesolve` (dynamiqs GPU+AD), **not** trajectories — the master-equation objective is smooth and differentiable; D² memory is the price.
- **If MCWF must be in a loop:** common random numbers (fix jump seeds → piecewise-smooth objective, restores finite-difference gradients), SPSA (2 evals/gradient, dimension-independent), and jump-class / deterministic-jump variance reduction (~10–1000× in the weak-dissipation limit: compute the no-jump branch once + the few single-jump branches; two-jump weight `<10⁻²`).
- **Real prerequisite:** every config has `c_ops = QuantumObject[]` and no T1/T2 specified. Until device `T1`, `Tφ`, and the |2⟩/coupler-T1 ratios are chosen, **no anchor — deterministic or stochastic — produces an actionable number.** Also: **1/f flux noise is non-Markovian** and dominant for flux-tunable couplers; captured by *neither* constant-Γ MCWF *nor* the Markovian Abad formula, and worst at the deep CCCZ coupler dives.

**Decision table**

| Objective | Memory | Smooth? | Use for |
|---|---|---|---|
| Perturbative (Abad) | D | ✓ deterministic | in-loop term + screening anchor (default) |
| Full-ρ adjoint/GRAPE | D² (69 GB @ d=4) | ✓ differentiable | gradient control; headline anchor at d=3 |
| MCWF | D, parallel | ✗ noisy | final fixed-pulse anchor only, large N once |

---

## Q3 — "We currently evolve dressed states. Will that have a problem with the rotating-frame step-count reduction?"

**Answer: No fundamental conflict. Frame (generator) and dressed injection/readout (basis) are orthogonal axes.** The current engine already evolves dressed idle-H₀ eigenstates in the bare *lab* frame (`helper.jl:65-72` integrates `dψ=-iH(t)ψ`; dressed states enter only as initial conditions `Dcomp` at `:163-172` and readout overlaps `Umat[i,k]=⟨dressed_i|ψ⟩` at `:199-209`). A frame is just `Ũ = V(T)†·U·V(0)` with `V` a fixed unitary — fidelity is preserved **iff the same V is applied to both injection and readout.** [wf-verified + measured]

### Three regimes
- **(A) Bare-mode frame `V=exp(iDt)`, `D=Σ ωₖ nₖ` (idle/park freqs).** Cancels only the bare diagonal; exchange survives → dressed idle state **not** stationary; step cut from GHz to the **difference/dressing scale (~0.14–1.4 GHz, ~10×)**, *not* pulse bandwidth, and only with the CRT (sum) terms dropped. **Cheap for matrix-free** (per-mode diagonal phase, sparsity preserved). **BUT** the frame phase is diagonal in the *bare* basis, not the dressed readout basis → it is **not** a clean weight-1-Z there (see measured results below).
- **(B) Dressed-H₀ frame `V=exp(iH₀ t)`.** Cancels the entire idle evolution incl. exchange AND the counter-rotating terms (no RWA needed) → dressed idle states **exactly stationary**, step set by **pulse bandwidth (max reduction)**, frame phase `exp(i E_dressed,k T)` **exactly diagonal in the dressed basis** → weight-1-Z gauge stays clean (residual ZZ is *genuine physics*). **Cost:** `U_transform` is dense (`system.jl:1600`), so working in the dressed eigenbasis **densifies** controls/couplings/c_ops (~100× at 1% H-density; sparse ladder ops blow up ~10⁴×; at d=4 a single dense control ≈ 69 GB → **memory wall**).
- **Hybrid (recommended for closed-system fidelity).** Step in the cheap **bare-mode frame**, apply the frame phase as a diagonal multiply, then project onto dressed states with **exact inner products** (`helper.jl:201-209` already does `dot(Dcomp, ψ)` — no weight-1 assumption). Score against the **frame-rotated target** `P_bare(T)·U_ideal`. Exact at any dressing strength; keeps sparsity. The only thing you give up: the weight-1-Z *shortcut* and absolute pre-gauge leakage accounting.

### Avoid
- Frame built from the **instantaneous** flux-modulated `ωₖ(t)` (Coriolis trap).
- The **fully-instantaneous HOQST true-adiabatic** frame — these gates are *diabatic*, so the nonadiabatic coupling `−iR†Ṙ` *is* the gate mechanism: little step benefit, time-dependent readout basis.

---

## The empirical check that was run (the "5-minute check")

**Goal:** at a real CCCZ operating point, materialize the bare-mode frame phase
`P_bare(T) = Diag(exp(i Σₖ ωₖ nₖ T))` and measure how far it is from a clean
weight-1-Z in the **dressed readout basis** — i.e. the gauge-unfixable error the
bare-mode frame would inject. Scripts left in the repo:
- `cccz_pbare_check2.jl` — the **robust, trustworthy** version (build the dressed comp subspace from the top-16 comp-weight eigenvectors of `H_idle`, align to bare comp labels — robust to degeneracy and to the lookup bug below — then measure P_bare contamination).
- `cccz_pbare_debug.jl` — per-state diagnostics that exposed the `dressed_number_state` bug.

### Results [measured]
- **Idle (readout) dressing is MILD:** dressed-comp self-overlaps min **0.984** / avg **0.992**; comp subspace cleanly separated (top-16 min comp-weight 0.968; 17th eigvec 0.014 — clean gap). `|1111⟩` *is* well-defined at idle, as the config intends.
- **Dive dressing is STRONG:** `minov` at couplers→5.18 GHz = **0.185 (prob) / 0.43 (amp)** — reproduces the reported 0.12–0.61. **This is a *dynamics* property, separate from the readout-gauge question.**
- **Bare-mode frame phase in the dressed comp basis (gauge-unfixable contamination):**

| T (ns) | min\|M_kk\| | max\|M_ab\| (a≠b) | subspace leak (avg) | leak (worst, 1−σ_min²) |
|---|---|---|---|---|
| 40   | 0.966 | 6.6e-3 | **3.4e-2** | 6.7e-2 |
| 100  | 0.972 | 3.8e-3 | 2.8e-2 | 5.5e-2 |
| 150  | 0.967 | 5.3e-3 | 3.3e-2 | 6.6e-2 |
| 250  | 0.953 | 8.4e-3 | **4.8e-2** | 9.4e-2 |
| 300  | 0.971 | 7.2e-3 | 2.9e-2 | 5.8e-2 |
| 1800 | 0.981 | 4.6e-3 | 1.9e-2 | 3.7e-2 |

  → Gauge-unfixable contamination **~3–5% typical, up to ~9% worst-case = 10–30× over a 0.997 gate's 3e-3 budget.** Persists at all T (still ~1.9% at the 1.8 µs native CCCZ).

- **Mechanism (clean):** the bare-mode frame converts the comp states' **idle coupler/|2⟩-admixture probability (~2–3% = 1 − purity)** directly into scrambled leakage, because coupler frame phases `ω_C·T` are huge (~2π·6 GHz·T, thousands of radians) — even a 0.99-self-overlap ("weak") state has its ~0.14-amplitude admixture rotated to an effectively random phase. So **bare-frame leakage ≈ (1 − idle comp purity)**, independent of how weak the dressing looks.
- **Why the hybrid still works:** that leakage appears identically in the computed gate *and* in the frame-rotated target `P_bare·U_ideal`, so it **cancels** when you score against the frame-rotated target. It only bites if you try to absorb `P_bare` with the weight-1-Z gauge (you can't) or read absolute leakage before accounting for the frame.
- **Dressed-H₀ frame:** clean by construction — its frame phase is exactly diagonal in the dressed basis (zero subspace leakage; off-diagonal verified ~8.6e-16 analytically/in prior cross-check), and its non-weight-1 content is *genuine ZZ*.

### Final confirmation (the user's closing question)
> "So dressed-basis evolution will NOT have a problem with the rotating-frame step-count reduction — correct?"

**Correct — with one qualifier about scoring, not evolution:**
- **Evolution + step reduction: no problem** (exact unitary change of variables; works for either frame).
- **The only catch is the gauge/scoring of the *bare-mode* frame.** Avoid the one bad combination (raw bare-mode frame scored with the weight-1-Z gauge). Use the **dressed-H₀ frame** *or* the **hybrid** and you are fully clean.

---

## ⚠ Bug found (independent of the solver) [measured]

`System.dressed_number_state` **mis-assigns two computational states at the default CCCZ idle**:
- `|1101⟩` → returned state has self-overlap **0.0002** with the bare `|1101⟩` (peaks on a Q4=2, C3=1 leakage state).
- `|1111⟩` → self-overlap **0.0119**.

A correct assignment exists (independent eigen+greedy finds these eigenvectors at 0.984/0.988 overlap), so this is a **greedy collision in `calc_dressed_bases` / `dressed_idx`**, not physics. `helper.jl` uses `dressed_number_state` to build both the injected `Dcomp` states and the readout bras → this **may be corrupting the CCCZ `|1111⟩` result right now** (and `|1111⟩` is the entire target of the CCCZ). Fix with a collision/degeneracy-aware assignment (e.g. Hungarian max-overlap over the comp subspace). **Verify whether production CCCZ runs hit this same idle point.**

---

## Action items for the GPU solver

1. **Frame.** Implement the **dressed-H₀ frame** (interaction picture w.r.t. full idle H₀) as the default for the open-system anchor — clean gauge + max step reduction. Implement it by working in the H₀ eigenbasis (`H₀ → diag(E)`, frame = diagonal phase); transform the pulse-deviation operators and c_ops into that basis once. Accept the denser per-step matvec (and the d=4 memory cost — see #6). Offer the **hybrid** (bare-mode stepping + exact dressed projection + frame-rotated target) where sparsity matters and you only need fidelity. **Never** the raw bare-mode frame scored with the weight-1-Z gauge.
2. **Generator.** Constant park-frame (idle H₀), never the instantaneous flux-modulated frequencies.
3. **Integrator.** Not fixed-step RK4. Use exponential / commutator-free Magnus / Krylov action-of-exp; **Arnoldi** for the non-Hermitian Liouvillian. Monitor `Tr ρ`, Hermiticity, min-eigenvalue; trace-renormalize per step; CPTP-guaranteed step (Rouchon/Kraus) for headline numbers. On GPU prefer a **fixed-step exponential/Magnus** (static kernel graph) over adaptive RK.
4. **Counter-rotating terms.** Configs have `en_counter_rotating_terms=true`. The dressed-H₀ frame removes them for free. If you instead use the bare-mode/hybrid path and want the full step win, validate the RWA per gate (`(g/Σ)² ~ 10⁻⁴` is borderline).
5. **Open-system metric.** Default to the **deterministic Abad first-order floor** for any in-loop term (memory D, smooth; formulas: arXiv:2302.13885 Eqs. 12/17/19). Use **MCWF or full-ρ only as a one-shot fixed-pulse anchor**, never as an optimizer objective. Jump operators in the **dressed eigenbasis**; single-mode `a`/`n`/`a†` are frame-invariant, multi-mode are not.
6. **Memory.** d=3 full-ρ = 0.69 GB (one GPU); d=4 full-ρ = 69 GB (multi-node). Dense superoperator is hopeless at any d → matrix-free mandatory. For d=4 strongly consider **batched MCWF** (state vectors, embarrassingly parallel) — but mind the `(1−F)` statistical weakness at F>0.999 (use jump-class/importance weighting or the perturbative floor).
7. **Precision.** Frame reference phase in **float64** (ωt ~ 3000 rad; float32 ULP ~2e-4 ≈ the target).
8. **Fix `dressed_number_state` mis-assignment** (|1101⟩, |1111⟩) — Hungarian/collision-aware assignment. Check production CCCZ idle.
9. **Set device T1/Tφ/coupler-loss** — currently empty everywhere; nothing is actionable until chosen. Note 1/f flux noise is non-Markovian (not captured by constant-Γ Lindblad) — flag for the deep-dive coupler regimes.
10. **Validate** with a d=3 `mesolve`-vs-`mcsolve` oracle on the **3Q2C block** (full 8-mode d=3 is near stock-library ceiling), and re-run `cccz_pbare_check2.jl` after any frame/readout change.

---

## References (verified arXiv IDs unless noted)
- Rotating-frame / interaction-picture, adiabatic frame for large ODE steps: HOQST (Comm. Phys. 2022, arXiv:2011.14046).
- Interaction frame of the full drift = dressed basis: arXiv:2003.12096 (npj QI 2021) — method consistent; exact quotation not verified verbatim.
- Commutator-free quasi-Magnus integrators: arXiv:2403.13889 (Quantum 8, 1567, 2024); Blanes–Moan / Alvermann–Fehske CFET; Al-Mohy–Higham `expmv`.
- First-order weak-decoherence fidelity (universality): Abad, Fernández-Pendás, Kockum, Johansson, PRL 129, 150504 (2022), arXiv:2110.15883.
- Leakage-aware decoherence-floor formulas (CZ/CCZS/sim-CZ): Abad et al., Quantum 9, 1684 (2025), arXiv:2302.13885.
- Trajectory-based open-system control needs SGD: Abdelhafez & Schuster, PRA 99, 052327 (2019), arXiv:1901.05541.
- Jump-class / deterministic-jump variance reduction: arXiv:2502.00764; arXiv:2603.04066.
- Dressed-state relaxation at 2g (jump ops must be dressed): arXiv:2601.11316.
- Common random numbers for stochastic optimization: Kleinman, Spall & Naiman, Manag. Sci. 45(11):1570 (1999). SPSA: Spall, IEEE TAC 37(3):332 (1992).

*Generated from a local design session, 2026-06-23. Empirical numbers measured against the 4Q4C repo at the default `config_4Q4C_cccz_ring.jl` idle.*
