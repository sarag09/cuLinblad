# GRAPE / adjoint-gradient design for the cuLindblad GPU solver (DRAFT)

Roadmap step 4 (after the Julia merge). Goal: gradient-based pulse optimization where the gradient
`dF/dε` over ALL pulse parameters costs ~one extra evolution (the adjoint), not `N_params` forward
solves. Prefer the **closed-system adjoint** (state vectors, cheap, memory `D` not `D²`); use the
deterministic Abad first-order decoherence floor for in-loop open terms; reserve the full-Lindblad
adjoint for leaky gates only.

Architecture: the C++ side is a **gradient service** in the `use_gpu` handoff (it already returns
ψ(T); GRAPE adds `dF/dε`). The Julia side keeps owning the optimizer loop, the target unitary, and
the dressed projection. So C++ returns the costate-projected gradient pieces; Julia assembles the
fidelity + gradient and steps the optimizer. This keeps the gauge/target/dressed conventions on the
Julia side (as for the forward handoff) and avoids re-encoding them in C++.

## 1. Closed-system adjoint (the math)

State: `H(t,ε) = H0 + Σ_i c_i(t,ε) H_i` (the existing matrix-free TD model; `c_i` are the
envelope/product-√ coefficients). For each computational initial ket `k`, forward
`i ψ̇_k = H(t) ψ_k`, `ψ_k(0)=|k⟩`. The propagator columns are `ψ_k(T)`.

Gate overlap (Julia-side, in the dressed comp basis after back-transform):
`m = Σ_k ⟨χ_k^T | ψ_k(T)⟩` where `χ_k^T = U_target |k⟩` projected to the dressed comp subspace; gate
fidelity `F = (|m|² + Σ_k⟨ψ_k|P|ψ_k⟩)/(N(N+1))` (Pedersen-style; exact form is the Julia scorer's).

Gradient via the adjoint. Define the **costate** `χ_k` with terminal condition `χ_k(T) = (∂F/∂ψ_k(T)*)`
(from the overlap; for `F=|m|²/N²`, `χ_k(T) = (m*/N²)·χ_k^T`). Evolve χ_k BACKWARD under the same H
(`i χ̇_k = H(t) χ_k`, integrated T→0). Then

  `dF/dε_j = 2 Re Σ_k ∫_0^T ⟨χ_k(t)| (∂H/∂ε_j) |ψ_k(t)⟩ (-i) dt`,
  with `∂H/∂ε_j = Σ_i (∂c_i/∂ε_j) H_i`.

`∂c_i/∂ε_j` is analytic per envelope (e.g. `∂[sine]/∂a = sin(πt/T)`, `∂/∂b = 1-sin(πt/T)`; slepian
λ via the chain rule through θ→ω_c; flat-top via its plateau params). Only the H_i that depend on ε_j
contribute. The integral is accumulated along the (shared) RK4 grid.

## 2. C++ implementation plan (reuse the matrix-free apply + custom RK4)

The forward `H|ψ⟩` apply (ts_rhs_function_cuda_full_model_schrodinger) and the custom RK4 are reused
verbatim for both the forward and backward solves — the adjoint H is the same operator (Hermitian).

1. **Forward + checkpoint.** Run the closed forward solve (evolve_state_vector_batch_cuda_ts) but
   store ψ_k at each RK4 step into a trajectory buffer (batch·D·num_steps complex; for 4Q4C closed
   D=6561, num_steps~1e3, batch=16 → ~3.4 GB — fits; or checkpoint every M steps + recompute, since
   the closed system is norm-preserving and cheaply reversible). A new `evolve_state_vector_forward_
   checkpointed` returning the trajectory handle.
2. **Terminal costate.** Build χ_k(T) from the overlap (Julia passes `χ_k^T` = target columns; C++
   forms the terminal condition, or Julia passes χ_k(T) directly). Keep it a thin C++ step.
3. **Backward solve + gradient accumulate.** A new `evolve_costate_backward_accumulate`: step χ_k
   backward; at each step evaluate, for each ε_j, `g_j += 2 Re Σ_k ⟨χ_k| ∂H/∂ε_j |ψ_k⟩(-i)` using the
   stored ψ_k(t). `∂H/∂ε_j|ψ⟩` reuses the per-term apply with the coefficient replaced by
   `∂c_i/∂ε_j` — a new batched kernel `apply_dH_deps` (mostly the existing term-apply with a different
   scalar). Accumulate g_j on the GPU (one reduction per param per step).
4. **Pulse-derivative coefficients.** Extend the envelope factory (pulse_envelope.cpp) with
   `make_envelope_dparam(spec, which)` returning `∂c/∂ε` as a TimeScalarFunction (analytic per shape).
   Thread the param→(term_i, which) map from the JSON (a new `grad_params` block listing which
   envelope params are free + which terms they touch).
5. **Return** `{F-pieces, dF/dε_j}` in the results file; Julia assembles F + gradient and steps
   (LBFGS / the existing Julia optimizer).

## 3. Validation (finite-difference gate)

For each free param ε_j: `dF/dε_j ≈ (F(ε_j+δ) - F(ε_j-δ))/(2δ)` from two forward solves (already have
the forward path). Require `|FD - adjoint| < 1e-6` (relative) on 2Q1C first (the sine/flattop case),
then 3Q2C. This is the analogue of validate_load_model.py for gradients; build it before trusting any
optimizer step. Start with a 1-qubit diagonal case (`H(t)=f(t)n`) where `∂φ/∂a = -∫ sin(πt/T) dt`
closed-form — a unit test for the adjoint machinery decoupled from the gate scorer.

## 4. Open-system in-loop (deterministic, NOT MCWF)

Per the roadmap, do NOT put the full Lindblad adjoint (D² + checkpointing, expensive) in the loop for
non-leaky gates. Use the **Abad first-order decoherence floor** (arXiv:2302.13885 Eqs 12/17/19):
a deterministic, smooth, memory-`D` correction to the closed-system fidelity capturing the leading
T1/Tφ infidelity, differentiable alongside the closed adjoint. Only for LEAKY gates (where the open
optimum moves) fall back to the full-Lindblad adjoint (ρ, D², checkpointed) — and the open ANCHOR
(fixed final pulse, run a handful of times) stays the multi-GPU state-sharded Lindblad solve, not the
loop.

## 5. Why closed-first / cost

Closed adjoint = 2 state-vector solves (forward + backward) + one gradient reduction per step,
independent of N_params — vs N_params forward solves for finite differences. At 4Q4C closed (D=6561)
this is ~2× a forward evolve regardless of how many pulse knobs, making many-parameter (slepian-λ +
per-mode) optimization tractable. The matrix-free apply, custom RK4, rotating frame, and the lab
back-transform are all reused; the only new GPU code is the checkpointed forward, the backward costate
solve, the `apply_dH_deps` kernel, and the per-param gradient reduction.

## Implementation segments (when GRAPE is taken up)
1. `make_envelope_dparam` + a unit test vs analytic/FD per shape (cpp, no solver).
2. `apply_dH_deps` batched kernel + a unit test vs `(H(ε+δ)-H(ε-δ))/2δ` applied to a vector.
3. Checkpointed forward + backward costate solve (reuse RK4); norm/adjoint sanity.
4. Gradient assembly + the FD gate on the 1-qubit diagonal case (analytic), then 2Q1C.
5. Wire into the handoff results (`grad_params` in/`dF/dε` out); Julia optimizer consumes it.
6. Abad floor for the in-loop open correction.
