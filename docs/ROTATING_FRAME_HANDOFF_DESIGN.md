# Rotating frame for the `-load_model` (loaded-model) path — design

Goal: evolve the deserialized handoff model in the rotating frame so the GPU fixed-step RK4 is not
lab-frame-stiff (ω_max·dt < 2.83 needs ~1e5 lab steps; rotating frame → ~9–15× fewer, making the OPEN
ρ run feasible). Mirror the EXISTING validated C++ representation (build_tunable_coupler_model,
tunable_coupler_benchmark.cpp:732–796): the frame H_R = Σ ω_frame_s n_s removes the bare diagonal, and
each coupling becomes `g(t)[cos(D t)h1 + sin(D t)h2 − cos(S t)h3 − sin(S t)h4]`, D=ω_i−ω_j, S=ω_i+ω_j,
with h1–h4 the unit-g rotated 2-site operators (make_two_site_rotated_coupling_operators). The bare ω·n
diagonal is dropped (frame); only anharmonicity + (ω(t)−ω_frame)·n (pulse modulation) survive.

## Where the decomposition lives
The **Julia exporter** does the decomposition (it has the coupling structure g + the frequencies),
emitting a rotating-frame term list; the **C++** gains one coefficient kind + the back-transform; the
**solver core is unchanged** (it already evolves TimeDependentTerms). This keeps the C++ matrix-free
path generic and the conventions on the Julia side (as for the lab handoff).

## Schema additions (rotating-frame export)
- `evolution.rotating_frame = true`, `frame_omegas = [ω_frame per site]` (bare ω, site-0-major) — present.
- Diagonal terms: exporter emits the frame-subtracted diagonal — Duffing → anharmonicity only (ω_frame=ω),
  HO tuned term → (ω(t)−ω_frame)·n. (No bare ω·n.)
- Coupling terms: per coupling, 4 time-dependent terms with **unit-g** blocks h1,h2,h3,h4 and a new
  coefficient kind:
  `"coefficient": {"kind":"rf", "base":{<envelope|product_sqrt|constant>}, "freq":<D or S>, "phase":"cos"|"sin", "sign":±1}`
  → C++ value = sign · base(t) · {cos|sin}(freq·t). (Static coupling: base = constant g. Tuned coupling:
  base = product_sqrt of √ω(t), g(ω=1) folded into the unit-g block scale.)
- `evolution.result_frame = "rotating"` → C++ back-transforms to lab before dumping.

## C++ changes (small, additive)
1. **rf coefficient kind** in model_io (pulse_envelope): `make_rf_coefficient(base, freq, phase, sign)` →
   `t -> sign*base(t)*cos_or_sin(freq*t)`. base reuses make_envelope / product_sqrt / a constant.
2. **Back-transform** in dump_loaded_model_result, gated on `result_frame=="rotating"`: precompute the
   frame energy E_k = Σ frame_omegas_s · occ_s(k) (occ from k via the site-0-major strides), then
   - closed: ψ_lab[k] = e^{−i E_k T} ψ_R[k];
   - open:   ρ_lab[k,l] = e^{−i (E_k−E_l) T} ρ_R[k,l].
3. No solver change — the rf terms are ordinary TimeDependentTerms.

## Julia exporter changes
`export_model(...; rotating_frame=true)`: build the frame-subtracted diagonal + the h1–h4 coupling
decomposition (reuse the local annihilation ops; D/S from frame_omegas) + the rf coefficients. The
static-reassembly gate becomes a t-sampled check that H_R(t) (reassembled) equals the rotating-frame
H from the lab H via U(t)(H−D)U†(t) at several t.

## Segments (validate-then-commit)
1. **C++ rf coefficient + back-transform** + a hand-written small lab-vs-rotating model
   (2 non-degenerate qubits + 1 coupling): evolve lab and rotating-frame variants, confirm the
   back-transformed final states match to ~1e-9. No Julia. ← the foundation, fully testable.
2. **Julia exporter rf path** (diagonal subtract + h1–h4 + rf coeffs); reassembly check H_R(t)==U(H−D)U†.
3. **Round-trip**: export 2Q1C rotating-frame model; C++ closed at ~10k rot steps == lab at 120k
   (self-consistency) and (on a working Julia node) == U_dress_comp.
4. **Open cross-check** (the payoff): C++ open at ~10k rot steps, T1=Tφ=100µs → ρ ≈ closed |ψ><ψ| to
   ~6e-4, for 2Q1C → 3Q2C → 4Q4C. Now feasible (rotating frame).

## Validation invariants
- t=0: the rotating-frame coupling reduces to g·(h1−h3) = the lab coupling op (sanity).
- Small-model lab-vs-rotating final-state match to ~1e-9 (segment 1) — the frame transform is exact.
- The open cross-check residual ~ the decoherence floor (6e-4), not the integrator.
