#pragma once

#include <string>
#include <vector>

#include "culindblad/model.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/transmon_chain_benchmark.hpp"  // BenchmarkStateSelection, TransmonChainBenchmarkTiming
#include "culindblad/types.hpp"

namespace culindblad {

// N-qubit / M-coupler tunable-coupler benchmark model.
//
// The device is a set of (num_qubits + num_couplers) transmon-like modes, each
// truncated to cutoff_dim levels. The interaction is ALL-TO-ALL (every element
// pair couples) regardless of the physical topology; the topology (linear vs
// square) only changes the per-pair coupling STRENGTH on real hardware, which we
// ignore for the benchmark (uniform g on all pairs). square_topology is metadata.
//
// Per-element bare Hamiltonian:   H_l = omega_l b_l^dag b_l + (alpha_l/2) b_l^dag b_l^dag b_l b_l
// Pairwise capacitive coupling:   H_lm = g (b_l^dag b_m + b_l b_m^dag - b_l^dag b_m^dag - b_l b_m)
// Dissipators (per element):      sqrt(1/T1_l) b_l   and   sqrt(1/Tphi_l) n_l
//
// This builder produces a STATIC model (no time-dependent frequency drive); it is
// the model used to verify the C++ solver against QuTiP. Time-dependent omega_l(t)
// gate pulses are a later addition.
struct TunableCouplerBenchmarkConfig {
    Index num_qubits = 0;
    Index num_couplers = 0;
    Index cutoff_dim = 0;

    // Metadata only: 4Q4C is a square (ring) topology, the smaller systems are
    // linear. Does not affect the benchmark Hamiltonian (all-to-all uniform g).
    bool square_topology = false;

    // Per-element parameters, indexed by global site 0..(num_qubits+num_couplers-1).
    std::vector<double> omega;
    std::vector<double> alpha;
    std::vector<double> t1;
    std::vector<double> tphi;

    // Uniform all-to-all coupling strength (benchmark: strengths are not
    // distance-weighted).
    double g = 0.0;

    double t0 = 0.0;
    double tfinal = 1.0;

    // Time-dependent gate drive: each driven element's frequency is modulated as a
    // single-sided sine, omega_k(t) = omega_k - A*sin(pi*t/T_gate). This makes the
    // driven element's bare number term time-dependent AND (because g = K*sqrt(w_l w_m))
    // every coupling edge incident to a driven element time-dependent. driven_sites
    // are global site indices (qubits occupy 0..num_qubits-1, couplers the rest).
    bool enable_drives = false;
    std::vector<Index> driven_sites;
    double drive_amplitude = 0.0;   // A
    double drive_t_gate = 0.0;      // T_gate (<=0 means use tfinal)

    // Evolve in the rotating frame of the bare idle frequencies H_R = sum_i omega_i n_i.
    // This is an EXACT time-dependent unitary basis change (no approximation): the fast
    // bare oscillation omega_i*n_i is removed (only the slow anharmonicity, drive
    // modulation, and detuning/sum-frequency-modulated couplings remain), so the
    // fixed-step RK4 integrator reaches a given accuracy in far fewer steps. The lab-frame
    // result is recovered by the diagonal phase rho_lab(t)_kl = exp(-i(E_k-E_l)t) rho_R_kl,
    // E_k = sum_i omega_i * occ_i(k). Dissipators are frame-invariant (the rotation phase
    // cancels in L rho L^dag and L^dag L). The reference frequencies are config.omega.
    bool enable_rotating_frame = false;

    // Rotating-wave approximation: in the rotating frame, drop the counter-rotating coupling
    // terms (H3,H4, oscillating at the sum frequency omega_i+omega_j ~ 2*omega ~ 10). They are
    // g-small but FAST, so dropping them lets the fixed-step integrator take much larger steps
    // -- at the cost of an O((g/Sigma)^2) ~ 1e-4 physics error. Only meaningful with
    // enable_rotating_frame. Used to measure the step reduction from RWA.
    bool drop_counter_rotating = false;

    BenchmarkStateSelection state_selection;
    Index batched_num_steps = 500;
};

// Total number of qutrit modes = num_qubits + num_couplers.
Index tunable_coupler_num_elements(const TunableCouplerBenchmarkConfig& config);

// Flat density-matrix indices of the COMPUTATIONAL-subspace basis density matrices
// |ket_a><ket_b|: qubits (sites 0..num_qubits-1) in {0,1}, couplers (the remaining
// sites) in |0>. Returns (2^num_qubits)^2 indices in (a,b) lexicographic order over
// the 2^num_qubits computational kets (qubit 0 = most-significant bit). Site-0-major.
std::vector<Index> tunable_coupler_computational_density_indices(
    Index num_qubits, Index num_couplers, Index cutoff_dim);

// Hilbert-space (D-space) indices of the computational basis KETS |a> (qubits in {0,1},
// couplers in |0>), 2^num_qubits entries. The state-vector analogue of the density version.
std::vector<Index> tunable_coupler_computational_state_indices(
    Index num_qubits, Index num_couplers, Index cutoff_dim);

// Mixed-dim (heterogeneous local_dims) computational-index generators for the loaded-model handoff
// (same Q1=MSB, site-0-major convention). Qubits are sites 0..num_qubits-1.
std::vector<Index> computational_state_indices_for_dims(
    const std::vector<Index>& local_dims, Index num_qubits);
std::vector<Index> computational_density_indices_for_dims(
    const std::vector<Index>& local_dims, Index num_qubits);

// Evolve a deserialized handoff Model in the LAB frame (matches the Julia reference; closed = |psi>
// Schrodinger, open = rho Lindblad). If explicit_states is non-empty it is used as the initial states
// (e.g. the dressed comp kets exported from Julia): closed evolves them as |psi>, open evolves the
// outer products |k><k|. Otherwise the selected computational kets / density columns are used.
// Uses the -tc_integrator dispatch (pass -tc_integrator 1 for the custom RK4).
TransmonChainBenchmarkTiming evolve_loaded_model_schrodinger(
    const Model& model, const std::vector<Index>& selected_indices,
    const std::vector<std::vector<Complex>>& explicit_states,
    double t0, double tfinal, int num_steps);
// Checkpoint/restart config for the long OPEN (Liouvillian) custom-RK4 runs (-tc_integrator 1). When
// `every > 0` and `path` is set, evolve_batch_custom snapshots the batched density state to `path`
// (atomically) every `every` steps; on (re)start evolve_loaded_model_liouvillian AUTO-RESUMES from
// `path` if a checkpoint is present (a strict header guard hard-aborts on ANY config mismatch --
// dtype, shape, shard identity, step window -- so a wrong checkpoint can never corrupt the anchor).
// Same command both writes and resumes: absent file -> fresh start; present+valid -> resume. Disabled
// by default (every == 0), so all non-checkpointed callers are unaffected.
struct BatchCheckpoint {
    Index       every = 0;        // write cadence in RK4 steps; 0 disables checkpointing
    std::string path;             // per-rank checkpoint file, e.g. <dir>/<tag>_rank<r>.ckpt
    int         shard_index = 0;  // MPI rank  (guarded on resume)
    int         shard_total = 1;  // MPI size  (guarded on resume)
    // Runtime shape/window, filled by evolve_loaded_model_liouvillian before use (guard fields):
    Index       hilbert_dim = 0;  // D
    Index       batch = 0;        // density matrices in this rank's shard
    Index       total_steps = 0;  // num_steps this checkpoint belongs to
    double      t0 = 0.0;
    double      tfinal = 0.0;
};

TransmonChainBenchmarkTiming evolve_loaded_model_liouvillian(
    const Model& model, const std::vector<Index>& selected_density_indices,
    const std::vector<std::vector<Complex>>& explicit_kets,
    const std::vector<std::vector<Complex>>& explicit_density_states,
    double t0, double tfinal, int num_steps,
    const BatchCheckpoint* ckpt = nullptr);

Model build_tunable_coupler_model(const TunableCouplerBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_tunable_coupler_cpu_benchmark(
    const TunableCouplerBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_tunable_coupler_cuda_benchmark(
    const TunableCouplerBenchmarkConfig& config);

// Closed-system (Schrodinger) run: evolves the computational basis |psi> through the gate via
// the matrix-free state-vector solver. final_states are length-hilbert_dim |psi(T)> vectors.
TransmonChainBenchmarkTiming run_tunable_coupler_schrodinger_cuda_benchmark(
    const TunableCouplerBenchmarkConfig& config);

// Closed-system Schrodinger evolve self-test: evolves a random |psi> via the state-vector solver
// and rho0=|psi><psi| via the (validated) closed density solver, then checks rho(T) == |psi(T)>
// <psi(T)| and norm conservation. Returns 0 on PASS. Self-contained (no QuTiP).
int run_psi_evolve_schrodinger_selftest();

} // namespace culindblad
