#pragma once

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

Model build_tunable_coupler_model(const TunableCouplerBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_tunable_coupler_cpu_benchmark(
    const TunableCouplerBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_tunable_coupler_cuda_benchmark(
    const TunableCouplerBenchmarkConfig& config);

} // namespace culindblad
