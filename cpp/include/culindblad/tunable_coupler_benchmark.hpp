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

    BenchmarkStateSelection state_selection;
    Index batched_num_steps = 500;
};

// Total number of qutrit modes = num_qubits + num_couplers.
Index tunable_coupler_num_elements(const TunableCouplerBenchmarkConfig& config);

Model build_tunable_coupler_model(const TunableCouplerBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_tunable_coupler_cpu_benchmark(
    const TunableCouplerBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_tunable_coupler_cuda_benchmark(
    const TunableCouplerBenchmarkConfig& config);

} // namespace culindblad
