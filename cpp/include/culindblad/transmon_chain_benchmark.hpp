#pragma once

#include <vector>

#include "culindblad/model.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct BenchmarkStateSelection {
    bool evolve_all_states = false;
    std::vector<Index> selected_state_indices;
};

struct TransmonChainBenchmarkConfig {
    Index num_transmons = 0;
    Index cutoff_dim = 0;

    std::vector<double> omega;
    std::vector<double> alpha;
    std::vector<double> g;
    std::vector<double> t1;
    std::vector<double> tphi;

    double drive_amplitude = 0.0;
    double drive_sigma = 1.0;
    double drive_center = 0.0;
    std::vector<double> drive_frequency;
    std::vector<Index> driven_sites;

    double t0 = 0.0;
    double tfinal = 1.0;

    BenchmarkStateSelection state_selection;

    bool use_batched_gpu_specific_state_path = false;
    Index batched_num_steps = 500;
};

struct TransmonChainBenchmarkTiming {
    double wall_seconds = 0.0;
    Index num_evolved_states = 0;
    std::vector<std::vector<Complex>> final_states;
};

std::vector<Index> make_first_n_state_indices(Index n);

std::vector<Index> resolve_state_selection(
    const Solver& solver,
    const BenchmarkStateSelection& selection);

Model build_transmon_chain_model(
    const TransmonChainBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_transmon_chain_cpu_benchmark(
    const TransmonChainBenchmarkConfig& config);

TransmonChainBenchmarkTiming run_transmon_chain_cuda_benchmark(
    const TransmonChainBenchmarkConfig& config);

} // namespace culindblad