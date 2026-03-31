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

struct Milestone0ValidationReport {
    std::vector<Complex> expected_state0;
    std::vector<Complex> single_state_gpu_state0;
    std::vector<Complex> batched_gpu_batch1_state0;
    std::vector<Complex> batched_gpu_batch2_state0;
    double max_diff_single_vs_expected = 0.0;
    double max_diff_batch1_vs_expected = 0.0;
    double max_diff_batch2_vs_expected = 0.0;
    double max_diff_single_vs_batch1 = 0.0;
    double max_diff_single_vs_batch2 = 0.0;
    double max_diff_batch1_vs_batch2 = 0.0;
    double single_state0_00_real = 0.0;
    double batch1_state0_00_real = 0.0;
    double batch2_state0_00_real = 0.0;
    bool single_matches_expected = false;
    bool batch1_matches_expected = false;
    bool batch2_matches_expected = false;
    bool single_matches_batch1 = false;
    bool single_matches_batch2 = false;
    bool batch1_matches_batch2 = false;
    bool single_state0_regime_ok = false;
    bool batch1_state0_regime_ok = false;
    bool batch2_state0_regime_ok = false;
    bool triple_invariance_passed = false;
    bool hard_failure_first_element = false;
    bool solver_semantics_mismatch_present = false;
    bool batched_path_batch_invariance_failure = false;
    bool label_based_cache_identity_present = false;
    bool multiple_sources_present = false;
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

Milestone0ValidationReport run_milestone0_n2_d2_gpu_validation();

} // namespace culindblad
