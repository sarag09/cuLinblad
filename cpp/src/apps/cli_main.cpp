#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <petscts.h>

#include "culindblad/batch_evolution.hpp"
#include "culindblad/model.hpp"
#include "culindblad/petsc_apply.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/transmon_chain_benchmark.hpp"
#include "culindblad/types.hpp"

namespace {

void print_selected_state_matrix(
    const std::vector<std::complex<double>>& rho,
    std::size_t hilbert_dim,
    std::size_t max_dim_to_print = 4)
{
    const std::size_t dim = std::min(hilbert_dim, max_dim_to_print);
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col < dim; ++col) {
            std::cout << rho[row * hilbert_dim + col] << " ";
        }
        std::cout << std::endl;
    }
}

void print_matrix_with_fixed_precision(
    const std::vector<std::complex<double>>& rho,
    std::size_t hilbert_dim)
{
    const std::streamsize old_precision = std::cout.precision();
    const std::ios::fmtflags old_flags = std::cout.flags();

    std::cout << std::setprecision(8) << std::defaultfloat;
    print_selected_state_matrix(rho, hilbert_dim, hilbert_dim);

    std::cout.precision(old_precision);
    std::cout.flags(old_flags);
}

bool milestone0_validation_passed(
    const culindblad::Milestone0ValidationReport& report)
{
    return !report.hard_failure_first_element &&
           report.single_matches_expected &&
           report.batch1_matches_expected &&
           report.batch2_matches_expected &&
           report.triple_invariance_passed;
}

std::vector<std::vector<std::complex<double>>> make_selected_basis_density_states_for_cli(
    const culindblad::Solver& solver,
    const std::vector<culindblad::Index>& selected_state_indices)
{
    using culindblad::Complex;
    using culindblad::Index;

    std::vector<std::vector<std::complex<double>>> states;
    states.reserve(selected_state_indices.size());

    for (Index state_index : selected_state_indices) {
        if (state_index >= solver.layout.density_dim) {
            throw std::invalid_argument(
                "make_selected_basis_density_states_for_cli: selected state index out of range");
        }

        std::vector<Complex> rho(
            solver.layout.density_dim,
            Complex{0.0, 0.0});
        rho[state_index] = Complex{1.0, 0.0};
        states.push_back(std::move(rho));
    }

    return states;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace culindblad;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr != 0) {
        std::cerr << "PetscInitialize failed." << std::endl;
        return 1;
    }

    try {
        PetscBool milestone0_validate = PETSC_FALSE;
        PetscCall(PetscOptionsGetBool(
            nullptr,
            nullptr,
            "-milestone0_validate",
            &milestone0_validate,
            nullptr));

        if (milestone0_validate == PETSC_TRUE) {
            const Milestone0ValidationReport report =
                run_milestone0_n2_d2_gpu_validation();

            std::cout << "===== Milestone 0 Validation (N=2, d=2) =====" << std::endl;
            std::cout << "Expected State 0 matrix:" << std::endl;
            print_matrix_with_fixed_precision(report.expected_state0, 4);
            std::cout << std::endl;

            std::cout << "CUDA single-state State 0:" << std::endl;
            print_matrix_with_fixed_precision(report.single_state_gpu_state0, 4);
            std::cout << "max|single - expected| = "
                      << report.max_diff_single_vs_expected << std::endl;
            std::cout << std::endl;

            std::cout << "CUDA batched State 0 with batch size 1:" << std::endl;
            print_matrix_with_fixed_precision(report.batched_gpu_batch1_state0, 4);
            std::cout << "max|batch1 - expected| = "
                      << report.max_diff_batch1_vs_expected << std::endl;
            std::cout << std::endl;

            std::cout << "CUDA batched State 0 with batch size 2:" << std::endl;
            print_matrix_with_fixed_precision(report.batched_gpu_batch2_state0, 4);
            std::cout << "max|batch2 - expected| = "
                      << report.max_diff_batch2_vs_expected << std::endl;
            std::cout << std::endl;

            std::cout << "max|single - batch1| = "
                      << report.max_diff_single_vs_batch1 << std::endl;
            std::cout << "max|single - batch2| = "
                      << report.max_diff_single_vs_batch2 << std::endl;
            std::cout << "max|batch1 - batch2| = "
                      << report.max_diff_batch1_vs_batch2 << std::endl;
            std::cout << std::endl;

            std::cout << "State 0 (0,0) real parts:" << std::endl;
            std::cout << "single: " << report.single_state0_00_real << std::endl;
            std::cout << "batch1: " << report.batch1_state0_00_real << std::endl;
            std::cout << "batch2: " << report.batch2_state0_00_real << std::endl;
            std::cout << "hard first-element failure (< 0.61): "
                      << (report.hard_failure_first_element ? "yes" : "no") << std::endl;
            std::cout << std::endl;

            std::cout << "Diagnosis:" << std::endl;
            std::cout << "single matches batch1: "
                      << (report.single_matches_batch1 ? "yes" : "no") << std::endl;
            std::cout << "solver semantics mismatch present: "
                      << (report.solver_semantics_mismatch_present ? "yes" : "no") << std::endl;
            std::cout << "batched path batch-invariance failure present: "
                      << (report.batched_path_batch_invariance_failure ? "yes" : "no") << std::endl;
            std::cout << "triple invariance passed: "
                      << (report.triple_invariance_passed ? "yes" : "no") << std::endl;
            std::cout << "label-based cache identity present: "
                      << (report.label_based_cache_identity_present ? "yes" : "no") << std::endl;
            std::cout << "multiple sources present: "
                      << (report.multiple_sources_present ? "yes" : "no") << std::endl;

            PetscFinalize();
            return milestone0_validation_passed(report) ? 0 : 2;
        }

        std::cout << "===== cuLindblad transmon-chain benchmark =====" << std::endl;

        TransmonChainBenchmarkConfig base_config{};
        base_config.num_transmons = 2;
        base_config.cutoff_dim = 2;

        base_config.omega = {5.0, 5.05};
        base_config.alpha = {-0.3, -0.3};
        base_config.g = {0.02};
        base_config.t1 = {20.0, 20.0};
        base_config.tphi = {30.0, 30.0};

        base_config.drive_amplitude = 0.1;
        base_config.drive_sigma = 10.0;
        base_config.drive_center = 20.0;
        base_config.drive_frequency = {5.0, 5.05};

        base_config.driven_sites = {0};

        base_config.t0 = 0.0;
        base_config.tfinal = 40.0;
        base_config.use_batched_gpu_specific_state_path = false;
        base_config.batched_num_steps = 500;

        // Milestone 0: keep the feedback loop focused on the specific-state sweep.
        //
        // {
        //     std::cout << "\n===== Full-state evolution (N=2, d=2) =====" << std::endl;
        //
        //     TransmonChainBenchmarkConfig config = base_config;
        //     config.state_selection.evolve_all_states = true;
        //     config.state_selection.selected_state_indices.clear();
        //
        //     Model model = build_transmon_chain_model(config);
        //     Solver solver = make_solver(model);
        //
        //     std::cout << "Hilbert dimension: "
        //               << solver.layout.hilbert_dim << std::endl;
        //     std::cout << "Density dimension: "
        //               << solver.layout.density_dim << std::endl;
        //
        //     TransmonChainBenchmarkTiming cpu_full =
        //         run_transmon_chain_cpu_benchmark(config);
        //
        //     TransmonChainBenchmarkTiming gpu_full =
        //         run_transmon_chain_cuda_benchmark(config);
        //
        //     std::cout << "CPU full-state time (s): "
        //               << cpu_full.wall_seconds << std::endl;
        //     std::cout << "GPU full-state time (s): "
        //               << gpu_full.wall_seconds << std::endl;
        //     std::cout << "Number of evolved states: "
        //               << gpu_full.num_evolved_states << std::endl;
        //
        //     std::cout << "\nFull evolved propagator basis outputs (GPU), first 4 states:" << std::endl;
        //     for (Index i = 0; i < std::min<Index>(4, gpu_full.final_states.size()); ++i) {
        //         std::cout << "State " << i << ":" << std::endl;
        //         print_selected_state_matrix(
        //             gpu_full.final_states[i],
        //             solver.layout.hilbert_dim,
        //             solver.layout.hilbert_dim);
        //         std::cout << std::endl;
        //     }
        // }

        std::cout << "\n===== Specific-state evolution sweep =====" << std::endl;

        for (Index first_n : std::vector<Index>{1, 2, 4}) {
            TransmonChainBenchmarkConfig config = base_config;
            config.state_selection.evolve_all_states = false;
            config.state_selection.selected_state_indices =
                make_first_n_state_indices(first_n);

            Model model = build_transmon_chain_model(config);
            Solver solver = make_solver(model);

            TransmonChainBenchmarkTiming cpu_selected =
                run_transmon_chain_cpu_benchmark(config);

            TransmonChainBenchmarkTiming gpu_selected =
                run_transmon_chain_cuda_benchmark(config);

            const double cpu_states_per_second =
                static_cast<double>(cpu_selected.num_evolved_states) / cpu_selected.wall_seconds;

            const double gpu_states_per_second =
                static_cast<double>(gpu_selected.num_evolved_states) / gpu_selected.wall_seconds;

            std::cout << "\nFirst " << first_n << " states" << std::endl;
            std::cout << "CPU selected-state time (s): "
                      << cpu_selected.wall_seconds << std::endl;
            std::cout << "GPU selected-state time (s): "
                      << gpu_selected.wall_seconds << std::endl;
            std::cout << "CPU selected states/s: "
                      << cpu_states_per_second << std::endl;
            std::cout << "GPU selected states/s: "
                      << gpu_states_per_second << std::endl;

            if (!gpu_selected.final_states.empty()) {
                std::cout << "Selected state 0 corresponds to basis density index "
                          << config.state_selection.selected_state_indices[0] << std::endl;
                print_selected_state_matrix(
                    gpu_selected.final_states[0],
                    solver.layout.hilbert_dim,
                    solver.layout.hilbert_dim);
            }
        }

        std::cout << "\n===== Stress test (N=3, d=3, 64 states) =====" << std::endl;

        TransmonChainBenchmarkConfig stress_config{};
        stress_config.num_transmons = 3;
        stress_config.cutoff_dim = 3;
        stress_config.omega = {5.00, 5.05, 5.10};
        stress_config.alpha = {-0.30, -0.30, -0.30};
        stress_config.g = {0.02, 0.02};
        stress_config.t1 = {20.0, 20.0, 20.0};
        stress_config.tphi = {30.0, 30.0, 30.0};
        stress_config.drive_amplitude = 0.1;
        stress_config.drive_sigma = 10.0;
        stress_config.drive_center = 20.0;
        stress_config.drive_frequency = {5.00, 5.05, 5.10};
        stress_config.driven_sites = {0};
        stress_config.t0 = 0.0;
        stress_config.tfinal = 40.0;
        stress_config.use_batched_gpu_specific_state_path = false;
        stress_config.batched_num_steps = 500;
        stress_config.state_selection.evolve_all_states = false;
        stress_config.state_selection.selected_state_indices =
            make_first_n_state_indices(64);

        Model stress_model = build_transmon_chain_model(stress_config);
        Solver stress_solver = make_solver(stress_model);

        const std::vector<Index> stress_indices =
            resolve_state_selection(stress_solver, stress_config.state_selection);
        const std::vector<std::vector<Complex>> stress_initial_states =
            make_selected_basis_density_states_for_cli(stress_solver, stress_indices);

        const BatchEvolutionConfig stress_batch_config{
            stress_config.t0,
            (stress_config.tfinal - stress_config.t0) /
                static_cast<double>(stress_config.batched_num_steps),
            stress_config.batched_num_steps,
            64
        };

        const BatchEvolutionTiming gpu_stress =
            time_density_batch_cuda_ts(
                stress_solver,
                stress_initial_states,
                stress_batch_config);

        std::cout << "Hilbert dimension: "
                  << stress_solver.layout.hilbert_dim << std::endl;
        std::cout << "Density dimension: "
                  << stress_solver.layout.density_dim << std::endl;
        std::cout << "GPU selected-state time (s): "
                  << gpu_stress.elapsed_seconds << std::endl;
        std::cout << "GPU kernel time (s): "
                  << gpu_stress.gpu_elapsed_seconds << std::endl;

        const BatchEvolutionTiming cpu_stress =
            time_density_batch_cpu_reference(
                stress_solver,
                stress_initial_states,
                stress_batch_config);

        const double cpu_stress_states_per_second =
            static_cast<double>(stress_indices.size()) / cpu_stress.elapsed_seconds;
        const double gpu_stress_states_per_second =
            static_cast<double>(stress_indices.size()) / gpu_stress.elapsed_seconds;

        std::cout << "CPU selected-state time (s): "
                  << cpu_stress.elapsed_seconds << std::endl;
        std::cout << "CPU selected states/s: "
                  << cpu_stress_states_per_second << std::endl;
        std::cout << "GPU selected states/s: "
                  << gpu_stress_states_per_second << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
