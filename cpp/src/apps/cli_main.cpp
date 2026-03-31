#include <complex>
#include <iostream>
#include <vector>

#include <petscts.h>

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

    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
