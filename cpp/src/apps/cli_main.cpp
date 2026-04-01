#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <petscts.h>

#include "culindblad/transmon_chain_benchmark.hpp"

namespace {

using Complex = std::complex<double>;

constexpr double kStateZeroMin00 = 0.61;
constexpr double kMatrixTolerance = 1.0e-4;
constexpr double kConsistencyTolerance = 1.0e-8;

struct TrustedBaseline {
    const char* label = nullptr;
    const char* reason = nullptr;
    std::vector<Complex> state_zero_matrix;
};

TrustedBaseline make_trusted_state_zero_baseline()
{
    return {
        "QuTiP-matched matrix",
        "The active benchmark path uses TSRK + TSRK5DP with TSADAPTNONE. "
        "The historical hardcoded matrix predates the current fixed-step solver semantics, "
        "while this matrix matches the same model under the current configuration.",
        {
            {0.618768, 0.0}, {-0.011736, -0.00655564}, {0.0358747, 0.0189598}, {0.0114891, 0.0038102},
            {-0.011736, 0.00655564}, {0.037414, 0.0}, {-0.0355817, -0.0742448}, {0.00936175, 0.0117862},
            {0.0358747, -0.0189598}, {-0.0355817, 0.0742448}, {0.334468, 0.0}, {-0.0370064, 0.00583516},
            {0.0114891, -0.0038102}, {0.00936175, -0.0117862}, {-0.0370064, -0.00583516}, {0.00935067, 0.0}
        }
    };
}

void print_selected_state_matrix(
    const std::vector<Complex>& rho,
    std::size_t hilbert_dim)
{
    for (std::size_t row = 0; row < hilbert_dim; ++row) {
        for (std::size_t col = 0; col < hilbert_dim; ++col) {
            std::cout << rho[row * hilbert_dim + col] << " ";
        }
        std::cout << '\n';
    }
}

double max_abs_difference(
    const std::vector<Complex>& lhs,
    const std::vector<Complex>& rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("max_abs_difference: size mismatch");
    }

    double max_diff = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

[[noreturn]] void fail_validation(const std::string& message)
{
    throw std::runtime_error("Validation failed: " + message);
}

culindblad::TransmonChainBenchmarkTiming run_gpu_selected_state_benchmark(
    const culindblad::TransmonChainBenchmarkConfig& base_config,
    culindblad::Index first_n)
{
    culindblad::TransmonChainBenchmarkConfig config = base_config;
    config.state_selection.evolve_all_states = false;
    config.state_selection.selected_state_indices =
        culindblad::make_first_n_state_indices(first_n);
    return culindblad::run_transmon_chain_cuda_benchmark(config);
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
        base_config.use_batched_gpu_specific_state_path = true;
        base_config.batched_num_steps = 1000;

        const Model model = build_transmon_chain_model(base_config);
        const Solver solver = make_solver(model);
        const TrustedBaseline baseline = make_trusted_state_zero_baseline();

        const TransmonChainBenchmarkTiming batch1 =
            run_gpu_selected_state_benchmark(base_config, 1);
        const TransmonChainBenchmarkTiming batch2 =
            run_gpu_selected_state_benchmark(base_config, 2);
        const TransmonChainBenchmarkTiming batch4 =
            run_gpu_selected_state_benchmark(base_config, 4);

        if (batch1.final_states.empty() || batch2.final_states.empty() || batch4.final_states.empty()) {
            fail_validation("missing GPU output state data");
        }

        const double batch1_vs_batch2 =
            max_abs_difference(batch1.final_states[0], batch2.final_states[0]);
        const double batch1_vs_baseline =
            max_abs_difference(batch1.final_states[0], baseline.state_zero_matrix);
        const double batch2_vs_baseline =
            max_abs_difference(batch2.final_states[0], baseline.state_zero_matrix);
        const double state_zero_00 = batch1.final_states[0][0].real();

        std::cout << "\n===== True Batched GPU Sweep =====" << std::endl;
        for (const Index first_n : std::vector<Index>{1, 2, 4}) {
            const TransmonChainBenchmarkTiming timing =
                (first_n == 1) ? batch1 : (first_n == 2) ? batch2 : batch4;

            std::cout << "\nBatch size " << first_n << std::endl;
            std::cout << "GPU selected-state time (s): "
                      << timing.wall_seconds << std::endl;
            std::cout << "GPU selected states/s: "
                      << static_cast<double>(timing.num_evolved_states) / timing.wall_seconds
                      << std::endl;
        }

        std::cout << "\nTrusted State 0 baseline: " << baseline.label << std::endl;
        std::cout << "Baseline rationale: " << baseline.reason << std::endl;
        std::cout << "\nValidation: State 0 batch(1) vs batch(2) max abs diff = "
                  << batch1_vs_batch2 << std::endl;
        std::cout << "Validation: State 0 batch(1) vs trusted baseline max abs diff = "
                  << batch1_vs_baseline << std::endl;
        std::cout << "Validation: State 0 batch(2) vs trusted baseline max abs diff = "
                  << batch2_vs_baseline << std::endl;
        std::cout << "Validation: State 0 element (0,0) = "
                  << batch1.final_states[0][0] << std::endl;

        std::cout << "\nState 0 matrix (batch size 1):" << std::endl;
        print_selected_state_matrix(
            batch1.final_states[0],
            solver.layout.hilbert_dim);

        if (batch1_vs_batch2 > kConsistencyTolerance) {
            fail_validation(
                "batch_size=1 and batch_size=2 produced different State 0 results; max abs diff=" +
                std::to_string(batch1_vs_batch2));
        }

        if (batch1_vs_baseline > kMatrixTolerance) {
            fail_validation(
                "batch_size=1 State 0 does not match trusted baseline; max abs diff=" +
                std::to_string(batch1_vs_baseline));
        }

        if (state_zero_00 < kStateZeroMin00) {
            fail_validation(
                "State 0 element (0,0)=" + std::to_string(state_zero_00) +
                " is below 0.61");
        }

        if (batch2_vs_baseline > kMatrixTolerance) {
            fail_validation(
                "batch_size=2 State 0 does not match trusted baseline; max abs diff=" +
                std::to_string(batch2_vs_baseline));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return 0;
}
