#include <algorithm>
#include <complex>
#include <cstdlib>
#include <sstream>
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
constexpr culindblad::Index kValidationBatchSteps = 1000;
constexpr culindblad::Index kDefaultStressNumTransmons = 6;
constexpr culindblad::Index kDefaultStressCutoffDim = 2;
constexpr culindblad::Index kDefaultStressBatchSteps = 250;

struct TrustedBaseline {
    const char* label = nullptr;
    const char* reason = nullptr;
    std::vector<Complex> state_zero_matrix;
};

struct StressTestOptions {
    bool enabled = false;
    bool run_cpu = true;
    culindblad::Index num_transmons = kDefaultStressNumTransmons;
    culindblad::Index cutoff_dim = kDefaultStressCutoffDim;
    culindblad::Index batched_num_steps = kDefaultStressBatchSteps;
    std::vector<culindblad::Index> batch_sizes{1, 2, 4};
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

culindblad::TransmonChainBenchmarkConfig make_transmon_chain_benchmark_config(
    culindblad::Index num_transmons,
    culindblad::Index cutoff_dim,
    culindblad::Index batched_num_steps)
{
    if ((num_transmons % 2) != 0) {
        throw std::invalid_argument(
            "benchmark configuration requires an even number of transmons so exactly half can be driven");
    }

    culindblad::TransmonChainBenchmarkConfig config{};
    config.num_transmons = num_transmons;
    config.cutoff_dim = cutoff_dim;

    config.omega.resize(num_transmons, 0.0);
    config.alpha.assign(num_transmons, -0.3);
    config.t1.assign(num_transmons, 20.0);
    config.tphi.assign(num_transmons, 30.0);
    config.drive_frequency.resize(num_transmons, 0.0);
    config.g.assign(num_transmons > 0 ? num_transmons - 1 : 0, 0.02);

    for (culindblad::Index i = 0; i < num_transmons; ++i) {
        const double omega = 5.0 + 0.05 * static_cast<double>(i);
        config.omega[i] = omega;
        config.drive_frequency[i] = omega;
    }

    config.drive_amplitude = 0.1;
    config.drive_sigma = 10.0;
    config.drive_center = 20.0;
    config.driven_sites.reserve(num_transmons / 2);
    for (culindblad::Index site = 0; site < num_transmons / 2; ++site) {
        config.driven_sites.push_back(site);
    }

    config.t0 = 0.0;
    config.tfinal = 40.0;
    config.use_batched_gpu_specific_state_path = true;
    config.batched_num_steps = batched_num_steps;
    return config;
}

std::vector<culindblad::Index> parse_batch_sizes_csv(
    const std::string& csv)
{
    std::vector<culindblad::Index> batch_sizes;
    std::stringstream stream(csv);
    std::string token;

    while (std::getline(stream, token, ',')) {
        const std::size_t first = token.find_first_not_of(" \t");
        if (first == std::string::npos) {
            continue;
        }

        const std::size_t last = token.find_last_not_of(" \t");
        const std::string trimmed = token.substr(first, last - first + 1);
        const int value = std::stoi(trimmed);
        if (value <= 0) {
            throw std::invalid_argument(
                "stress batch sizes must be positive integers");
        }
        batch_sizes.push_back(static_cast<culindblad::Index>(value));
    }

    if (batch_sizes.empty()) {
        throw std::invalid_argument(
            "stress batch sizes list must contain at least one value");
    }

    return batch_sizes;
}

StressTestOptions parse_stress_test_options()
{
    StressTestOptions options{};

    PetscBool enabled = PETSC_FALSE;
    PetscCallAbort(
        PETSC_COMM_SELF,
        PetscOptionsGetBool(
            nullptr,
            nullptr,
            "-stress_test",
            &enabled,
            nullptr));
    options.enabled = (enabled == PETSC_TRUE);

    PetscBool run_cpu = PETSC_TRUE;
    PetscCallAbort(
        PETSC_COMM_SELF,
        PetscOptionsGetBool(
            nullptr,
            nullptr,
            "-stress_run_cpu",
            &run_cpu,
            nullptr));
    options.run_cpu = (run_cpu == PETSC_TRUE);

    PetscBool set = PETSC_FALSE;
    PetscInt value = 0;
    PetscCallAbort(
        PETSC_COMM_SELF,
        PetscOptionsGetInt(
            nullptr,
            nullptr,
            "-stress_num_transmons",
            &value,
            &set));
    if (set == PETSC_TRUE) {
        options.num_transmons = static_cast<culindblad::Index>(value);
    }

    PetscCallAbort(
        PETSC_COMM_SELF,
        PetscOptionsGetInt(
            nullptr,
            nullptr,
            "-stress_cutoff_dim",
            &value,
            &set));
    if (set == PETSC_TRUE) {
        options.cutoff_dim = static_cast<culindblad::Index>(value);
    }

    PetscCallAbort(
        PETSC_COMM_SELF,
        PetscOptionsGetInt(
            nullptr,
            nullptr,
            "-stress_batched_num_steps",
            &value,
            &set));
    if (set == PETSC_TRUE) {
        options.batched_num_steps = static_cast<culindblad::Index>(value);
    }

    char batch_sizes_buffer[256] = {};
    PetscCallAbort(
        PETSC_COMM_SELF,
        PetscOptionsGetString(
            nullptr,
            nullptr,
            "-stress_batch_sizes",
            batch_sizes_buffer,
            sizeof(batch_sizes_buffer),
            &set));
    if (set == PETSC_TRUE) {
        options.batch_sizes = parse_batch_sizes_csv(batch_sizes_buffer);
    }

    return options;
}

void print_gpu_sweep(
    const std::string& heading,
    const culindblad::TransmonChainBenchmarkConfig& config,
    const std::vector<culindblad::Index>& batch_sizes,
    const std::vector<culindblad::TransmonChainBenchmarkTiming>& timings)
{
    const culindblad::Model model = culindblad::build_transmon_chain_model(config);
    const culindblad::Solver solver = culindblad::make_solver(model);

    std::cout << "\n===== " << heading << " =====" << std::endl;
    std::cout << "Transmons: " << config.num_transmons
              << ", cutoff dim: " << config.cutoff_dim
              << ", Hilbert dimension: " << solver.layout.hilbert_dim
              << ", density dimension: " << solver.layout.density_dim
              << ", batched TS steps: " << config.batched_num_steps
              << std::endl;

    for (std::size_t i = 0; i < batch_sizes.size(); ++i) {
        const culindblad::Index first_n = batch_sizes[i];
        const culindblad::TransmonChainBenchmarkTiming& timing = timings.at(i);

        std::cout << "\nBatch size " << first_n << std::endl;
        std::cout << "GPU selected-state time (s): "
                  << timing.wall_seconds << std::endl;
        std::cout << "GPU selected states/s: "
                  << static_cast<double>(timing.num_evolved_states) / timing.wall_seconds
                  << std::endl;
    }
}

void print_stress_test_results(
    const StressTestOptions& options,
    const culindblad::TransmonChainBenchmarkConfig& config,
    const std::vector<culindblad::Index>& batch_sizes)
{
    const culindblad::Model model = culindblad::build_transmon_chain_model(config);
    const culindblad::Solver solver = culindblad::make_solver(model);

    std::cout << "\n===== CPU vs GPU Stress Test =====" << std::endl;
    std::cout << "Transmons: " << config.num_transmons
              << ", cutoff dim: " << config.cutoff_dim
              << ", Hilbert dimension: " << solver.layout.hilbert_dim
              << ", density dimension: " << solver.layout.density_dim
              << ", batched TS steps: " << config.batched_num_steps
              << ", run CPU benchmark: " << (options.run_cpu ? "yes" : "no")
              << std::endl;

    for (const culindblad::Index batch_size : batch_sizes) {
        culindblad::TransmonChainBenchmarkConfig run_config = config;
        run_config.state_selection.evolve_all_states = false;
        run_config.state_selection.selected_state_indices =
            culindblad::make_first_n_state_indices(batch_size);

        const culindblad::TransmonChainBenchmarkTiming gpu_timing =
            culindblad::run_transmon_chain_cuda_benchmark(run_config);
        culindblad::TransmonChainBenchmarkTiming cpu_timing{};
        if (options.run_cpu) {
            cpu_timing = culindblad::run_transmon_chain_cpu_benchmark(run_config);
        }

        std::cout << "\nBatch size " << batch_size << std::endl;
        std::cout << "GPU selected-state time (s): "
                  << gpu_timing.wall_seconds << std::endl;
        std::cout << "GPU selected states/s: "
                  << static_cast<double>(gpu_timing.num_evolved_states) / gpu_timing.wall_seconds
                  << std::endl;
        if (options.run_cpu) {
            std::cout << "CPU selected-state time (s): "
                      << cpu_timing.wall_seconds << std::endl;
            std::cout << "CPU selected states/s: "
                      << static_cast<double>(cpu_timing.num_evolved_states) / cpu_timing.wall_seconds
                      << std::endl;
            std::cout << "CPU/GPU speedup: "
                      << cpu_timing.wall_seconds / gpu_timing.wall_seconds
                      << std::endl;
        }
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

        const StressTestOptions stress_options = parse_stress_test_options();

        if (stress_options.enabled) {
            const TransmonChainBenchmarkConfig stress_config =
                make_transmon_chain_benchmark_config(
                    stress_options.num_transmons,
                    stress_options.cutoff_dim,
                    stress_options.batched_num_steps);
            print_stress_test_results(
                stress_options,
                stress_config,
                stress_options.batch_sizes);
            PetscFinalize();
            return 0;
        }

        const TransmonChainBenchmarkConfig base_config =
            make_transmon_chain_benchmark_config(2, 2, kValidationBatchSteps);

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

        print_gpu_sweep(
            "True Batched GPU Sweep",
            base_config,
            {1, 2, 4},
            {batch1, batch2, batch4});

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
