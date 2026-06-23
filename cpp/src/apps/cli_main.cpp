#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <petscts.h>

#include "culindblad/transmon_chain_benchmark.hpp"
#include "culindblad/tunable_coupler_benchmark.hpp"

namespace {

using Complex = std::complex<double>;

constexpr double kStateZeroMin00 = 0.61;
constexpr double kMatrixTolerance = 1.0e-4;
constexpr double kConsistencyTolerance = 1.0e-8;
constexpr culindblad::Index kValidationBatchSteps = 1000;
constexpr culindblad::Index kDefaultStressNumTransmons = 5;
constexpr culindblad::Index kDefaultStressCutoffDim = 2;
constexpr culindblad::Index kDefaultStressBatchSteps = 250;

struct TrustedBaseline {
    const char* label = nullptr;
    const char* reason = nullptr;
    std::vector<Complex> state_zero_matrix;
};

struct StressTestOptions {
    bool enabled = false;
    bool run_cpu = false;
    culindblad::Index num_transmons = kDefaultStressNumTransmons;
    culindblad::Index cutoff_dim = kDefaultStressCutoffDim;
    culindblad::Index batched_num_steps = kDefaultStressBatchSteps;
    std::vector<culindblad::Index> batch_sizes{1, 2, 4};
};

TrustedBaseline make_trusted_state_zero_baseline()
{
    return {
        "QuTiP-matched matrix",
        "The active benchmark path uses TSRK + TSRK4 with TSADAPTNONE. "
        "The historical hardcoded matrix predates the current fixed-step solver semantics, "
        "while this matrix matches the same model under the current configuration.",
        {
            {0.618762, 0.0}, {-0.0116998, -0.00646817}, {0.0359727, 0.0187981}, {0.0114843, 0.00366108},
            {-0.0116998, 0.00646817}, {0.0373947, 0.0}, {-0.0355948, -0.0742133}, {0.00937813, 0.0117711},
            {0.0359727, -0.0187981}, {-0.0355948, 0.0742133}, {0.334491, 0.0}, {-0.0370112, 0.00588792},
            {0.0114843, -0.00366108}, {0.00937813, -0.0117711}, {-0.0370112, -0.00588792}, {0.00935285, 0.0}
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

    PetscBool run_cpu = PETSC_FALSE;
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

void dump_one_propagator(
    const char* backend,
    const culindblad::TransmonChainBenchmarkConfig& config,
    const culindblad::Solver& solver,
    const culindblad::TransmonChainBenchmarkTiming& timing,
    culindblad::Index batch_size)
{
    std::cout << "PROPAGATOR_DUMP_BEGIN backend=" << backend
              << " num_transmons=" << config.num_transmons
              << " cutoff_dim=" << config.cutoff_dim
              << " hilbert_dim=" << solver.layout.hilbert_dim
              << " density_dim=" << solver.layout.density_dim
              << " steps=" << config.batched_num_steps
              << " batch_size=" << batch_size
              << " num_states=" << timing.final_states.size() << "\n";
    const std::streamsize old_prec = std::cout.precision();
    std::cout << std::setprecision(17);
    for (std::size_t s = 0; s < timing.final_states.size(); ++s) {
        std::cout << "STATE " << s << "\n";
        const std::vector<Complex>& rho = timing.final_states[s];
        for (std::size_t i = 0; i < rho.size(); ++i) {
            std::cout << rho[i].real() << " " << rho[i].imag() << "\n";
        }
    }
    std::cout << std::setprecision(old_prec);
    std::cout << "PROPAGATOR_DUMP_END backend=" << backend
              << " batch_size=" << batch_size << "\n";
}

// Evolve ALL density_dim basis density matrices |row><col| (row-major flat index
// s = row*D + col) so the full Lindblad propagator can be reconstructed and
// compared entry-by-entry against the QuTiP reference. Non-invasive: gated behind
// -dump_propagator and never touches the trusted validation path.
void dump_full_propagator(
    const culindblad::TransmonChainBenchmarkConfig& base_config,
    bool run_cpu)
{
    using namespace culindblad;

    TransmonChainBenchmarkConfig config = base_config;
    config.state_selection.evolve_all_states = true;
    config.state_selection.selected_state_indices.clear();

    const Model model = build_transmon_chain_model(config);
    const Solver solver = make_solver(model);

    const TransmonChainBenchmarkTiming gpu_timing =
        run_transmon_chain_cuda_benchmark(config);
    dump_one_propagator("gpu", config, solver, gpu_timing,
                        gpu_timing.final_states.size());

    if (run_cpu) {
        const TransmonChainBenchmarkTiming cpu_timing =
            run_transmon_chain_cpu_benchmark(config);
        dump_one_propagator("cpu", config, solver, cpu_timing,
                            cpu_timing.final_states.size());
    }
}

// For each requested batch size B, evolve the FIRST B basis density matrices in a
// single batched-GPU solve (batch dimension = B) and dump them. This confirms the
// batched GPU path is independent of batch size (state s must be identical across
// every B>s) and that each evolved state matches QuTiP.
void dump_batch_sweep(
    const culindblad::TransmonChainBenchmarkConfig& base_config,
    const std::vector<culindblad::Index>& batch_sizes)
{
    using namespace culindblad;

    const Model model = build_transmon_chain_model(base_config);
    const Solver solver = make_solver(model);

    for (const Index batch_size : batch_sizes) {
        if (batch_size > solver.layout.density_dim) {
            throw std::invalid_argument(
                "dump_batch_sweep: batch size exceeds density_dim");
        }
        TransmonChainBenchmarkConfig config = base_config;
        config.state_selection.evolve_all_states = false;
        config.state_selection.selected_state_indices =
            make_first_n_state_indices(batch_size);

        const TransmonChainBenchmarkTiming timing =
            run_transmon_chain_cuda_benchmark(config);
        dump_one_propagator("gpu", config, solver, timing, batch_size);
    }
}

// ===== Tunable-coupler (N-qubit / M-coupler) propagator dump =====
// Same PROPAGATOR_DUMP format as the chain dump, for the all-to-all tunable-coupler
// model. Used to verify the C++ solver against the QuTiP reference (2Q1C etc).

culindblad::TunableCouplerBenchmarkConfig make_tunable_coupler_benchmark_config(
    culindblad::Index num_qubits,
    culindblad::Index num_couplers,
    culindblad::Index cutoff_dim,
    culindblad::Index batched_num_steps,
    bool enable_drives)
{
    culindblad::TunableCouplerBenchmarkConfig config{};
    config.num_qubits = num_qubits;
    config.num_couplers = num_couplers;
    config.cutoff_dim = cutoff_dim;
    config.square_topology = (num_couplers == num_qubits);  // 4Q4C ring; others linear

    const culindblad::Index n_elem = num_qubits + num_couplers;
    config.omega.resize(n_elem, 0.0);
    config.alpha.assign(n_elem, -0.3);
    config.t1.assign(n_elem, 20.0);
    config.tphi.assign(n_elem, 30.0);
    for (culindblad::Index i = 0; i < n_elem; ++i) {
        config.omega[i] = 5.0 + 0.05 * static_cast<double>(i);
    }
    config.g = 0.02;
    config.t0 = 0.0;
    config.tfinal = 40.0;
    config.batched_num_steps = batched_num_steps;

    // Drive the first num_qubits/2 qubits (sites 0..) and the first num_couplers/2
    // couplers (sites num_qubits..), i.e. ~half the elements. For 4Q4C this is
    // Q1,Q2,C1,C2; for 2Q1C just Q1.
    if (enable_drives) {
        config.enable_drives = true;
        config.drive_amplitude = 0.5;
        config.drive_t_gate = config.tfinal;
        const culindblad::Index nq_drive = num_qubits / 2;
        const culindblad::Index nc_drive = num_couplers / 2;
        for (culindblad::Index q = 0; q < nq_drive; ++q) {
            config.driven_sites.push_back(q);                  // qubit site q
        }
        for (culindblad::Index c = 0; c < nc_drive; ++c) {
            config.driven_sites.push_back(num_qubits + c);     // coupler site
        }
    }
    return config;
}

void tc_dump_one_propagator(
    const char* backend,
    const culindblad::TunableCouplerBenchmarkConfig& config,
    const culindblad::Solver& solver,
    const culindblad::TransmonChainBenchmarkTiming& timing,
    culindblad::Index batch_size)
{
    std::cout << "PROPAGATOR_DUMP_BEGIN backend=" << backend
              << " num_qubits=" << config.num_qubits
              << " num_couplers=" << config.num_couplers
              << " cutoff_dim=" << config.cutoff_dim
              << " hilbert_dim=" << solver.layout.hilbert_dim
              << " density_dim=" << solver.layout.density_dim
              << " steps=" << config.batched_num_steps
              << " batch_size=" << batch_size
              << " num_states=" << timing.final_states.size() << "\n";
    const std::streamsize old_prec = std::cout.precision();
    std::cout << std::setprecision(17);
    for (std::size_t s = 0; s < timing.final_states.size(); ++s) {
        std::cout << "STATE " << s << "\n";
        const std::vector<Complex>& rho = timing.final_states[s];
        for (std::size_t i = 0; i < rho.size(); ++i) {
            std::cout << rho[i].real() << " " << rho[i].imag() << "\n";
        }
    }
    std::cout << std::setprecision(old_prec);
    std::cout << "PROPAGATOR_DUMP_END backend=" << backend
              << " batch_size=" << batch_size << "\n";
}

// Dump the states selected by config.state_selection (full propagator if
// evolve_all_states, otherwise the listed indices, e.g. computational states).
void tc_dump_states(
    const culindblad::TunableCouplerBenchmarkConfig& config,
    bool run_cpu)
{
    using namespace culindblad;
    const Model model = build_tunable_coupler_model(config);
    const Solver solver = make_solver(model);

    const TransmonChainBenchmarkTiming gpu_timing =
        run_tunable_coupler_cuda_benchmark(config);
    tc_dump_one_propagator("gpu", config, solver, gpu_timing,
                           gpu_timing.final_states.size());

    if (run_cpu) {
        const TransmonChainBenchmarkTiming cpu_timing =
            run_tunable_coupler_cpu_benchmark(config);
        tc_dump_one_propagator("cpu", config, solver, cpu_timing,
                               cpu_timing.final_states.size());
    }
}

void tc_dump_batch_sweep(
    const culindblad::TunableCouplerBenchmarkConfig& base_config,
    const std::vector<culindblad::Index>& batch_sizes)
{
    using namespace culindblad;
    const Model model = build_tunable_coupler_model(base_config);
    const Solver solver = make_solver(model);

    for (const Index batch_size : batch_sizes) {
        if (batch_size > solver.layout.density_dim) {
            throw std::invalid_argument(
                "tc_dump_batch_sweep: batch size exceeds density_dim");
        }
        TunableCouplerBenchmarkConfig config = base_config;
        config.state_selection.evolve_all_states = false;
        config.state_selection.selected_state_indices =
            make_first_n_state_indices(batch_size);
        const TransmonChainBenchmarkTiming timing =
            run_tunable_coupler_cuda_benchmark(config);
        tc_dump_one_propagator("gpu", config, solver, timing, batch_size);
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

        // Tunable-coupler dump mode: build the N-qubit/M-coupler all-to-all model and
        // dump its propagator for verification against the QuTiP reference. Gated
        // behind -dump_tc; never touches the trusted chain validation path.
        PetscBool dump_tc = PETSC_FALSE;
        PetscCallAbort(
            PETSC_COMM_SELF,
            PetscOptionsGetBool(nullptr, nullptr, "-dump_tc", &dump_tc, nullptr));
        if (dump_tc == PETSC_TRUE) {
            PetscInt tc_nq = 2;
            PetscInt tc_nc = 1;
            PetscInt tc_cd = 3;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetInt(
                nullptr, nullptr, "-tc_num_qubits", &tc_nq, nullptr));
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetInt(
                nullptr, nullptr, "-tc_num_couplers", &tc_nc, nullptr));
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetInt(
                nullptr, nullptr, "-dump_cutoff_dim", &tc_cd, nullptr));

            PetscInt dump_steps = 0;
            PetscBool steps_set = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetInt(
                nullptr, nullptr, "-dump_steps", &dump_steps, &steps_set));
            const Index steps =
                (steps_set == PETSC_TRUE && dump_steps > 0)
                    ? static_cast<Index>(dump_steps)
                    : kValidationBatchSteps;

            PetscBool tc_drives = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                nullptr, nullptr, "-tc_drives", &tc_drives, nullptr));
            PetscBool tc_comp = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                nullptr, nullptr, "-tc_computational", &tc_comp, nullptr));

            TunableCouplerBenchmarkConfig tc_config =
                make_tunable_coupler_benchmark_config(
                    static_cast<Index>(tc_nq),
                    static_cast<Index>(tc_nc),
                    static_cast<Index>(tc_cd),
                    steps,
                    tc_drives == PETSC_TRUE);

            char tc_batch_buf[256] = {};
            PetscBool tc_batch_set = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetString(
                nullptr, nullptr, "-dump_batch_sizes",
                tc_batch_buf, sizeof(tc_batch_buf), &tc_batch_set));

            // Computational-subspace states (qubits in {0,1}, couplers in |0>) vs the
            // full propagator (all density_dim basis states).
            if (tc_comp == PETSC_TRUE) {
                tc_config.state_selection.evolve_all_states = false;
                tc_config.state_selection.selected_state_indices =
                    tunable_coupler_computational_density_indices(
                        static_cast<Index>(tc_nq),
                        static_cast<Index>(tc_nc),
                        static_cast<Index>(tc_cd));
            } else {
                tc_config.state_selection.evolve_all_states = true;
                tc_config.state_selection.selected_state_indices.clear();
            }

            if (tc_batch_set == PETSC_TRUE && tc_comp != PETSC_TRUE) {
                tc_dump_batch_sweep(tc_config, parse_batch_sizes_csv(tc_batch_buf));
            } else {
                PetscBool dump_cpu = PETSC_FALSE;
                PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                    nullptr, nullptr, "-dump_propagator_cpu", &dump_cpu, nullptr));
                tc_dump_states(tc_config, dump_cpu == PETSC_TRUE);
            }
            PetscFinalize();
            return 0;
        }

        // Full-propagator dump mode: evolve all 16 basis density matrices and emit
        // each final state for entry-by-entry comparison with the QuTiP reference.
        PetscBool dump_prop = PETSC_FALSE;
        PetscCallAbort(
            PETSC_COMM_SELF,
            PetscOptionsGetBool(
                nullptr, nullptr, "-dump_propagator", &dump_prop, nullptr));
        if (dump_prop == PETSC_TRUE) {
            // System size for the dump (defaults to the N=2 d=2 fixture).
            PetscInt dump_nt = 2;
            PetscInt dump_cd = 2;
            PetscCallAbort(
                PETSC_COMM_SELF,
                PetscOptionsGetInt(
                    nullptr, nullptr, "-dump_num_transmons", &dump_nt, nullptr));
            PetscCallAbort(
                PETSC_COMM_SELF,
                PetscOptionsGetInt(
                    nullptr, nullptr, "-dump_cutoff_dim", &dump_cd, nullptr));

            PetscInt dump_steps = 0;
            PetscBool steps_set = PETSC_FALSE;
            PetscCallAbort(
                PETSC_COMM_SELF,
                PetscOptionsGetInt(
                    nullptr, nullptr, "-dump_steps", &dump_steps, &steps_set));
            const Index steps =
                (steps_set == PETSC_TRUE && dump_steps > 0)
                    ? static_cast<Index>(dump_steps)
                    : kValidationBatchSteps;

            const TransmonChainBenchmarkConfig dump_config =
                make_transmon_chain_benchmark_config(
                    static_cast<Index>(dump_nt),
                    static_cast<Index>(dump_cd),
                    steps);

            char batch_sizes_buf[256] = {};
            PetscBool batch_set = PETSC_FALSE;
            PetscCallAbort(
                PETSC_COMM_SELF,
                PetscOptionsGetString(
                    nullptr, nullptr, "-dump_batch_sizes",
                    batch_sizes_buf, sizeof(batch_sizes_buf), &batch_set));

            if (batch_set == PETSC_TRUE) {
                dump_batch_sweep(dump_config, parse_batch_sizes_csv(batch_sizes_buf));
            } else {
                PetscBool dump_cpu = PETSC_FALSE;
                PetscCallAbort(
                    PETSC_COMM_SELF,
                    PetscOptionsGetBool(
                        nullptr, nullptr, "-dump_propagator_cpu", &dump_cpu, nullptr));
                dump_full_propagator(dump_config, dump_cpu == PETSC_TRUE);
            }
            PetscFinalize();
            return 0;
        }

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
