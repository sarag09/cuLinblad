#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>  // gethostname (per-rank device log)

#include <petscts.h>

#include <cuda_runtime.h>
#include <random>

#include "culindblad/batch_evolution.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/model.hpp"
#include "culindblad/model_io.hpp"
#include "culindblad/pulse_envelope.hpp"
#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/time_dependent_term.hpp"
#include "culindblad/transmon_chain_benchmark.hpp"
#include "culindblad/tunable_coupler_benchmark.hpp"
#include "culindblad/types.hpp"

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

// Set from -tc_realistic_freqs before make_tunable_coupler_benchmark_config is called.
static bool s_tc_realistic_freqs = false;

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
    // Realistic device frequency layout (opt-in via -tc_realistic_freqs): qubits spread
    // across 3.5-4.5 GHz, couplers across 4.6-5.5 GHz, so qubit-coupler detunings reach
    // ~1-2 GHz (vs the near-degenerate default). Only affects the gain measurement, not
    // correctness. Set after the default so the flag can override config.omega.
    if (s_tc_realistic_freqs) {
        auto band = [](culindblad::Index k, culindblad::Index n, double lo, double hi) {
            const double f = (n > 1) ? static_cast<double>(k) / (n - 1) : 0.0;
            return lo + (hi - lo) * f;
        };
        for (culindblad::Index q = 0; q < num_qubits; ++q) {
            config.omega[q] = band(q, num_qubits, 3.5, 4.5);
        }
        for (culindblad::Index c = 0; c < num_couplers; ++c) {
            config.omega[num_qubits + c] = band(c, num_couplers, 4.6, 5.5);
        }
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
    culindblad::Index batch_size,
    std::ostream& out = std::cout)
{
    out << "PROPAGATOR_DUMP_BEGIN backend=" << backend
        << " num_qubits=" << config.num_qubits
        << " num_couplers=" << config.num_couplers
        << " cutoff_dim=" << config.cutoff_dim
        << " hilbert_dim=" << solver.layout.hilbert_dim
        << " density_dim=" << solver.layout.density_dim
        << " steps=" << config.batched_num_steps
        << " batch_size=" << batch_size
        << " num_states=" << timing.final_states.size() << "\n";
    const std::streamsize old_prec = out.precision();
    out << std::setprecision(17);
    for (std::size_t s = 0; s < timing.final_states.size(); ++s) {
        out << "STATE " << s << "\n";
        const std::vector<Complex>& rho = timing.final_states[s];
        for (std::size_t i = 0; i < rho.size(); ++i) {
            out << rho[i].real() << " " << rho[i].imag() << "\n";
        }
    }
    out << std::setprecision(old_prec);
    out << "PROPAGATOR_DUMP_END backend=" << backend
        << " batch_size=" << batch_size << "\n";
}

// MPI state-sharding: keep only rank r's contiguous slice of the selected-index list (low ranks
// take the remainder, so the union is exact with no gaps/overlap and the GLOBAL index values are
// preserved for the dump/merge). Identity for size<=1 (single-process unchanged). Never empty
// unless size>n.
std::vector<culindblad::Index> shard_indices(
    const std::vector<culindblad::Index>& all, int rank, int size)
{
    if (size <= 1) {
        return all;
    }
    const std::size_t n = all.size();
    const std::size_t base = n / static_cast<std::size_t>(size);
    const std::size_t rem = n % static_cast<std::size_t>(size);
    const std::size_t r = static_cast<std::size_t>(rank);
    const std::size_t begin = r * base + std::min<std::size_t>(r, rem);
    const std::size_t cnt = base + (r < rem ? 1 : 0);
    return std::vector<culindblad::Index>(all.begin() + begin, all.begin() + begin + cnt);
}

// Dump the states selected by config.state_selection (full propagator if
// evolve_all_states, otherwise the listed indices, e.g. computational states).
void tc_dump_states(
    const culindblad::TunableCouplerBenchmarkConfig& config,
    bool run_cpu,
    int mpi_rank = 0,
    int mpi_size = 1,
    const std::string& tag = "")
{
    using namespace culindblad;
    const Model model = build_tunable_coupler_model(config);
    const Solver solver = make_solver(model);

    const TransmonChainBenchmarkTiming gpu_timing =
        run_tunable_coupler_cuda_benchmark(config);

    // Multi-rank: each rank writes its shard to its own file (avoids stdout interleave) with STATE
    // numbered LOCALLY 0..cnt-1; merge_ranks.py concatenates rank files in rank order and renumbers
    // to the global 0..N-1, reconstructing the single-process ordering bit-identically.
    if (mpi_size > 1) {
        std::error_code ec;
        std::filesystem::create_directories("propcheck", ec);
        const std::string fname =
            "propcheck/" + tag + "_rank" + std::to_string(mpi_rank) + ".txt";
        std::ofstream f(fname);
        if (!f.is_open()) {
            // Hard-abort the whole job: a silently-missing shard would feed a wrong merged anchor.
            std::cerr << "[rank " << mpi_rank << "] cannot open " << fname << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
        tc_dump_one_propagator("gpu", config, solver, gpu_timing,
                               gpu_timing.final_states.size(), f);
        std::cerr << "[rank " << mpi_rank << "] wrote " << fname << " ("
                  << gpu_timing.final_states.size() << " states)" << std::endl;
        return;  // CPU reference + stdout dump are single-process only
    }

    tc_dump_one_propagator("gpu", config, solver, gpu_timing,
                           gpu_timing.final_states.size());

    if (run_cpu) {
        const TransmonChainBenchmarkTiming cpu_timing =
            run_tunable_coupler_cpu_benchmark(config);
        tc_dump_one_propagator("cpu", config, solver, cpu_timing,
                               cpu_timing.final_states.size());
    }
}

// Closed-system (Schrodinger) dump: each STATE is a length-hilbert_dim |psi(T)> vector. Tagged
// mode=schrodinger / state_len=hilbert_dim so the parser reads D entries, not D^2.
void tc_dump_states_schrodinger(
    const culindblad::TunableCouplerBenchmarkConfig& config,
    int mpi_rank = 0,
    int mpi_size = 1,
    const std::string& tag = "")
{
    using namespace culindblad;
    const Model model = build_tunable_coupler_model(config);
    const Solver solver = make_solver(model);
    const TransmonChainBenchmarkTiming timing =
        run_tunable_coupler_schrodinger_cuda_benchmark(config);

    // Rank-aware (mirrors tc_dump_states): per-rank file when size>1, std::cout unchanged for
    // size==1. STATE numbered locally; merge_ranks.py reconstructs the global order.
    std::ofstream rank_file;
    if (mpi_size > 1) {
        std::error_code ec;
        std::filesystem::create_directories("propcheck", ec);
        const std::string fname =
            "propcheck/" + tag + "_rank" + std::to_string(mpi_rank) + ".txt";
        rank_file.open(fname);
        if (!rank_file.is_open()) {
            std::cerr << "[rank " << mpi_rank << "] cannot open " << fname << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
    }
    std::ostream& out = (mpi_size > 1) ? rank_file : std::cout;

    out << "PROPAGATOR_DUMP_BEGIN backend=gpu mode=schrodinger"
        << " num_qubits=" << config.num_qubits
        << " num_couplers=" << config.num_couplers
        << " cutoff_dim=" << config.cutoff_dim
        << " hilbert_dim=" << solver.layout.hilbert_dim
        << " state_len=" << solver.layout.hilbert_dim
        << " steps=" << config.batched_num_steps
        << " batch_size=" << timing.final_states.size()
        << " num_states=" << timing.final_states.size() << "\n";
    const std::streamsize old_prec = out.precision();
    out << std::setprecision(17);
    for (std::size_t s = 0; s < timing.final_states.size(); ++s) {
        out << "STATE " << s << "\n";
        const std::vector<Complex>& psi = timing.final_states[s];
        for (std::size_t i = 0; i < psi.size(); ++i) {
            out << psi[i].real() << " " << psi[i].imag() << "\n";
        }
    }
    out << std::setprecision(old_prec);
    out << "PROPAGATOR_DUMP_END backend=gpu mode=schrodinger batch_size="
        << timing.final_states.size() << "\n";
    if (mpi_size > 1) {
        std::cerr << "[rank " << mpi_rank << "] wrote propcheck/" << tag << "_rank"
                  << mpi_rank << ".txt (" << timing.final_states.size() << " states)" << std::endl;
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

// ---- State-vector kernel self-test (Schrodinger solver, segment 2). -----------------------
// Validates the one-sided -i*H|psi> left-action kernels (sparse + diagonal) against a dense
// brute-force reference that embeds the target-site operator into the full D-space by digit
// decomposition -- independent of the grouped-layout index arithmetic the kernels use, so a
// stride/packing bug shows up. Invoked with -psi_selftest; exits after reporting.
namespace {
using culindblad::Complex;
using culindblad::Index;

// Mixed-radix digit of `full` at global site s (site-0-major full strides given).
inline Index site_digit(Index full, Index s, const std::vector<Index>& full_stride,
                        const std::vector<Index>& dims) {
    return (full / full_stride[s]) % dims[s];
}
// Pack the target/complement sub-index in build_desc convention (subset[0] most significant).
inline Index pack_subset(Index full, const std::vector<Index>& subset,
                         const std::vector<Index>& full_stride, const std::vector<Index>& dims) {
    Index v = 0;
    for (Index p = 0; p < subset.size(); ++p) {
        v = v * dims[subset[p]] + site_digit(full, subset[p], full_stride, dims);
    }
    return v;
}

// Reference -i * (H embedded on target_sites, identity on complement) * psi, for one state.
std::vector<Complex> brute_force_minus_i_H_psi(
    const std::vector<Index>& dims, const std::vector<Index>& target_sites,
    const std::vector<Complex>& target_op, Index target_dim, const std::vector<Complex>& psi) {
    const Index n = dims.size();
    Index D = 1; for (Index d : dims) D *= d;
    std::vector<Index> full_stride(n, 1);
    for (Index i = n; i-- > 1;) full_stride[i - 1] = full_stride[i] * dims[i];
    std::vector<bool> is_t(n, false); for (Index s : target_sites) is_t[s] = true;
    std::vector<Index> comp_sites; for (Index s = 0; s < n; ++s) if (!is_t[s]) comp_sites.push_back(s);
    std::vector<Complex> out(D, Complex(0.0, 0.0));
    const Complex minus_i(0.0, -1.0);
    for (Index fr = 0; fr < D; ++fr) {
        const Index tr = pack_subset(fr, target_sites, full_stride, dims);
        const Index cr = pack_subset(fr, comp_sites, full_stride, dims);
        Complex acc(0.0, 0.0);
        for (Index fc = 0; fc < D; ++fc) {
            if (pack_subset(fc, comp_sites, full_stride, dims) != cr) continue;  // identity on complement
            const Index tc = pack_subset(fc, target_sites, full_stride, dims);
            acc += target_op[tr * target_dim + tc] * psi[fc];
        }
        out[fr] = minus_i * acc;
    }
    return out;
}

// Segment 1 gate: round-trip a hand-written term-list JSON through load_model_from_json and assert
// the Model is populated element-exact (row-major), plus the dim + system/dissipator invariants fire.
int run_load_model_selftest() {
    using namespace culindblad;
    int fails = 0;
    auto check = [&](bool ok, const std::string& msg) {
        if (!ok) { std::cerr << "  [load_model_selftest] FAIL: " << msg << std::endl; ++fails; }
    };

    const std::string good_path = "/tmp/culindblad_model_selftest.json";
    const char* good = R"JSON({
      "schema_version": 1,
      "device": { "site_labels":["Q1","C1"], "local_dims":[2,2], "n_qubits":1, "n_couplers":1 },
      "system": "closed",
      "hamiltonian_terms": [
        { "name":"h1", "sites":[0], "row_dim":2, "col_dim":2,
          "matrix": { "real":[1,2,3,4], "imag":[0,0,0,0.5] } },
        { "name":"h2", "sites":[0,1], "row_dim":4, "col_dim":4,
          "matrix": { "real":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                      "imag":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] } }
      ],
      "dissipator_terms": [],
      "time_dependent_terms": [],
      "evolution": { "t0":0.0, "tfinal":1.0e-8, "num_steps":123, "rotating_frame":true,
                     "frame_omegas":[1.5, 2.5], "result_frame":"lab" },
      "initial_states": { "encoding":"computational", "num_states":2 }
    })JSON";
    { std::ofstream f(good_path); f << good; }

    try {
        const LoadedModel lm = load_model_from_json(good_path);
        check(lm.model.local_dims == std::vector<Index>({2, 2}), "local_dims");
        check(!lm.is_open, "system closed");
        check(lm.num_qubits == 1 && lm.num_couplers == 1, "n_qubits/couplers");
        check(lm.model.hamiltonian_terms.size() == 2, "num hamiltonian_terms");
        check(lm.model.dissipator_terms.empty(), "no dissipators (closed)");
        const OperatorTerm& h1 = lm.model.hamiltonian_terms.at(0);
        check(h1.kind == TermKind::Hamiltonian && h1.sites == std::vector<Index>({0}), "h1 kind/sites");
        check(h1.row_dim == 2 && h1.col_dim == 2 && h1.matrix.size() == 4, "h1 dims");
        // row-major [1,2,3,4] with imag[3]=0.5 -> matrix[3] == (4, 0.5)
        check(h1.matrix[0] == Complex(1, 0) && h1.matrix[3] == Complex(4, 0.5), "h1 row-major values");
        const OperatorTerm& h2 = lm.model.hamiltonian_terms.at(1);
        check(h2.sites == std::vector<Index>({0, 1}), "h2 sites");
        check(h2.row_dim == 4 && h2.col_dim == 4 && h2.matrix.size() == 16, "h2 dims");
        check(h2.matrix[0] == Complex(1, 0) && h2.matrix[15] == Complex(16, 0), "h2 endpoints");
        check(lm.evolution.num_steps == 123 && lm.evolution.rotating_frame, "evolution num_steps/frame");
        check(lm.evolution.frame_omegas == std::vector<double>({1.5, 2.5}), "frame_omegas");
        check(lm.evolution.result_frame == "lab", "result_frame");
        check(lm.initial_states.num_states == 2, "num_states");
    } catch (const std::exception& e) {
        std::cerr << "  [load_model_selftest] good doc threw: " << e.what() << std::endl;
        ++fails;
    }
    std::remove(good_path.c_str());

    // Invariant 1: system=="closed" with a dissipator must throw (valid dims so the dim check passes).
    auto expect_throw = [&](const char* doc, const std::string& msg) {
        const std::string p = "/tmp/culindblad_model_selftest_bad.json";
        { std::ofstream f(p); f << doc; }
        bool threw = false;
        try { (void)load_model_from_json(p); } catch (const std::exception&) { threw = true; }
        check(threw, msg);
        std::remove(p.c_str());
    };
    expect_throw(R"JSON({ "device":{"local_dims":[2]}, "system":"closed", "hamiltonian_terms":[],
      "dissipator_terms":[{"name":"d","sites":[0],"row_dim":2,"col_dim":2,
        "matrix":{"real":[0,0,0,0],"imag":[0,0,0,0]}}],
      "time_dependent_terms":[], "evolution":{"tfinal":1e-8}, "initial_states":{} })JSON",
      "closed + dissipator should throw");
    // Invariant 2: a block whose row/col_dim != product of site dims must throw (the convention guard).
    expect_throw(R"JSON({ "device":{"local_dims":[3,3]}, "system":"closed",
      "hamiltonian_terms":[{"name":"bad","sites":[0],"row_dim":2,"col_dim":2,
        "matrix":{"real":[0,0,0,0],"imag":[0,0,0,0]}}],
      "dissipator_terms":[], "time_dependent_terms":[], "evolution":{"tfinal":1e-8},
      "initial_states":{} })JSON",
      "row_dim != product of site dims should throw");

    if (fails == 0) std::cout << "  [load_model_selftest] PASS" << std::endl;
    return fails == 0 ? 0 : 1;
}

// Segment 2 gate: the pulse-envelope factory matches the Julia gate_sim.jl formulas at sample points,
// and the product-of-sqrt coupling coefficient is built correctly through the full loader.
int run_pulse_factory_selftest() {
    using namespace culindblad;
    int fails = 0;
    auto close = [&](double got, double want, const std::string& msg) {
        if (std::abs(got - want) > 1e-12) {
            std::cerr << "  [pulse_factory_selftest] FAIL " << msg << ": got "
                      << std::setprecision(17) << got << " want " << want << std::endl;
            ++fails;
        }
    };
    {
        EnvelopeSpec s; s.shape = "sine"; s.a = 2; s.b = 1; s.T = 10;
        const TimeScalarFunction f = make_envelope(s);
        close(f(5.0), 2.0, "sine t=5"); close(f(0.0), 1.0, "sine t=0");
        close(f(10.0), 1.0, "sine t=10"); close(f(2.5), 1.7071067811865475, "sine t=2.5");
    }
    {
        EnvelopeSpec s; s.shape = "flattop"; s.a = 2; s.b = 1; s.T = 10; s.tr = 2;
        const TimeScalarFunction f = make_envelope(s);
        close(f(5.0), 2.0, "flattop hold"); close(f(1.0), 1.5, "flattop ramp");
        close(f(0.0), 1.0, "flattop idle");
    }
    {
        EnvelopeSpec s; s.shape = "flat_top_gaussian"; s.a = 2; s.b = 1; s.t0 = 3; s.tau = 4; s.sigma = 1;
        close(make_envelope(s)(5.0), 1.9544997361036416, "flat_top_gaussian center");
    }
    {
        EnvelopeSpec s; s.shape = "slepian"; s.omega1 = 1; s.K = 1; s.cg = std::sqrt(2.0); s.T = 10;
        s.omega_c_idle = 99.0; s.ugrid = {0.0, 1.0}; s.thetagrid = {0.6, 0.6};
        const TimeScalarFunction f = make_envelope(s);
        close(f(5.0), 6.109428656516023, "slepian const-theta"); close(f(0.0), 99.0, "slepian idle");
    }
    // product_sqrt coupling coefficient through the full loader: sqrt(4)*sqrt(9) = 6 at t=5.
    {
        const std::string p = "/tmp/culindblad_pulse_selftest.json";
        const char* doc = R"JSON({
          "device":{"local_dims":[2,2],"n_qubits":1,"n_couplers":1}, "system":"closed",
          "hamiltonian_terms":[], "dissipator_terms":[],
          "time_dependent_terms":[
            {"name":"cpl","sites":[0,1],"row_dim":4,"col_dim":4,
             "matrix":{"real":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       "imag":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},
             "coefficient":{"kind":"product_sqrt","keys":["e1","e2"]}}],
          "envelopes":{"e1":{"shape":"sine","a":4,"b":4,"T":10},
                       "e2":{"shape":"sine","a":9,"b":9,"T":10}},
          "evolution":{"tfinal":10.0}, "initial_states":{"encoding":"computational","num_states":2} })JSON";
        { std::ofstream f(p); f << doc; }
        try {
            const LoadedModel lm = load_model_from_json(p);
            if (lm.model.time_dependent_hamiltonian_terms.size() != 1) {
                std::cerr << "  [pulse_factory_selftest] FAIL: expected 1 td term" << std::endl; ++fails;
            } else {
                close(lm.model.time_dependent_hamiltonian_terms[0].coefficient(5.0), 6.0,
                      "product_sqrt sqrt(4)*sqrt(9)");
            }
        } catch (const std::exception& e) {
            std::cerr << "  [pulse_factory_selftest] product_sqrt threw: " << e.what() << std::endl; ++fails;
        }
        std::remove(p.c_str());
    }
    // GRAPE seg1: analytic d c/d{a,b} matches central finite differences of make_envelope to ~1e-6.
    auto check_dparam = [&](EnvelopeSpec s, const std::string& p, double t) {
        const double dlt = 1e-5;
        EnvelopeSpec sp = s, sm = s;
        if (p == "a") { sp.a += dlt; sm.a -= dlt; } else { sp.b += dlt; sm.b -= dlt; }
        const double fd = (make_envelope(sp)(t) - make_envelope(sm)(t)) / (2.0 * dlt);
        const double an = make_envelope_dparam(s, p)(t);
        if (std::abs(an - fd) > 1e-6) {
            std::cerr << "  [pulse_factory_selftest] FAIL dparam " << s.shape << "/" << p
                      << " @t=" << t << ": analytic " << std::setprecision(12) << an
                      << " vs FD " << fd << std::endl;
            ++fails;
        }
    };
    { EnvelopeSpec s; s.shape = "sine"; s.a = 2; s.b = 1; s.T = 10;
      check_dparam(s, "a", 2.5); check_dparam(s, "b", 2.5); check_dparam(s, "a", 7.0); }
    { EnvelopeSpec s; s.shape = "flattop"; s.a = 2; s.b = 1; s.T = 10; s.tr = 2;
      check_dparam(s, "a", 1.0); check_dparam(s, "b", 5.0); check_dparam(s, "a", 9.0); }
    { EnvelopeSpec s; s.shape = "flat_top_gaussian"; s.a = 2; s.b = 1; s.t0 = 3; s.tau = 4; s.sigma = 1;
      check_dparam(s, "a", 5.0); check_dparam(s, "b", 5.0); }
    // deferred derivatives must throw (not silently return wrong values)
    { EnvelopeSpec s; s.shape = "sine"; s.a = 2; s.b = 1; s.T = 10;
      bool threw = false; try { (void)make_envelope_dparam(s, "T"); } catch (...) { threw = true; }
      if (!threw) { std::cerr << "  [pulse_factory_selftest] FAIL: dparam 'T' should throw\n"; ++fails; } }
    { EnvelopeSpec s; s.shape = "slepian"; s.omega1 = 1; s.K = 1; s.cg = std::sqrt(2.0); s.T = 10;
      s.omega_c_idle = 99; s.ugrid = {0.0, 1.0}; s.thetagrid = {0.6, 0.6};
      bool threw = false; try { (void)make_envelope_dparam(s, "a"); } catch (...) { threw = true; }
      if (!threw) { std::cerr << "  [pulse_factory_selftest] FAIL: slepian dparam should throw\n"; ++fails; } }

    if (fails == 0) std::cout << "  [pulse_factory_selftest] PASS" << std::endl;
    return fails == 0 ? 0 : 1;
}

// Dump the loaded-model evolution result in the PROPAGATOR_DUMP text format (lab-frame; the loaded
// model is already lab-frame and evolved in lab, so no back-transform). Parser-compatible with the
// existing verify_*.py / golden_regression.py readers.
void dump_loaded_model_result(std::ostream& out, const culindblad::LoadedModel& lm,
                              const culindblad::TransmonChainBenchmarkTiming& timing,
                              const std::string& mode) {
    using namespace culindblad;
    Index D = 1;
    for (const Index d : lm.model.local_dims) D *= d;
    out << "PROPAGATOR_DUMP_BEGIN backend=gpu mode=" << mode
        << " num_qubits=" << lm.num_qubits << " num_couplers=" << lm.num_couplers
        << " hilbert_dim=" << D;
    if (mode == "schrodinger") {
        out << " state_len=" << D;
    } else {
        out << " density_dim=" << (D * D);
    }
    out << " steps=" << lm.evolution.num_steps
        << " result_frame=" << lm.evolution.result_frame
        << " batch_size=" << timing.final_states.size()
        << " num_states=" << timing.final_states.size() << "\n";
    // Rotating-frame back-transform to lab: |psi>_lab[k] = e^{-i E_k T} |psi>_R[k] (closed) and
    // rho_lab[k,l] = e^{-i (E_k - E_l) T} rho_R[k,l] (open), with E_k = sum_s frame_omegas[s]*occ_s(k)
    // (site-0-major) and T = tfinal - t0. No-op unless result_frame=="rotating".
    std::vector<double> Efr;
    if (lm.evolution.result_frame == "rotating" && !lm.evolution.frame_omegas.empty()) {
        Efr.resize(static_cast<std::size_t>(D));
        const std::vector<double>& fo = lm.evolution.frame_omegas;
        const std::vector<Index>& dims = lm.model.local_dims;
        for (Index k = 0; k < D; ++k) {
            double e = 0.0; Index rem = k, stride = D;
            for (std::size_t si = 0; si < dims.size(); ++si) {
                stride /= dims[si];
                e += fo[si] * static_cast<double>(rem / stride);
                rem %= stride;
            }
            Efr[static_cast<std::size_t>(k)] = e;
        }
    }
    const double Tbt = lm.evolution.tfinal - lm.evolution.t0;
    const std::streamsize old_prec = out.precision();
    out << std::setprecision(17);
    for (std::size_t s = 0; s < timing.final_states.size(); ++s) {
        out << "STATE " << s << "\n";
        const std::vector<Complex>& st = timing.final_states[s];
        if (Efr.empty()) {
            for (const Complex& z : st) out << z.real() << " " << z.imag() << "\n";
        } else if (mode == "schrodinger") {
            for (Index k = 0; k < D; ++k) {
                const double ph = -Efr[static_cast<std::size_t>(k)] * Tbt;
                const Complex z = st[static_cast<std::size_t>(k)] * Complex(std::cos(ph), std::sin(ph));
                out << z.real() << " " << z.imag() << "\n";
            }
        } else {  // lindblad: row-major D x D
            for (Index k = 0; k < D; ++k)
                for (Index l = 0; l < D; ++l) {
                    const double ph = -(Efr[static_cast<std::size_t>(k)] - Efr[static_cast<std::size_t>(l)]) * Tbt;
                    const Complex z = st[static_cast<std::size_t>(k * D + l)] * Complex(std::cos(ph), std::sin(ph));
                    out << z.real() << " " << z.imag() << "\n";
                }
        }
    }
    out << std::setprecision(old_prec);
    out << "PROPAGATOR_DUMP_END backend=gpu mode=" << mode << "\n";
}

// -load_model <json> [-results_out <path>]: deserialize a Julia-exported term-list model, evolve the
// computational basis in the LAB frame (closed |psi> or open rho per the "system" field), and dump.
int run_loaded_model(const std::string& model_path, const std::string& results_path) {
    using namespace culindblad;
    try {
        const LoadedModel lm = load_model_from_json(model_path);
        const int num_steps = lm.evolution.num_steps > 0 ? lm.evolution.num_steps : 1000;
        std::ofstream fout;
        std::ostream* out = &std::cout;
        if (!results_path.empty()) {
            fout.open(results_path);
            if (!fout.is_open()) {
                std::cerr << "[load_model] cannot open results file: " << results_path << std::endl;
                return 1;
            }
            out = &fout;
        }
        std::cerr << "[load_model] " << (lm.is_open ? "open (Lindblad rho)" : "closed (Schrodinger psi)")
                  << ", " << lm.model.local_dims.size() << " sites, num_qubits=" << lm.num_qubits
                  << ", num_steps=" << num_steps
                  << ", terms=" << lm.model.hamiltonian_terms.size() << "+"
                  << lm.model.time_dependent_hamiltonian_terms.size() << "td" << std::endl;
        const std::vector<std::vector<Complex>>& explicit_states = lm.initial_states.explicit_states;
        const bool use_explicit = !explicit_states.empty();
        if (use_explicit) {
            std::cerr << "[load_model] explicit initial states: " << explicit_states.size()
                      << " (e.g. dressed comp kets)" << std::endl;
        }
        if (lm.is_open) {
            const std::vector<Index> idx = use_explicit
                ? std::vector<Index>{}
                : computational_density_indices_for_dims(lm.model.local_dims, lm.num_qubits);
            const TransmonChainBenchmarkTiming timing = evolve_loaded_model_liouvillian(
                lm.model, idx, explicit_states, lm.evolution.t0, lm.evolution.tfinal, num_steps);
            std::cerr << "[load_model] evolution wall_seconds=" << timing.wall_seconds
                      << " (states=" << timing.final_states.size() << ", steps=" << num_steps << ")"
                      << std::endl;
            dump_loaded_model_result(*out, lm, timing, "lindblad");
        } else {
            const std::vector<Index> idx = use_explicit
                ? std::vector<Index>{}
                : computational_state_indices_for_dims(lm.model.local_dims, lm.num_qubits);
            const TransmonChainBenchmarkTiming timing = evolve_loaded_model_schrodinger(
                lm.model, idx, explicit_states, lm.evolution.t0, lm.evolution.tfinal, num_steps);
            std::cerr << "[load_model] evolution wall_seconds=" << timing.wall_seconds
                      << " (states=" << timing.final_states.size() << ", steps=" << num_steps << ")"
                      << std::endl;
            dump_loaded_model_result(*out, lm, timing, "schrodinger");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[load_model] error: " << e.what() << std::endl;
        return 1;
    }
}

int run_psi_kernel_selftest() {
    using culindblad::CudaGroupedStateLayout;
    using culindblad::GroupedStateLayout;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto rand_vec = [&](Index k) { std::vector<Complex> v(k);
        for (auto& z : v) z = Complex(u(rng), u(rng)); return v; };

    struct Case { std::vector<Index> dims; std::vector<Index> target; bool diagonal; };
    const std::vector<Case> cases = {
        {{3, 3, 3}, {0, 1}, false},   // 2-site sparse, complement {2} (comp_dim=3)
        {{2, 2, 2}, {1, 2}, false},   // 2-site sparse, complement {0}
        {{3, 3, 3}, {1, 2}, false},   // 2-site sparse, complement {0} (comp_dim=3)
        {{3, 3}, {0}, true},          // single-site diagonal, complement {1}
        {{2, 2, 2}, {1}, true},       // single-site diagonal, complement {0,2}
    };
    const Index batch = 2;
    double worst = 0.0; bool ok = true;
    for (const Case& c : cases) {
        GroupedStateLayout host_layout = culindblad::make_grouped_state_layout(c.target, c.dims, false);
        CudaGroupedStateLayout cuda_layout{};
        if (!culindblad::create_cuda_psi_grouped_state_layout(host_layout, cuda_layout)) {
            std::cerr << "  layout creation failed\n"; return 1;
        }
        const Index D = host_layout.hilbert_dim;
        Index target_dim = 1; for (Index s : c.target) target_dim *= c.dims[s];

        std::vector<Complex> psi = rand_vec(D * batch);
        std::vector<Complex> target_op = rand_vec(target_dim * target_dim);
        if (c.diagonal) {  // zero off-diagonals; the diagonal kernel only sees the diagonal
            for (Index i = 0; i < target_dim; ++i)
                for (Index j = 0; j < target_dim; ++j)
                    if (i != j) target_op[i * target_dim + j] = Complex(0.0, 0.0);
        }

        Complex *d_psi = nullptr, *d_grp = nullptr, *d_f = nullptr;
        cudaMalloc(&d_psi, D * batch * sizeof(Complex));
        cudaMalloc(&d_grp, D * batch * sizeof(Complex));
        cudaMalloc(&d_f, D * batch * sizeof(Complex));
        cudaMemcpy(d_psi, psi.data(), D * batch * sizeof(Complex), cudaMemcpyHostToDevice);
        cudaMemset(d_f, 0, D * batch * sizeof(Complex));
        culindblad::launch_flat_to_grouped_vec_batched_kernel(cuda_layout, batch, d_psi, d_grp, 0);

        if (c.diagonal) {
            std::vector<Complex> h_diag(target_dim);
            for (Index i = 0; i < target_dim; ++i) h_diag[i] = target_op[i * target_dim + i];
            Complex* d_diag = nullptr; cudaMalloc(&d_diag, target_dim * sizeof(Complex));
            cudaMemcpy(d_diag, h_diag.data(), target_dim * sizeof(Complex), cudaMemcpyHostToDevice);
            culindblad::launch_grouped_diagonal_left_to_flat_kernel(cuda_layout, d_diag, batch, d_grp, d_f, 0);
            cudaFree(d_diag);
        } else {
            std::vector<Complex> vals; std::vector<Index> rows, cols;
            for (Index i = 0; i < target_dim; ++i)
                for (Index j = 0; j < target_dim; ++j) {
                    vals.push_back(target_op[i * target_dim + j]); rows.push_back(i); cols.push_back(j);
                }
            Complex* d_vals = nullptr; Index *d_rows = nullptr, *d_cols = nullptr;
            cudaMalloc(&d_vals, vals.size() * sizeof(Complex));
            cudaMalloc(&d_rows, rows.size() * sizeof(Index));
            cudaMalloc(&d_cols, cols.size() * sizeof(Index));
            cudaMemcpy(d_vals, vals.data(), vals.size() * sizeof(Complex), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(Index), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(Index), cudaMemcpyHostToDevice);
            culindblad::launch_grouped_sparse_left_to_flat_kernel(
                cuda_layout, d_vals, d_rows, d_cols, static_cast<Index>(vals.size()),
                batch, d_grp, d_f, 0);
            cudaFree(d_vals); cudaFree(d_rows); cudaFree(d_cols);
        }
        cudaDeviceSynchronize();
        std::vector<Complex> f(D * batch);
        cudaMemcpy(f.data(), d_f, D * batch * sizeof(Complex), cudaMemcpyDeviceToHost);
        cudaFree(d_psi); cudaFree(d_grp); cudaFree(d_f);
        culindblad::destroy_cuda_grouped_state_layout(cuda_layout);

        for (Index b = 0; b < batch; ++b) {
            std::vector<Complex> psi_b(psi.begin() + b * D, psi.begin() + (b + 1) * D);
            std::vector<Complex> ref = brute_force_minus_i_H_psi(c.dims, c.target, target_op, target_dim, psi_b);
            for (Index i = 0; i < D; ++i)
                worst = std::max(worst, std::abs(f[b * D + i] - ref[i]));
        }
        std::cout << "  case dims=" << c.dims.size() << " target=" << c.target.size()
                  << "site " << (c.diagonal ? "diag" : "sparse") << " D=" << D
                  << " worst-so-far=" << worst << "\n";
    }
    std::cout << "PSI KERNEL SELFTEST worst |gpu - bruteforce| = " << worst
              << (worst < 1e-12 ? "   PASS" : "   FAIL") << std::endl;
    return worst < 1e-12 ? 0 : 1;
}

// Full Schrodinger RHS check: build a 2Q1C driven model, call ts_rhs_function_cuda_full_model_
// schrodinger, and compare to a CPU -i*H_full(t)|psi> assembled term-by-term (static + time-
// dependent), at t=0 and mid-gate. Exercises regroup + effective-H assembly + diagonal/sparse
// dispatch through the real context, including the dynamic (driven) path.
int run_psi_rhs_selftest() {
    using namespace culindblad;
    TunableCouplerBenchmarkConfig cfg{};
    cfg.num_qubits = 2; cfg.num_couplers = 1; cfg.cutoff_dim = 3;
    cfg.omega = {5.0, 5.2, 4.8}; cfg.alpha = {-0.2, -0.2, -0.2};
    cfg.t1 = {1.0e4, 1.0e4, 1.0e4}; cfg.tphi = {1.0e4, 1.0e4, 1.0e4};
    cfg.g = 0.02; cfg.t0 = 0.0; cfg.tfinal = 15.0;
    cfg.enable_drives = true; cfg.driven_sites = {0};
    cfg.drive_amplitude = 0.1; cfg.drive_t_gate = 15.0;
    Model model = build_tunable_coupler_model(cfg);
    Solver solver = make_solver(model);
    const std::vector<Index>& dims = solver.model.local_dims;
    Index D = 1; for (Index d : dims) D *= d;
    const Index batch = 2;

    CudaBatchExecutionContext ctx{};
    if (create_cuda_state_vector_execution_context(solver, batch, ctx) != 0) {
        std::cerr << "  psi context creation failed\n"; return 1;
    }
    std::mt19937 rng(999); std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<Complex> psi(D * batch);
    for (auto& z : psi) z = Complex(u(rng), u(rng));
    {
        PetscScalar* arr = nullptr; VecGetArray(ctx.x, &arr);
        for (Index i = 0; i < D * batch; ++i) arr[i] = psi[i];
        VecRestoreArray(ctx.x, &arr);
    }
    Vec f = nullptr; VecDuplicate(ctx.x, &f);

    double worst = 0.0;
    for (double t : {0.0, 7.5}) {
        ts_rhs_function_cuda_full_model_schrodinger(nullptr, t, ctx.x, f, &ctx.rhs_ctx);
        const PetscScalar* farr = nullptr; VecGetArrayRead(f, &farr);
        for (Index b = 0; b < batch; ++b) {
            std::vector<Complex> psi_b(psi.begin() + b * D, psi.begin() + (b + 1) * D);
            std::vector<Complex> ref(D, Complex(0.0, 0.0));
            auto add_term = [&](const std::vector<Index>& sites, const std::vector<Complex>& mat,
                                Index rdim, double scale) {
                std::vector<Complex> scaled(mat.size());
                for (std::size_t k = 0; k < mat.size(); ++k) scaled[k] = mat[k] * scale;
                std::vector<Complex> c = brute_force_minus_i_H_psi(dims, sites, scaled, rdim, psi_b);
                for (Index i = 0; i < D; ++i) ref[i] += c[i];
            };
            for (const OperatorTerm& h : solver.model.hamiltonian_terms)
                add_term(h.sites, h.matrix, h.row_dim, 1.0);
            for (const TimeDependentTerm& td : solver.model.time_dependent_hamiltonian_terms)
                add_term(td.sites, td.matrix, td.rows, evaluate_time_dependent_coefficient(td, t));
            for (Index i = 0; i < D; ++i)
                worst = std::max(worst, std::abs(static_cast<Complex>(farr[b * D + i]) - ref[i]));
        }
        VecRestoreArrayRead(f, &farr);
    }
    VecDestroy(&f);
    destroy_cuda_batch_execution_context(ctx);
    std::cout << "PSI RHS SELFTEST (2Q1C lab+drive, t=0 & 7.5) worst |gpu - cpu| = " << worst
              << (worst < 1e-9 ? "   PASS" : "   FAIL") << std::endl;
    return worst < 1e-9 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv)
{
    using namespace culindblad;

    // ---- MPI one-rank-per-GPU device binding (state-sharding) -------------------------------
    // CORRECTNESS MECHANISM (verified against the PETSc source): PETSc captures its CUDA device
    // ONCE, inside PetscInitialize (PetscDeviceInitializeFromOptions_Internal). For the default
    // (PETSC_DECIDE) it uses MPI_Comm_rank(PETSC_COMM_WORLD)%ndev -- EXCEPT it first honors the
    // env pair {TORCHELASTIC_RUN_ID, LOCAL_RANK}. The capture cannot be changed afterward (a later
    // PetscOptionsSetValue("-device_select") is a no-op). So to make PETSc's VECCUDA land on the
    // SAME physical GPU as the app's raw cudaMalloc/cuTENSOR, we (1) clear that env override BEFORE
    // PetscInitialize so PETSc deterministically picks rank%ndev, and (2) call cudaSetDevice with
    // the identical rank%ndev for the app's raw CUDA. Both then equal rank%ndev on PETSC_COMM_WORLD.
    // No-op for single-process (rank0 -> dev0); single-process behavior is byte-for-byte unchanged.
    unsetenv("LOCAL_RANK");
    unsetenv("TORCHELASTIC_RUN_ID");

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr != 0) {
        std::cerr << "PetscInitialize failed." << std::endl;
        return 1;
    }

    PetscMPIInt mpi_rank = 0;
    PetscMPIInt mpi_size = 1;
    PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank));
    PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
    {
        int ndev = 0;
        const cudaError_t cerr = cudaGetDeviceCount(&ndev);
        if (cerr != cudaSuccess || ndev <= 0) {
            std::cerr << "[rank " << mpi_rank << "] cudaGetDeviceCount failed ("
                      << cudaGetErrorString(cerr) << ") or no GPU visible." << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
        const int my_dev = static_cast<int>(mpi_rank) % ndev;
        if (cudaSetDevice(my_dev) != cudaSuccess) {
            std::cerr << "[rank " << mpi_rank << "] cudaSetDevice(" << my_dev << ") failed."
                      << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
        // Per-rank device log so a real multi-GPU run can be verified to have landed on DISTINCT
        // physical GPUs (rank%ndev is fragile under per-rank CUDA_VISIBLE_DEVICES masking).
        if (mpi_size > 1) {
            char pci[64] = "?";
            cudaDeviceGetPCIBusId(pci, static_cast<int>(sizeof(pci)), my_dev);
            char host[256] = {};
            if (gethostname(host, sizeof(host) - 1) != 0) std::snprintf(host, sizeof(host), "?");
            std::cerr << "[rank " << mpi_rank << "/" << mpi_size << "] host=" << host
                      << " ndev=" << ndev << " -> GPU " << my_dev << " (PCI " << pci << ")"
                      << std::endl;
        }
    }

    PetscBool psi_selftest = PETSC_FALSE;
    PetscOptionsGetBool(nullptr, nullptr, "-psi_selftest", &psi_selftest, nullptr);
    if (psi_selftest == PETSC_TRUE) {
        const int rc_kernel = run_psi_kernel_selftest();
        const int rc_rhs = run_psi_rhs_selftest();
        const int rc_evolve = culindblad::run_psi_evolve_schrodinger_selftest();
        PetscFinalize();
        return (rc_kernel == 0 && rc_rhs == 0 && rc_evolve == 0) ? 0 : 1;
    }

    PetscBool load_model_selftest = PETSC_FALSE;
    PetscOptionsGetBool(nullptr, nullptr, "-load_model_selftest", &load_model_selftest, nullptr);
    if (load_model_selftest == PETSC_TRUE) {
        const int rc_des = run_load_model_selftest();
        const int rc_pulse = run_pulse_factory_selftest();
        PetscFinalize();
        return (rc_des == 0 && rc_pulse == 0) ? 0 : 1;
    }

    char model_path_buf[1024] = {};
    PetscBool model_set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, "-load_model",
                          model_path_buf, sizeof(model_path_buf), &model_set);
    if (model_set == PETSC_TRUE) {
        char results_buf[1024] = {};
        PetscBool results_set = PETSC_FALSE;
        PetscOptionsGetString(nullptr, nullptr, "-results_out",
                              results_buf, sizeof(results_buf), &results_set);
        const int rc = run_loaded_model(
            std::string(model_path_buf),
            results_set == PETSC_TRUE ? std::string(results_buf) : std::string());
        PetscFinalize();
        return rc;
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
            PetscBool tc_rotating = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                nullptr, nullptr, "-tc_rotating_frame", &tc_rotating, nullptr));
            PetscBool tc_drop_cr = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                nullptr, nullptr, "-tc_drop_counter_rotating", &tc_drop_cr, nullptr));
            PetscBool tc_realistic = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                nullptr, nullptr, "-tc_realistic_freqs", &tc_realistic, nullptr));
            s_tc_realistic_freqs = (tc_realistic == PETSC_TRUE);
            PetscBool tc_schrod = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                nullptr, nullptr, "-tc_schrodinger", &tc_schrod, nullptr));

            TunableCouplerBenchmarkConfig tc_config =
                make_tunable_coupler_benchmark_config(
                    static_cast<Index>(tc_nq),
                    static_cast<Index>(tc_nc),
                    static_cast<Index>(tc_cd),
                    steps,
                    tc_drives == PETSC_TRUE);
            tc_config.enable_rotating_frame = (tc_rotating == PETSC_TRUE);
            tc_config.drop_counter_rotating = (tc_drop_cr == PETSC_TRUE);

            char tc_batch_buf[256] = {};
            PetscBool tc_batch_set = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetString(
                nullptr, nullptr, "-dump_batch_sizes",
                tc_batch_buf, sizeof(tc_batch_buf), &tc_batch_set));

            // Per-rank dump tag (multi-rank writes propcheck/<tag>_rank<r>.txt).
            char tc_tag_buf[128] = "tc_shard";
            PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetString(
                nullptr, nullptr, "-dump_tag", tc_tag_buf, sizeof(tc_tag_buf), nullptr));

            // Computational-subspace states (qubits in {0,1}, couplers in |0>) vs the
            // full propagator (all density_dim basis states).
            if (tc_comp == PETSC_TRUE) {
                tc_config.state_selection.evolve_all_states = false;
                tc_config.state_selection.selected_state_indices =
                    (tc_schrod == PETSC_TRUE)
                        ? tunable_coupler_computational_state_indices(
                              static_cast<Index>(tc_nq),
                              static_cast<Index>(tc_nc),
                              static_cast<Index>(tc_cd))
                        : tunable_coupler_computational_density_indices(
                              static_cast<Index>(tc_nq),
                              static_cast<Index>(tc_nc),
                              static_cast<Index>(tc_cd));
            } else {
                tc_config.state_selection.evolve_all_states = true;
                tc_config.state_selection.selected_state_indices.clear();
            }

            // MPI sharding only shards the EXPLICIT computational selection. The full-propagator
            // (evolve_all_states) and batch-sweep paths regenerate their own index lists and would
            // have every rank evolve the full set (duplicates) and dump to interleaved stdout.
            // Reject them under multi-rank rather than silently corrupt the merged anchor.
            if (mpi_size > 1 && tc_comp != PETSC_TRUE) {
                if (mpi_rank == 0) {
                    std::cerr << "multi-rank -dump_tc requires -tc_computational "
                                 "(full-propagator / batch-sweep dumps are single-process only)"
                              << std::endl;
                }
                MPI_Abort(PETSC_COMM_WORLD, 1);
            }

            // Guard the empty-shard misconfiguration (more ranks than states): a 0-length slice
            // makes the density path throw and the Schrodinger path SILENTLY regenerate the full
            // set. Abort with a clear message rather than corrupt the merge.
            if (mpi_size > 1 &&
                static_cast<std::size_t>(mpi_size) >
                    tc_config.state_selection.selected_state_indices.size()) {
                if (mpi_rank == 0) {
                    std::cerr << "too many ranks (" << mpi_size << ") for "
                              << tc_config.state_selection.selected_state_indices.size()
                              << " computational states -- use <= that many ranks" << std::endl;
                }
                MPI_Abort(PETSC_COMM_WORLD, 1);
            }

            // MPI state-sharding: each rank evolves a disjoint contiguous slice of the selected
            // computational states (identity for single-process). The solver/run-entry are
            // unchanged -- they size the batch to the (now per-rank) selected_state_indices.
            tc_config.state_selection.selected_state_indices = shard_indices(
                tc_config.state_selection.selected_state_indices, mpi_rank, mpi_size);
            if (mpi_size > 1) {
                std::cerr << "[rank " << mpi_rank << "/" << mpi_size << "] evolving "
                          << tc_config.state_selection.selected_state_indices.size()
                          << " state(s)" << std::endl;
            }

            if (tc_schrod == PETSC_TRUE) {
                tc_dump_states_schrodinger(tc_config,
                                           static_cast<int>(mpi_rank), static_cast<int>(mpi_size),
                                           std::string(tc_tag_buf));
            } else if (tc_batch_set == PETSC_TRUE && tc_comp != PETSC_TRUE) {
                tc_dump_batch_sweep(tc_config, parse_batch_sizes_csv(tc_batch_buf));
            } else {
                PetscBool dump_cpu = PETSC_FALSE;
                PetscCallAbort(PETSC_COMM_SELF, PetscOptionsGetBool(
                    nullptr, nullptr, "-dump_propagator_cpu", &dump_cpu, nullptr));
                tc_dump_states(tc_config, dump_cpu == PETSC_TRUE,
                               static_cast<int>(mpi_rank), static_cast<int>(mpi_size),
                               std::string(tc_tag_buf));
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
