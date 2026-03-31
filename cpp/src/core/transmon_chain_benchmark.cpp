#include "culindblad/transmon_chain_benchmark.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <petscts.h>
#include <petscvec.h>

#include <cuda_runtime.h>

#include "culindblad/backend.hpp"
#include "culindblad/batch_evolution.hpp"
#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/state_buffer.hpp"
#include "culindblad/time_dependence.hpp"
#include "culindblad/time_dependent_term.hpp"

namespace culindblad {

namespace {

struct CpuTsRhsContext {
    const Solver* solver = nullptr;
};

struct CpuBenchmarkExecutionContext {
    TS ts = nullptr;
    Vec x = nullptr;
    CpuTsRhsContext rhs_ctx{};
    bool initialized = false;
};

PetscErrorCode ts_rhs_function_cpu_full_model_liouvillian(
    TS,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<CpuTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;

    const PetscScalar* x_ptr = nullptr;
    PetscScalar* f_ptr = nullptr;

    PetscCall(VecGetArrayRead(x, &x_ptr));
    PetscCall(VecGetArray(f, &f_ptr));

    ConstStateBuffer in_buf{
        reinterpret_cast<const Complex*>(x_ptr),
        solver.layout.density_dim
    };

    StateBuffer out_buf{
        reinterpret_cast<Complex*>(f_ptr),
        solver.layout.density_dim
    };

    apply_liouvillian_at_time(
        solver,
        static_cast<double>(t),
        in_buf,
        out_buf);

    PetscCall(VecRestoreArray(f, &f_ptr));
    PetscCall(VecRestoreArrayRead(x, &x_ptr));
    return 0;
}

PetscErrorCode create_cpu_benchmark_execution_context(
    const Solver& solver,
    CpuBenchmarkExecutionContext& ctx)
{
    ctx.ts = nullptr;
    ctx.x = nullptr;
    ctx.initialized = false;
    ctx.rhs_ctx.solver = &solver;

    PetscCall(TSCreate(PETSC_COMM_SELF, &ctx.ts));
    PetscCall(TSSetType(ctx.ts, TSRK));

    PetscCall(VecCreate(PETSC_COMM_SELF, &ctx.x));
    PetscCall(VecSetSizes(ctx.x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetFromOptions(ctx.x));
    PetscCall(VecSet(ctx.x, 0.0));

    PetscCall(TSSetRHSFunction(
        ctx.ts,
        nullptr,
        ts_rhs_function_cpu_full_model_liouvillian,
        &ctx.rhs_ctx));

    PetscCall(TSSetMaxSteps(ctx.ts, 1000000));
    PetscCall(TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ctx.ts));

    ctx.initialized = true;
    return 0;
}

PetscErrorCode destroy_cpu_benchmark_execution_context(
    CpuBenchmarkExecutionContext& ctx)
{
    if (!ctx.initialized) {
        return 0;
    }

    if (ctx.x != nullptr) {
        PetscCall(VecDestroy(&ctx.x));
    }

    if (ctx.ts != nullptr) {
        PetscCall(TSDestroy(&ctx.ts));
    }

    ctx.initialized = false;
    return 0;
}

std::vector<Complex> evolve_single_state_cpu_ts(
    const Solver& solver,
    const std::vector<Complex>& initial_state,
    double t0,
    double tfinal,
    CpuBenchmarkExecutionContext& ctx)
{
    if (!ctx.initialized) {
        throw std::runtime_error(
            "evolve_single_state_cpu_ts: CPU benchmark context not initialized");
    }

    PetscCallAbort(PETSC_COMM_SELF, VecSet(ctx.x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        x_ptr[i] = initial_state[i];
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    const double dt_initial = (tfinal - t0) / 1000.0;

    PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, t0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt_initial));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxTime(ctx.ts, tfinal));
    PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));

    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    std::vector<Complex> final_state(solver.layout.density_dim, Complex{0.0, 0.0});
    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        final_state[i] = x_ptr[i];
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    return final_state;
}

std::vector<Complex> make_local_annihilation_operator(Index d)
{
    std::vector<Complex> a(d * d, Complex{0.0, 0.0});
    for (Index n = 1; n < d; ++n) {
        a[(n - 1) * d + n] = Complex{std::sqrt(static_cast<double>(n)), 0.0};
    }
    return a;
}

std::vector<Complex> make_local_creation_operator(Index d)
{
    std::vector<Complex> adag(d * d, Complex{0.0, 0.0});
    for (Index n = 1; n < d; ++n) {
        adag[n * d + (n - 1)] = Complex{std::sqrt(static_cast<double>(n)), 0.0};
    }
    return adag;
}

std::vector<Complex> make_local_number_operator(Index d)
{
    std::vector<Complex> n_op(d * d, Complex{0.0, 0.0});
    for (Index n = 0; n < d; ++n) {
        n_op[n * d + n] = Complex{static_cast<double>(n), 0.0};
    }
    return n_op;
}

std::vector<Complex> make_local_number_squared_shifted_operator(Index d)
{
    std::vector<Complex> op(d * d, Complex{0.0, 0.0});
    for (Index n = 0; n < d; ++n) {
        const double value =
            0.5 * static_cast<double>(n) * static_cast<double>(n - 1);
        op[n * d + n] = Complex{value, 0.0};
    }
    return op;
}

std::vector<Complex> make_two_site_hopping_operator(Index d)
{
    const Index dd = d * d;
    std::vector<Complex> op(dd * dd, Complex{0.0, 0.0});

    auto flatten = [d](Index left, Index right) {
        return left * d + right;
    };

    for (Index n0 = 0; n0 < d; ++n0) {
        for (Index n1 = 0; n1 < d; ++n1) {
            const Index col = flatten(n0, n1);

            if (n0 + 1 < d && n1 > 0) {
                const Index row = flatten(n0 + 1, n1 - 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0 + 1)) *
                    std::sqrt(static_cast<double>(n1));
                op[row * dd + col] += Complex{coeff, 0.0};
            }

            if (n0 > 0 && n1 + 1 < d) {
                const Index row = flatten(n0 - 1, n1 + 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0)) *
                    std::sqrt(static_cast<double>(n1 + 1));
                op[row * dd + col] += Complex{coeff, 0.0};
            }
        }
    }

    return op;
}

std::vector<Complex> add_operators(
    const std::vector<Complex>& a,
    const std::vector<Complex>& b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("add_operators: size mismatch");
    }

    std::vector<Complex> out(a.size(), Complex{0.0, 0.0});
    for (Index i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<Complex> subtract_operators(
    const std::vector<Complex>& a,
    const std::vector<Complex>& b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("subtract_operators: size mismatch");
    }

    std::vector<Complex> out(a.size(), Complex{0.0, 0.0});
    for (Index i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

std::vector<Complex> scale_operator(
    const std::vector<Complex>& a,
    Complex scale)
{
    std::vector<Complex> out(a.size(), Complex{0.0, 0.0});
    for (Index i = 0; i < a.size(); ++i) {
        out[i] = scale * a[i];
    }
    return out;
}

TimeScalarFunction make_gaussian_cosine_drive(
    double amplitude,
    double sigma,
    double center,
    double frequency)
{
    return [=](double t) {
        const double x = (t - center) / sigma;
        const double envelope = amplitude * std::exp(-0.5 * x * x);
        return envelope * std::cos(frequency * t);
    };
}

TimeScalarFunction make_gaussian_sine_drive(
    double amplitude,
    double sigma,
    double center,
    double frequency)
{
    return [=](double t) {
        const double x = (t - center) / sigma;
        const double envelope = amplitude * std::exp(-0.5 * x * x);
        return envelope * std::sin(frequency * t);
    };
}

std::vector<std::vector<Complex>> make_selected_basis_density_states(
    const Solver& solver,
    const std::vector<Index>& selected_state_indices)
{
    std::vector<std::vector<Complex>> states;
    states.reserve(selected_state_indices.size());

    for (Index state_index : selected_state_indices) {
        if (state_index >= solver.layout.density_dim) {
            throw std::invalid_argument(
                "make_selected_basis_density_states: selected state index out of range");
        }

        std::vector<Complex> rho(
            solver.layout.density_dim,
            Complex{0.0, 0.0});
        rho[state_index] = Complex{1.0, 0.0};
        states.push_back(std::move(rho));
    }

    return states;
}

std::vector<Complex> evolve_single_state_cuda_ts(
    const Solver& solver,
    const std::vector<Complex>& initial_state,
    double t0,
    double tfinal,
    CudaBatchExecutionContext& ctx)
{
    if (!ctx.initialized || ctx.ts == nullptr || ctx.x == nullptr) {
        throw std::runtime_error(
            "evolve_single_state_cuda_ts: CUDA benchmark context not initialized");
    }

    PetscScalar* x_ptr = nullptr;
#if defined(PETSC_HAVE_CUDA)
    PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayWrite(ctx.x, &x_ptr));
#else
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
#endif
#if defined(PETSC_HAVE_CUDA)
    if (cudaMemcpy(
            x_ptr,
            initial_state.data(),
            solver.layout.density_dim * sizeof(Complex),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayWrite(ctx.x, &x_ptr));
        throw std::runtime_error(
            "evolve_single_state_cuda_ts: failed to upload initial state to device");
    }
#else
    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        x_ptr[i] = initial_state[i];
    }
#endif
#if defined(PETSC_HAVE_CUDA)
    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayWrite(ctx.x, &x_ptr));
#else
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));
#endif

    const double dt_initial = (tfinal - t0) / 1000.0;

    PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, t0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt_initial));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxTime(ctx.ts, tfinal));
    PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));

    const PetscScalar* x_read_ptr = nullptr;
#if defined(PETSC_HAVE_CUDA)
    PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayRead(ctx.x, &x_read_ptr));
#else
    PetscCallAbort(PETSC_COMM_SELF, VecGetArrayRead(ctx.x, &x_read_ptr));
#endif
    std::vector<Complex> final_state(solver.layout.density_dim, Complex{0.0, 0.0});
#if defined(PETSC_HAVE_CUDA)
    if (cudaMemcpy(
            final_state.data(),
            x_read_ptr,
            solver.layout.density_dim * sizeof(Complex),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(ctx.x, &x_read_ptr));
        throw std::runtime_error(
            "evolve_single_state_cuda_ts: failed to download final state from device");
    }
#else
    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        final_state[i] = x_read_ptr[i];
    }
#endif
#if defined(PETSC_HAVE_CUDA)
    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(ctx.x, &x_read_ptr));
#else
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(ctx.x, &x_read_ptr));
#endif

    return final_state;
}

} // namespace

std::vector<Index> make_first_n_state_indices(Index n)
{
    std::vector<Index> indices(n, 0);
    for (Index i = 0; i < n; ++i) {
        indices[i] = i;
    }
    return indices;
}

std::vector<Index> resolve_state_selection(
    const Solver& solver,
    const BenchmarkStateSelection& selection)
{
    if (selection.evolve_all_states) {
        std::vector<Index> all_indices(solver.layout.density_dim, 0);
        for (Index i = 0; i < solver.layout.density_dim; ++i) {
            all_indices[i] = i;
        }
        return all_indices;
    }

    if (selection.selected_state_indices.empty()) {
        throw std::invalid_argument(
            "resolve_state_selection: specific-state evolution requested with empty selection");
    }

    for (Index idx : selection.selected_state_indices) {
        if (idx >= solver.layout.density_dim) {
            throw std::invalid_argument(
                "resolve_state_selection: selected state index out of range");
        }
    }

    return selection.selected_state_indices;
}

Model build_transmon_chain_model(
    const TransmonChainBenchmarkConfig& config)
{
    if (config.num_transmons == 0) {
        throw std::invalid_argument(
            "build_transmon_chain_model: num_transmons must be > 0");
    }

    if (config.cutoff_dim < 2) {
        throw std::invalid_argument(
            "build_transmon_chain_model: cutoff_dim must be >= 2");
    }

    if (config.omega.size() != config.num_transmons ||
        config.alpha.size() != config.num_transmons ||
        config.t1.size() != config.num_transmons ||
        config.tphi.size() != config.num_transmons ||
        config.drive_frequency.size() != config.num_transmons) {
        throw std::invalid_argument(
            "build_transmon_chain_model: per-site parameter size mismatch");
    }

    if (config.num_transmons > 1 && config.g.size() != config.num_transmons - 1) {
        throw std::invalid_argument(
            "build_transmon_chain_model: coupling list size mismatch");
    }

    std::vector<Index> local_dims(
        config.num_transmons,
        config.cutoff_dim);

    const std::vector<Complex> a_local =
        make_local_annihilation_operator(config.cutoff_dim);
    const std::vector<Complex> adag_local =
        make_local_creation_operator(config.cutoff_dim);
    const std::vector<Complex> n_local =
        make_local_number_operator(config.cutoff_dim);
    const std::vector<Complex> n2_shifted_local =
        make_local_number_squared_shifted_operator(config.cutoff_dim);
    const std::vector<Complex> hopping_local =
        make_two_site_hopping_operator(config.cutoff_dim);

    const std::vector<Complex> drive_cos_local =
        add_operators(a_local, adag_local);

    const std::vector<Complex> drive_sin_local =
        scale_operator(
            subtract_operators(a_local, adag_local),
            Complex{0.0, 1.0});

    std::vector<OperatorTerm> hamiltonian_terms;
    std::vector<OperatorTerm> dissipator_terms;
    std::vector<TimeDependentTerm> time_dependent_terms;

    for (Index i = 0; i < config.num_transmons; ++i) {
        std::vector<Complex> omega_term = n_local;
        for (Complex& x : omega_term) {
            x *= config.omega[i];
        }

        hamiltonian_terms.push_back(OperatorTerm{
            TermKind::Hamiltonian,
            "omega_n_" + std::to_string(i),
            {i},
            omega_term,
            config.cutoff_dim,
            config.cutoff_dim
        });

        std::vector<Complex> alpha_term = n2_shifted_local;
        for (Complex& x : alpha_term) {
            x *= config.alpha[i];
        }

        hamiltonian_terms.push_back(OperatorTerm{
            TermKind::Hamiltonian,
            "alpha_n2_" + std::to_string(i),
            {i},
            alpha_term,
            config.cutoff_dim,
            config.cutoff_dim
        });

        std::vector<Complex> relax_op = a_local;
        for (Complex& x : relax_op) {
            x *= std::sqrt(1.0 / config.t1[i]);
        }

        dissipator_terms.push_back(OperatorTerm{
            TermKind::Dissipator,
            "relax_" + std::to_string(i),
            {i},
            relax_op,
            config.cutoff_dim,
            config.cutoff_dim
        });

        std::vector<Complex> dephase_op = n_local;
        for (Complex& x : dephase_op) {
            x *= std::sqrt(1.0 / config.tphi[i]);
        }

        dissipator_terms.push_back(OperatorTerm{
            TermKind::Dissipator,
            "dephase_" + std::to_string(i),
            {i},
            dephase_op,
            config.cutoff_dim,
            config.cutoff_dim
        });
    }

    for (Index i = 0; i + 1 < config.num_transmons; ++i) {
        std::vector<Complex> coupling_term = hopping_local;
        for (Complex& x : coupling_term) {
            x *= config.g[i];
        }

        hamiltonian_terms.push_back(OperatorTerm{
            TermKind::Hamiltonian,
            "coupling_" + std::to_string(i) + "_" + std::to_string(i + 1),
            {i, i + 1},
            coupling_term,
            config.cutoff_dim * config.cutoff_dim,
            config.cutoff_dim * config.cutoff_dim
        });
    }

    for (Index site : config.driven_sites) {
        if (site >= config.num_transmons) {
            throw std::invalid_argument(
                "build_transmon_chain_model: driven site out of range");
        }

        time_dependent_terms.push_back(TimeDependentTerm{
            "drive_cos_" + std::to_string(site),
            {site},
            drive_cos_local,
            config.cutoff_dim,
            config.cutoff_dim,
            make_gaussian_cosine_drive(
                config.drive_amplitude,
                config.drive_sigma,
                config.drive_center,
                config.drive_frequency[site])
        });

        time_dependent_terms.push_back(TimeDependentTerm{
            "drive_sin_" + std::to_string(site),
            {site},
            drive_sin_local,
            config.cutoff_dim,
            config.cutoff_dim,
            make_gaussian_sine_drive(
                config.drive_amplitude,
                config.drive_sigma,
                config.drive_center,
                config.drive_frequency[site])
        });
    }

    return Model{
        local_dims,
        hamiltonian_terms,
        dissipator_terms,
        time_dependent_terms
    };
}

TransmonChainBenchmarkTiming run_transmon_chain_cpu_benchmark(
    const TransmonChainBenchmarkConfig& config)
{
    const Model model = build_transmon_chain_model(config);
    const Solver solver = make_solver(model);

    const std::vector<Index> selected_indices =
        resolve_state_selection(solver, config.state_selection);

    const std::vector<std::vector<Complex>> initial_states =
        make_selected_basis_density_states(solver, selected_indices);

    CpuBenchmarkExecutionContext ctx{};
    PetscCallAbort(PETSC_COMM_SELF, create_cpu_benchmark_execution_context(solver, ctx));

    const auto t_start = std::chrono::steady_clock::now();

    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = selected_indices.size();
    timing.final_states.resize(selected_indices.size());

    for (Index i = 0; i < initial_states.size(); ++i) {
        timing.final_states[i] =
            evolve_single_state_cpu_ts(
                solver,
                initial_states[i],
                config.t0,
                config.tfinal,
                ctx);
    }

    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    PetscCallAbort(PETSC_COMM_SELF, destroy_cpu_benchmark_execution_context(ctx));
    return timing;
}

TransmonChainBenchmarkTiming run_transmon_chain_cuda_benchmark(
    const TransmonChainBenchmarkConfig& config)
{
    const Model model = build_transmon_chain_model(config);
    const Solver solver = make_solver(model);

    const std::vector<Index> selected_indices =
        resolve_state_selection(solver, config.state_selection);

    const std::vector<std::vector<Complex>> initial_states =
        make_selected_basis_density_states(solver, selected_indices);

    CudaBatchExecutionContext ctx{};
    PetscCallAbort(PETSC_COMM_SELF, create_cuda_batch_execution_context(solver, ctx));

    PetscCallAbort(PETSC_COMM_SELF, TSSetType(ctx.ts, TSRK));
    PetscCallAbort(PETSC_COMM_SELF, TSSetRHSFunction(
        ctx.ts,
        nullptr,
        ts_rhs_function_cuda_full_model_liouvillian,
        &ctx.rhs_ctx));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, 1000000));
    PetscCallAbort(PETSC_COMM_SELF, TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCallAbort(PETSC_COMM_SELF, TSSetFromOptions(ctx.ts));

    const auto t_start = std::chrono::steady_clock::now();

    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = selected_indices.size();
    timing.final_states.resize(selected_indices.size());

    for (Index i = 0; i < initial_states.size(); ++i) {
        timing.final_states[i] =
            evolve_single_state_cuda_ts(
                solver,
                initial_states[i],
                config.t0,
                config.tfinal,
                ctx);
    }

    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx));
    return timing;
}

} // namespace culindblad
