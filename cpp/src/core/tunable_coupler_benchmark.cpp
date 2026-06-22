#include "culindblad/tunable_coupler_benchmark.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <petscts.h>
#include <petscvec.h>

#include "culindblad/backend.hpp"
#include "culindblad/batch_evolution.hpp"
#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/state_buffer.hpp"

namespace culindblad {

namespace {

// ---------------------------------------------------------------------------
// Local single-site operators (Fock basis, cutoff d).
// ---------------------------------------------------------------------------

std::vector<Complex> make_local_annihilation_operator(Index d)
{
    std::vector<Complex> a(d * d, Complex{0.0, 0.0});
    for (Index n = 1; n < d; ++n) {
        a[(n - 1) * d + n] = Complex{std::sqrt(static_cast<double>(n)), 0.0};
    }
    return a;
}

std::vector<Complex> make_local_number_operator(Index d)
{
    std::vector<Complex> n_op(d * d, Complex{0.0, 0.0});
    for (Index n = 0; n < d; ++n) {
        n_op[n * d + n] = Complex{static_cast<double>(n), 0.0};
    }
    return n_op;
}

// 0.5 * n (n - 1) = 0.5 * b^dag b^dag b b  (the anharmonicity operator, /alpha).
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

// Two-site FULL capacitive coupling operator on the d*d local product space,
// flatten(n0, n1) = n0*d + n1 (site[0] = major). Builds (without the scalar g):
//   a0^dag a1 + a0 a1^dag - a0^dag a1^dag - a0 a1
// This is Hermitian and symmetric under site swap, so coupling on {i, j} is
// orientation-independent.
std::vector<Complex> make_two_site_full_coupling_operator(Index d)
{
    const Index dd = d * d;
    std::vector<Complex> op(dd * dd, Complex{0.0, 0.0});

    auto flatten = [d](Index left, Index right) {
        return left * d + right;
    };

    for (Index n0 = 0; n0 < d; ++n0) {
        for (Index n1 = 0; n1 < d; ++n1) {
            const Index col = flatten(n0, n1);

            // + a0^dag a1 : n0 -> n0+1, n1 -> n1-1
            if (n0 + 1 < d && n1 > 0) {
                const Index row = flatten(n0 + 1, n1 - 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0 + 1)) *
                    std::sqrt(static_cast<double>(n1));
                op[row * dd + col] += Complex{coeff, 0.0};
            }

            // + a0 a1^dag : n0 -> n0-1, n1 -> n1+1
            if (n0 > 0 && n1 + 1 < d) {
                const Index row = flatten(n0 - 1, n1 + 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0)) *
                    std::sqrt(static_cast<double>(n1 + 1));
                op[row * dd + col] += Complex{coeff, 0.0};
            }

            // - a0^dag a1^dag : n0 -> n0+1, n1 -> n1+1
            if (n0 + 1 < d && n1 + 1 < d) {
                const Index row = flatten(n0 + 1, n1 + 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0 + 1)) *
                    std::sqrt(static_cast<double>(n1 + 1));
                op[row * dd + col] -= Complex{coeff, 0.0};
            }

            // - a0 a1 : n0 -> n0-1, n1 -> n1-1
            if (n0 > 0 && n1 > 0) {
                const Index row = flatten(n0 - 1, n1 - 1);
                const double coeff =
                    std::sqrt(static_cast<double>(n0)) *
                    std::sqrt(static_cast<double>(n1));
                op[row * dd + col] -= Complex{coeff, 0.0};
            }
        }
    }

    return op;
}

// ---------------------------------------------------------------------------
// CPU execution (fixed-step TSRK4, mirrors the validated chain benchmark path).
// ---------------------------------------------------------------------------

struct CpuTsRhsContext {
    const Solver* solver = nullptr;
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
        solver.layout.density_dim};
    StateBuffer out_buf{
        reinterpret_cast<Complex*>(f_ptr),
        solver.layout.density_dim};

    apply_liouvillian_at_time(solver, static_cast<double>(t), in_buf, out_buf);

    PetscCall(VecRestoreArray(f, &f_ptr));
    PetscCall(VecRestoreArrayRead(x, &x_ptr));
    return 0;
}

struct CpuBenchmarkExecutionContext {
    TS ts = nullptr;
    Vec x = nullptr;
    CpuTsRhsContext rhs_ctx{};
    bool initialized = false;
};

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
    PetscCall(TSRKSetType(ctx.ts, TSRK4));
    TSAdapt cpu_adapt = nullptr;
    PetscCall(TSGetAdapt(ctx.ts, &cpu_adapt));
    PetscCall(TSAdaptSetType(cpu_adapt, TSADAPTNONE));

    PetscCall(VecCreate(PETSC_COMM_SELF, &ctx.x));
    PetscCall(VecSetSizes(ctx.x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetFromOptions(ctx.x));
    PetscCall(VecSet(ctx.x, 0.0));

    PetscCall(TSSetRHSFunction(
        ctx.ts, nullptr, ts_rhs_function_cpu_full_model_liouvillian, &ctx.rhs_ctx));

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
    Index num_steps,
    CpuBenchmarkExecutionContext& ctx)
{
    if (!ctx.initialized) {
        throw std::runtime_error(
            "evolve_single_state_cpu_ts: CPU benchmark context not initialized");
    }
    if (num_steps == 0) {
        throw std::invalid_argument(
            "evolve_single_state_cpu_ts: num_steps must be > 0");
    }

    PetscCallAbort(PETSC_COMM_SELF, VecSet(ctx.x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index i = 0; i < solver.layout.density_dim; ++i) {
        x_ptr[i] = initial_state[i];
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    const double dt_initial = (tfinal - t0) / static_cast<double>(num_steps);

    PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, t0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt_initial));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, static_cast<PetscInt>(num_steps)));
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

// ---------------------------------------------------------------------------
// CUDA batched execution (fixed-step TSRK4, mirrors the validated chain path).
// ---------------------------------------------------------------------------

std::vector<std::vector<Complex>> evolve_state_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    double t0,
    double tfinal,
    Index num_steps,
    CudaBatchExecutionContext& ctx)
{
    if (!ctx.initialized || ctx.ts == nullptr || ctx.x == nullptr) {
        throw std::runtime_error(
            "evolve_state_batch_cuda_ts: CUDA benchmark context not initialized");
    }
    if (num_steps == 0) {
        throw std::invalid_argument(
            "evolve_state_batch_cuda_ts: num_steps must be > 0");
    }
    if (initial_states.size() != ctx.rhs_ctx.batch_size) {
        throw std::invalid_argument(
            "evolve_state_batch_cuda_ts: initial state count must match context batch size");
    }

    PetscCallAbort(PETSC_COMM_SELF, VecSet(ctx.x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index batch = 0; batch < initial_states.size(); ++batch) {
        if (initial_states[batch].size() != solver.layout.density_dim) {
            PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));
            throw std::invalid_argument(
                "evolve_state_batch_cuda_ts: initial state size mismatch");
        }
        for (Index i = 0; i < solver.layout.density_dim; ++i) {
            x_ptr[batch * solver.layout.density_dim + i] = initial_states[batch][i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    const double dt_initial = (tfinal - t0) / static_cast<double>(num_steps);

    PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, t0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetStepNumber(ctx.ts, 0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, dt_initial));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, static_cast<PetscInt>(num_steps)));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxTime(ctx.ts, tfinal));
    PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));

    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    std::vector<std::vector<Complex>> final_states(
        initial_states.size(),
        std::vector<Complex>(solver.layout.density_dim, Complex{0.0, 0.0}));
    for (Index batch = 0; batch < initial_states.size(); ++batch) {
        for (Index i = 0; i < solver.layout.density_dim; ++i) {
            final_states[batch][i] = x_ptr[batch * solver.layout.density_dim + i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    return final_states;
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
        std::vector<Complex> rho(solver.layout.density_dim, Complex{0.0, 0.0});
        rho[state_index] = Complex{1.0, 0.0};
        states.push_back(std::move(rho));
    }
    return states;
}

} // namespace

Index tunable_coupler_num_elements(const TunableCouplerBenchmarkConfig& config)
{
    return config.num_qubits + config.num_couplers;
}

Model build_tunable_coupler_model(const TunableCouplerBenchmarkConfig& config)
{
    const Index n_elem = tunable_coupler_num_elements(config);

    if (n_elem == 0) {
        throw std::invalid_argument(
            "build_tunable_coupler_model: need at least one element");
    }
    if (config.cutoff_dim < 2) {
        throw std::invalid_argument(
            "build_tunable_coupler_model: cutoff_dim must be >= 2");
    }
    if (config.omega.size() != n_elem ||
        config.alpha.size() != n_elem ||
        config.t1.size() != n_elem ||
        config.tphi.size() != n_elem) {
        throw std::invalid_argument(
            "build_tunable_coupler_model: per-element parameter size mismatch");
    }

    const Index d = config.cutoff_dim;
    std::vector<Index> local_dims(n_elem, d);

    const std::vector<Complex> a_local = make_local_annihilation_operator(d);
    const std::vector<Complex> n_local = make_local_number_operator(d);
    const std::vector<Complex> n2_shifted_local =
        make_local_number_squared_shifted_operator(d);
    const std::vector<Complex> coupling_local =
        make_two_site_full_coupling_operator(d);

    std::vector<OperatorTerm> hamiltonian_terms;
    std::vector<OperatorTerm> dissipator_terms;
    std::vector<TimeDependentTerm> time_dependent_terms;  // static model: empty

    // Per-element bare Hamiltonian + dissipators.
    for (Index i = 0; i < n_elem; ++i) {
        std::vector<Complex> omega_term = n_local;
        for (Complex& x : omega_term) {
            x *= config.omega[i];
        }
        hamiltonian_terms.push_back(OperatorTerm{
            TermKind::Hamiltonian, "omega_n_" + std::to_string(i),
            {i}, omega_term, d, d});

        std::vector<Complex> alpha_term = n2_shifted_local;
        for (Complex& x : alpha_term) {
            x *= config.alpha[i];
        }
        hamiltonian_terms.push_back(OperatorTerm{
            TermKind::Hamiltonian, "alpha_n2_" + std::to_string(i),
            {i}, alpha_term, d, d});

        std::vector<Complex> relax_op = a_local;
        for (Complex& x : relax_op) {
            x *= std::sqrt(1.0 / config.t1[i]);
        }
        dissipator_terms.push_back(OperatorTerm{
            TermKind::Dissipator, "relax_" + std::to_string(i),
            {i}, relax_op, d, d});

        std::vector<Complex> dephase_op = n_local;
        for (Complex& x : dephase_op) {
            x *= std::sqrt(1.0 / config.tphi[i]);
        }
        dissipator_terms.push_back(OperatorTerm{
            TermKind::Dissipator, "dephase_" + std::to_string(i),
            {i}, dephase_op, d, d});
    }

    // ALL-TO-ALL pairwise capacitive coupling (uniform g for the benchmark).
    const Index dd = d * d;
    for (Index i = 0; i < n_elem; ++i) {
        for (Index j = i + 1; j < n_elem; ++j) {
            std::vector<Complex> coupling_term = coupling_local;
            for (Complex& x : coupling_term) {
                x *= config.g;
            }
            hamiltonian_terms.push_back(OperatorTerm{
                TermKind::Hamiltonian,
                "coupling_" + std::to_string(i) + "_" + std::to_string(j),
                {i, j}, coupling_term, dd, dd});
        }
    }

    return Model{
        local_dims, hamiltonian_terms, dissipator_terms, time_dependent_terms};
}

TransmonChainBenchmarkTiming run_tunable_coupler_cpu_benchmark(
    const TunableCouplerBenchmarkConfig& config)
{
    const Model model = build_tunable_coupler_model(config);
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
        timing.final_states[i] = evolve_single_state_cpu_ts(
            solver, initial_states[i], config.t0, config.tfinal,
            config.batched_num_steps, ctx);
    }

    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

    PetscCallAbort(PETSC_COMM_SELF, destroy_cpu_benchmark_execution_context(ctx));
    return timing;
}

TransmonChainBenchmarkTiming run_tunable_coupler_cuda_benchmark(
    const TunableCouplerBenchmarkConfig& config)
{
    const Model model = build_tunable_coupler_model(config);
    const Solver solver = make_solver(model);

    const std::vector<Index> selected_indices =
        resolve_state_selection(solver, config.state_selection);
    const std::vector<std::vector<Complex>> initial_states =
        make_selected_basis_density_states(solver, selected_indices);

    CudaBatchExecutionContext ctx{};
    PetscCallAbort(
        PETSC_COMM_SELF,
        create_cuda_batch_execution_context(solver, selected_indices.size(), ctx));

    PetscCallAbort(PETSC_COMM_SELF, TSSetType(ctx.ts, TSRK));
    PetscCallAbort(PETSC_COMM_SELF, TSRKSetType(ctx.ts, TSRK4));
    TSAdapt gpu_adapt = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, TSGetAdapt(ctx.ts, &gpu_adapt));
    PetscCallAbort(PETSC_COMM_SELF, TSAdaptSetType(gpu_adapt, TSADAPTNONE));
    PetscCallAbort(PETSC_COMM_SELF, TSSetRHSFunction(
        ctx.ts, nullptr, ts_rhs_function_cuda_full_model_liouvillian, &ctx.rhs_ctx));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, 1000000));
    PetscCallAbort(PETSC_COMM_SELF, TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCallAbort(PETSC_COMM_SELF, TSSetFromOptions(ctx.ts));

    const auto t_start = std::chrono::steady_clock::now();

    TransmonChainBenchmarkTiming timing;
    timing.num_evolved_states = selected_indices.size();
    timing.final_states = evolve_state_batch_cuda_ts(
        solver, initial_states, config.t0, config.tfinal,
        config.batched_num_steps, ctx);

    const auto t_end = std::chrono::steady_clock::now();
    timing.wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx));
    return timing;
}

} // namespace culindblad
