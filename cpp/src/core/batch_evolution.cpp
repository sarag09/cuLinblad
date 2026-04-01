#include "culindblad/batch_evolution.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

#include <petscts.h>
#include <petscvec.h>

#include "culindblad/backend.hpp"
#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/local_operator_utils.hpp"
#include "culindblad/time_dependent_term.hpp"

namespace culindblad {

namespace {

bool same_sites(
    const std::vector<Index>& a,
    const std::vector<Index>& b)
{
    if (a.size() != b.size()) {
        return false;
    }

    for (Index i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

std::vector<Index> choose_seed_target_sites(
    const Solver& solver)
{
    if (!solver.model.hamiltonian_terms.empty() &&
        !solver.model.hamiltonian_terms.front().sites.empty()) {
        return solver.model.hamiltonian_terms.front().sites;
    }

    if (!solver.model.dissipator_terms.empty() &&
        !solver.model.dissipator_terms.front().sites.empty()) {
        return solver.model.dissipator_terms.front().sites;
    }

    if (!solver.model.time_dependent_hamiltonian_terms.empty() &&
        !solver.model.time_dependent_hamiltonian_terms.front().sites.empty()) {
        return solver.model.time_dependent_hamiltonian_terms.front().sites;
    }

    if (solver.model.local_dims.empty()) {
        throw std::runtime_error(
            "choose_seed_target_sites: solver model has no subsystems");
    }

    return {0};
}

std::vector<CachedDissipatorAuxiliaries> build_cached_static_dissipators(
    const Solver& solver)
{
    std::vector<CachedDissipatorAuxiliaries> cached;

    for (const OperatorTerm& d_term : solver.model.dissipator_terms) {
        CachedDissipatorAuxiliaries aux;
        aux.name = d_term.name;
        aux.sites = d_term.sites;
        aux.l_op = d_term.matrix;
        aux.l_dag = local_conjugate_transpose(d_term.matrix, d_term.row_dim);
        aux.l_dag_l = local_multiply_square(aux.l_dag, d_term.matrix, d_term.row_dim);
        cached.push_back(std::move(aux));
    }

    return cached;
}

std::vector<CachedGroupedLayoutEntry> build_cached_grouped_layouts(
    const Solver& solver,
    Index batch_size)
{
    std::vector<CachedGroupedLayoutEntry> cached;

    auto add_if_missing = [&](const std::vector<Index>& sites) {
        for (const CachedGroupedLayoutEntry& entry : cached) {
            if (same_sites(entry.sites, sites)) {
                return;
            }
        }

        CachedGroupedLayoutEntry entry;
        entry.sites = sites;
        entry.grouped_layout =
            make_grouped_state_layout(sites, solver.model.local_dims);
        entry.d_grouped_input = nullptr;
        entry.d_grouped_term = nullptr;
        entry.d_grouped_accum = nullptr;
        entry.grouped_bytes =
            static_cast<std::size_t>(batch_size) *
            static_cast<std::size_t>(entry.grouped_layout.grouped_size) *
            sizeof(Complex);

        if (!create_cuda_grouped_state_layout(
                entry.grouped_layout,
                entry.cuda_grouped_layout)) {
            throw std::runtime_error(
                "build_cached_grouped_layouts: failed to create CUDA grouped layout");
        }

        if (cudaMalloc(&entry.d_grouped_input, entry.grouped_bytes) != cudaSuccess ||
            cudaMalloc(&entry.d_grouped_term, entry.grouped_bytes) != cudaSuccess ||
            cudaMalloc(&entry.d_grouped_accum, entry.grouped_bytes) != cudaSuccess) {
            if (entry.d_grouped_accum != nullptr) {
                cudaFree(entry.d_grouped_accum);
                entry.d_grouped_accum = nullptr;
            }
            if (entry.d_grouped_term != nullptr) {
                cudaFree(entry.d_grouped_term);
                entry.d_grouped_term = nullptr;
            }
            if (entry.d_grouped_input != nullptr) {
                cudaFree(entry.d_grouped_input);
                entry.d_grouped_input = nullptr;
            }
            (void)destroy_cuda_grouped_state_layout(entry.cuda_grouped_layout);
            throw std::runtime_error(
                "build_cached_grouped_layouts: failed to allocate grouped RHS buffers");
        }

        cached.push_back(std::move(entry));
    };

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        add_if_missing(h_term.sites);
    }

    for (const OperatorTerm& d_term : solver.model.dissipator_terms) {
        add_if_missing(d_term.sites);
    }

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        add_if_missing(td_term.sites);
    }

    return cached;
}

void destroy_cached_grouped_layouts(
    std::vector<CachedGroupedLayoutEntry>& cached)
{
    for (CachedGroupedLayoutEntry& entry : cached) {
        if (entry.d_grouped_accum != nullptr) {
            cudaFree(entry.d_grouped_accum);
            entry.d_grouped_accum = nullptr;
        }
        if (entry.d_grouped_term != nullptr) {
            cudaFree(entry.d_grouped_term);
            entry.d_grouped_term = nullptr;
        }
        if (entry.d_grouped_input != nullptr) {
            cudaFree(entry.d_grouped_input);
            entry.d_grouped_input = nullptr;
        }
        entry.grouped_bytes = 0;
        (void)destroy_cuda_grouped_state_layout(entry.cuda_grouped_layout);
    }
    cached.clear();
}

PetscErrorCode create_rhs_work_vectors(
    Vec prototype,
    PetscCudaTsRhsContext& rhs_ctx)
{
    rhs_ctx.work_vec_a = nullptr;
    rhs_ctx.work_vec_b = nullptr;
    rhs_ctx.work_vec_c = nullptr;

    PetscCall(VecDuplicate(prototype, &rhs_ctx.work_vec_a));
    PetscCall(VecDuplicate(prototype, &rhs_ctx.work_vec_b));
    PetscCall(VecDuplicate(prototype, &rhs_ctx.work_vec_c));
    return 0;
}

PetscErrorCode destroy_rhs_work_vectors(
    PetscCudaTsRhsContext& rhs_ctx)
{
    if (rhs_ctx.work_vec_c != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
    }
    if (rhs_ctx.work_vec_b != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
    }
    if (rhs_ctx.work_vec_a != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
    }
    return 0;
}

} // namespace

PetscErrorCode create_cuda_batch_execution_context(
    const Solver& solver,
    Index batch_size,
    CudaBatchExecutionContext& ctx)
{
    if (batch_size == 0) {
        return PETSC_ERR_ARG_OUTOFRANGE;
    }

    ctx.ts = nullptr;
    ctx.x = nullptr;
    ctx.initialized = false;

    const std::vector<Index> seed_target_sites =
        choose_seed_target_sites(solver);

    PetscCall(TSCreate(PETSC_COMM_SELF, &ctx.ts));
    PetscCall(TSSetType(ctx.ts, TSEULER));

    PetscCall(VecCreate(PETSC_COMM_SELF, &ctx.x));
    PetscCall(
        VecSetSizes(
            ctx.x,
            PETSC_DECIDE,
            batch_size * solver.layout.density_dim));
    PetscCall(VecSetType(ctx.x, VECCUDA));
    PetscCall(VecSet(ctx.x, 0.0));

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout(seed_target_sites, solver.model.local_dims);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(grouped_layout, cuda_grouped_layout)) {
        PetscCall(VecDestroy(&ctx.x));
        PetscCall(TSDestroy(&ctx.ts));
        return PETSC_ERR_LIB;
    }

    ctx.rhs_ctx = PetscCudaTsRhsContext{
        &solver,
        batch_size,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        seed_target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        build_cached_static_dissipators(solver),
        build_cached_grouped_layouts(solver, batch_size),
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };

    if (cudaStreamCreate(&ctx.rhs_ctx.elementwise_stream) != cudaSuccess) {
        destroy_cached_grouped_layouts(ctx.rhs_ctx.cached_grouped_layouts);
        (void)destroy_cuda_grouped_state_layout(ctx.rhs_ctx.cuda_grouped_layout);
        PetscCall(VecDestroy(&ctx.x));
        PetscCall(TSDestroy(&ctx.ts));
        return PETSC_ERR_LIB;
    }

    PetscCall(create_rhs_work_vectors(ctx.x, ctx.rhs_ctx));

    PetscCall(TSSetRHSFunction(
        ctx.ts,
        nullptr,
        ts_rhs_function_cuda_full_model_liouvillian,
        &ctx.rhs_ctx));

    ctx.initialized = true;
    return 0;
}

PetscErrorCode destroy_cuda_batch_execution_context(
    CudaBatchExecutionContext& ctx)
{
    if (!ctx.initialized) {
        return 0;
    }

    PetscCall(destroy_rhs_work_vectors(ctx.rhs_ctx));
    (void)destroy_cutensor_executor_cache(ctx.rhs_ctx.executor_cache);
    destroy_cached_grouped_layouts(ctx.rhs_ctx.cached_grouped_layouts);
    (void)destroy_cuda_grouped_state_layout(ctx.rhs_ctx.cuda_grouped_layout);

    if (ctx.rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(ctx.rhs_ctx.elementwise_stream);
        ctx.rhs_ctx.elementwise_stream = nullptr;
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

BatchEvolutionResult evolve_density_batch_cpu_reference(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    if (config.batch_size == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cpu_reference: batch_size must be > 0");
    }

    if (config.num_steps == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cpu_reference: num_steps must be > 0");
    }

    BatchEvolutionResult result;
    result.final_states.resize(initial_states.size());

    const Index density_dim = solver.layout.density_dim;

    for (Index batch_begin = 0; batch_begin < initial_states.size(); batch_begin += config.batch_size) {
        const Index batch_end =
            std::min<Index>(batch_begin + config.batch_size, initial_states.size());

        for (Index state_idx = batch_begin; state_idx < batch_end; ++state_idx) {
            if (initial_states[state_idx].size() != density_dim) {
                throw std::invalid_argument(
                    "evolve_density_batch_cpu_reference: initial state size mismatch");
            }

            std::vector<Complex> state = initial_states[state_idx];
            std::vector<Complex> rhs(density_dim, Complex{0.0, 0.0});

            for (Index step = 0; step < config.num_steps; ++step) {
                const double t =
                    config.t0 + static_cast<double>(step) * config.dt;

                ConstStateBuffer in_buf{state.data(), state.size()};
                StateBuffer rhs_buf{rhs.data(), rhs.size()};

                apply_liouvillian_at_time(solver, t, in_buf, rhs_buf);

                for (Index i = 0; i < density_dim; ++i) {
                    state[i] += config.dt * rhs[i];
                }
            }

            result.final_states[state_idx] = std::move(state);
        }
    }

    return result;
}

BatchEvolutionResult evolve_density_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    if (config.batch_size == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cuda_ts: batch_size must be > 0");
    }

    if (config.num_steps == 0) {
        throw std::invalid_argument(
            "evolve_density_batch_cuda_ts: num_steps must be > 0");
    }

    BatchEvolutionResult result;
    result.final_states.resize(initial_states.size());

    const Index density_dim = solver.layout.density_dim;

    if (initial_states.size() != config.batch_size) {
        throw std::invalid_argument(
            "evolve_density_batch_cuda_ts: initial_states size must equal config.batch_size for true batching");
    }

    for (const std::vector<Complex>& state : initial_states) {
        if (state.size() != density_dim) {
            throw std::invalid_argument(
                "evolve_density_batch_cuda_ts: initial state size mismatch");
        }
    }

    CudaBatchExecutionContext ctx{};
    PetscCallAbort(
        PETSC_COMM_SELF,
        create_cuda_batch_execution_context(
            solver,
            config.batch_size,
            ctx));

    PetscCallAbort(PETSC_COMM_SELF, VecSet(ctx.x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index batch = 0; batch < config.batch_size; ++batch) {
        for (Index i = 0; i < density_dim; ++i) {
            x_ptr[batch * density_dim + i] = initial_states[batch][i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    PetscCallAbort(PETSC_COMM_SELF, TSSetType(ctx.ts, TSEULER));
    PetscCallAbort(
        PETSC_COMM_SELF,
        TSSetRHSFunction(
            ctx.ts,
            nullptr,
            ts_rhs_function_cuda_full_model_liouvillian,
            &ctx.rhs_ctx));
    PetscCallAbort(PETSC_COMM_SELF, TSSetTime(ctx.ts, config.t0));
    PetscCallAbort(PETSC_COMM_SELF, TSSetTimeStep(ctx.ts, config.dt));
    PetscCallAbort(PETSC_COMM_SELF, TSSetMaxSteps(ctx.ts, static_cast<PetscInt>(config.num_steps)));
    PetscCallAbort(
        PETSC_COMM_SELF,
        TSSetMaxTime(
            ctx.ts,
            config.t0 + static_cast<double>(config.num_steps) * config.dt));
    PetscCallAbort(PETSC_COMM_SELF, TSSetExactFinalTime(ctx.ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCallAbort(PETSC_COMM_SELF, TSSetFromOptions(ctx.ts));
    PetscCallAbort(PETSC_COMM_SELF, TSSolve(ctx.ts, ctx.x));

    PetscCallAbort(PETSC_COMM_SELF, VecGetArray(ctx.x, &x_ptr));
    for (Index batch = 0; batch < config.batch_size; ++batch) {
        result.final_states[batch].resize(density_dim);
        for (Index i = 0; i < density_dim; ++i) {
            result.final_states[batch][i] = x_ptr[batch * density_dim + i];
        }
    }
    PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(ctx.x, &x_ptr));

    PetscCallAbort(PETSC_COMM_SELF, destroy_cuda_batch_execution_context(ctx));
    return result;
}

BatchEvolutionTiming time_density_batch_cpu_reference(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    const auto t_start = std::chrono::steady_clock::now();

    BatchEvolutionResult result =
        evolve_density_batch_cpu_reference(
            solver,
            initial_states,
            config);

    const auto t_end = std::chrono::steady_clock::now();

    BatchEvolutionTiming timing;
    timing.elapsed_seconds =
        std::chrono::duration<double>(t_end - t_start).count();
    timing.gpu_elapsed_seconds = 0.0;
    timing.result = std::move(result);
    return timing;
}

BatchEvolutionTiming time_density_batch_cuda_ts(
    const Solver& solver,
    const std::vector<std::vector<Complex>>& initial_states,
    const BatchEvolutionConfig& config)
{
    const auto t_start = std::chrono::steady_clock::now();

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;

    if (cudaEventCreate(&gpu_start) != cudaSuccess) {
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventCreate start failed");
    }

    if (cudaEventCreate(&gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventCreate stop failed");
    }

    if (cudaEventRecord(gpu_start, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventRecord start failed");
    }

    BatchEvolutionResult result =
        evolve_density_batch_cuda_ts(
            solver,
            initial_states,
            config);

    if (cudaEventRecord(gpu_stop, 0) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventRecord stop failed");
    }

    if (cudaEventSynchronize(gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventSynchronize failed");
    }

    float gpu_milliseconds = 0.0f;
    if (cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop) != cudaSuccess) {
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(gpu_start);
        throw std::runtime_error("time_density_batch_cuda_ts: cudaEventElapsedTime failed");
    }

    cudaEventDestroy(gpu_stop);
    cudaEventDestroy(gpu_start);

    const auto t_end = std::chrono::steady_clock::now();

    BatchEvolutionTiming timing;
    timing.elapsed_seconds =
        std::chrono::duration<double>(t_end - t_start).count();
    timing.gpu_elapsed_seconds =
        static_cast<double>(gpu_milliseconds) * 1.0e-3;
    timing.result = std::move(result);
    return timing;
}

} // namespace culindblad
