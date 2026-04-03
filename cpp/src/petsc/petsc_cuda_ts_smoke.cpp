#include "culindblad/petsc_cuda_ts_smoke.hpp"

#include <petscts.h>
#include <petscvec.h>

#include <algorithm>
#include <stdexcept>

#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/local_operator_utils.hpp"
#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/time_dependent_term.hpp"
#include "culindblad/types.hpp"

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

bool contains_sites(
    const std::vector<std::vector<Index>>& site_sets,
    const std::vector<Index>& sites)
{
    for (const std::vector<Index>& existing_sites : site_sets) {
        if (same_sites(existing_sites, sites)) {
            return true;
        }
    }

    return false;
}

std::size_t estimate_executor_cache_entries(
    const Solver& solver)
{
    constexpr std::size_t kMinimumCacheEntries = 6;
    constexpr std::size_t kCacheSlackEntries = 4;

    std::vector<std::vector<Index>> commutator_site_sets;
    std::vector<std::vector<Index>> dissipator_site_sets;

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        if (!contains_sites(commutator_site_sets, h_term.sites)) {
            commutator_site_sets.push_back(h_term.sites);
        }
    }

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        if (!contains_sites(commutator_site_sets, td_term.sites)) {
            commutator_site_sets.push_back(td_term.sites);
        }
    }

    for (const OperatorTerm& d_term : solver.model.dissipator_terms) {
        if (!contains_sites(dissipator_site_sets, d_term.sites)) {
            dissipator_site_sets.push_back(d_term.sites);
        }
    }

    const std::size_t required_entries =
        2 * commutator_site_sets.size() +
        4 * dissipator_site_sets.size();

    return std::max(
        kMinimumCacheEntries,
        required_entries + kCacheSlackEntries);
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
        entry.grouped_bytes =
            static_cast<std::size_t>(batch_size) *
            static_cast<std::size_t>(entry.grouped_layout.grouped_size) *
            sizeof(Complex);

        if (!create_cuda_grouped_state_layout(
                entry.grouped_layout,
                entry.cuda_grouped_layout)) {
            throw std::runtime_error("build_cached_grouped_layouts: failed to create CUDA grouped layout");
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

PetscErrorCode create_grouped_rhs_scratch_buffers(
    PetscCudaTsRhsContext& rhs_ctx)
{
    rhs_ctx.grouped_scratch = GroupedRhsScratchBuffers{};

    std::size_t max_grouped_bytes = 0;
    for (const CachedGroupedLayoutEntry& entry : rhs_ctx.cached_grouped_layouts) {
        if (entry.grouped_bytes > max_grouped_bytes) {
            max_grouped_bytes = entry.grouped_bytes;
        }
    }

    rhs_ctx.grouped_scratch.grouped_bytes = max_grouped_bytes;
    if (max_grouped_bytes == 0) {
        return 0;
    }

    if (cudaMalloc(&rhs_ctx.grouped_scratch.d_grouped_input, max_grouped_bytes) != cudaSuccess ||
        cudaMalloc(&rhs_ctx.grouped_scratch.d_grouped_term, max_grouped_bytes) != cudaSuccess ||
        cudaMalloc(&rhs_ctx.grouped_scratch.d_grouped_accum, max_grouped_bytes) != cudaSuccess) {
        if (rhs_ctx.grouped_scratch.d_grouped_accum != nullptr) {
            cudaFree(rhs_ctx.grouped_scratch.d_grouped_accum);
            rhs_ctx.grouped_scratch.d_grouped_accum = nullptr;
        }
        if (rhs_ctx.grouped_scratch.d_grouped_term != nullptr) {
            cudaFree(rhs_ctx.grouped_scratch.d_grouped_term);
            rhs_ctx.grouped_scratch.d_grouped_term = nullptr;
        }
        if (rhs_ctx.grouped_scratch.d_grouped_input != nullptr) {
            cudaFree(rhs_ctx.grouped_scratch.d_grouped_input);
            rhs_ctx.grouped_scratch.d_grouped_input = nullptr;
        }
        rhs_ctx.grouped_scratch.grouped_bytes = 0;
        return PETSC_ERR_MEM;
    }

    return 0;
}

void destroy_grouped_rhs_scratch_buffers(
    PetscCudaTsRhsContext& rhs_ctx)
{
    if (rhs_ctx.grouped_scratch.d_grouped_accum != nullptr) {
        cudaFree(rhs_ctx.grouped_scratch.d_grouped_accum);
        rhs_ctx.grouped_scratch.d_grouped_accum = nullptr;
    }
    if (rhs_ctx.grouped_scratch.d_grouped_term != nullptr) {
        cudaFree(rhs_ctx.grouped_scratch.d_grouped_term);
        rhs_ctx.grouped_scratch.d_grouped_term = nullptr;
    }
    if (rhs_ctx.grouped_scratch.d_grouped_input != nullptr) {
        cudaFree(rhs_ctx.grouped_scratch.d_grouped_input);
        rhs_ctx.grouped_scratch.d_grouped_input = nullptr;
    }
    rhs_ctx.grouped_scratch.grouped_bytes = 0;
}

} // namespace

PetscErrorCode run_ts_cuda_grouped_left_smoke_test(
    const Solver& solver,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    Index row,
    Index col,
    Complex& value_out)
{
    TS ts = nullptr;
    Vec x = nullptr;

    PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
    PetscCall(TSSetType(ts, TSEULER));

    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(x, VECCUDA));
    PetscCall(VecSet(x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCall(VecGetArray(x, &x_ptr));
    x_ptr[row * solver.layout.hilbert_dim + col] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x, &x_ptr));

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout(target_sites, solver.model.local_dims);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(grouped_layout, cuda_grouped_layout)) {
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    PetscCudaTsRhsContext rhs_ctx{
        &solver,
        1,
        &local_op,
        nullptr,
        nullptr,
        nullptr,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        {},
        {},
        GroupedRhsScratchBuffers{},
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    rhs_ctx.executor_cache.max_entries =
        estimate_executor_cache_entries(solver);

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    PetscCall(create_rhs_work_vectors(x, rhs_ctx));

    PetscCall(TSSetRHSFunction(ts, nullptr, ts_rhs_function_cuda_grouped_commutator, &rhs_ctx));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetTimeStep(ts, 1.0e-3));
    PetscCall(TSSetMaxSteps(ts, 1));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSolve(ts, x));

    PetscCall(VecGetArray(x, &x_ptr));
    value_out = reinterpret_cast<Complex*>(x_ptr)[row * solver.layout.hilbert_dim + col];
    PetscCall(VecRestoreArray(x, &x_ptr));

    PetscCall(destroy_rhs_work_vectors(rhs_ctx));
    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

PetscErrorCode run_ts_cuda_grouped_liouvillian_smoke_test(
    const Solver& solver,
    const std::vector<Complex>& h_local_op,
    const std::vector<Complex>& d_local_op,
    const std::vector<Complex>& d_local_op_dag,
    const std::vector<Complex>& d_local_op_dag_op,
    const std::vector<Index>& target_sites,
    Index row,
    Index col,
    Complex& value_out)
{
    TS ts = nullptr;
    Vec x = nullptr;

    PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
    PetscCall(TSSetType(ts, TSEULER));

    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(x, VECCUDA));
    PetscCall(VecSet(x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCall(VecGetArray(x, &x_ptr));
    x_ptr[row * solver.layout.hilbert_dim + col] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x, &x_ptr));

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout(target_sites, solver.model.local_dims);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(grouped_layout, cuda_grouped_layout)) {
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    PetscCudaTsRhsContext rhs_ctx{
        &solver,
        1,
        &h_local_op,
        &d_local_op,
        &d_local_op_dag,
        &d_local_op_dag_op,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        {},
        {},
        GroupedRhsScratchBuffers{},
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    rhs_ctx.executor_cache.max_entries =
        estimate_executor_cache_entries(solver);

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    PetscCall(create_rhs_work_vectors(x, rhs_ctx));

    PetscCall(TSSetRHSFunction(ts, nullptr, ts_rhs_function_cuda_grouped_liouvillian, &rhs_ctx));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetTimeStep(ts, 1.0e-3));
    PetscCall(TSSetMaxSteps(ts, 1));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSolve(ts, x));

    PetscCall(VecGetArray(x, &x_ptr));
    value_out = reinterpret_cast<Complex*>(x_ptr)[row * solver.layout.hilbert_dim + col];
    PetscCall(VecRestoreArray(x, &x_ptr));

    PetscCall(destroy_rhs_work_vectors(rhs_ctx));
    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

PetscErrorCode run_ts_cuda_static_model_liouvillian_smoke_test(
    const Solver& solver,
    Index row,
    Index col,
    Complex& value_out)
{
    TS ts = nullptr;
    Vec x = nullptr;

    PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
    PetscCall(TSSetType(ts, TSEULER));

    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(x, VECCUDA));
    PetscCall(VecSet(x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCall(VecGetArray(x, &x_ptr));
    x_ptr[row * solver.layout.hilbert_dim + col] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x, &x_ptr));

    const std::vector<Index> target_sites = {0, 1, 2};

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout(target_sites, solver.model.local_dims);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(grouped_layout, cuda_grouped_layout)) {
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    PetscCudaTsRhsContext rhs_ctx{
        &solver,
        1,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        build_cached_static_dissipators(solver),
        build_cached_grouped_layouts(solver, 1),
        GroupedRhsScratchBuffers{},
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    rhs_ctx.executor_cache.max_entries =
        estimate_executor_cache_entries(solver);

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    {
        const PetscErrorCode scratch_ierr =
            create_grouped_rhs_scratch_buffers(rhs_ctx);
        if (scratch_ierr != 0) {
            destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
            (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
            if (rhs_ctx.elementwise_stream != nullptr) {
                cudaStreamDestroy(rhs_ctx.elementwise_stream);
                rhs_ctx.elementwise_stream = nullptr;
            }
            PetscCall(VecDestroy(&x));
            PetscCall(TSDestroy(&ts));
            return scratch_ierr;
        }
    }

    PetscCall(create_rhs_work_vectors(x, rhs_ctx));

    PetscCall(TSSetRHSFunction(ts, nullptr, ts_rhs_function_cuda_static_model_liouvillian, &rhs_ctx));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetTimeStep(ts, 1.0e-3));
    PetscCall(TSSetMaxSteps(ts, 1));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSolve(ts, x));

    PetscCall(VecGetArray(x, &x_ptr));
    value_out = reinterpret_cast<Complex*>(x_ptr)[row * solver.layout.hilbert_dim + col];
    PetscCall(VecRestoreArray(x, &x_ptr));

    PetscCall(destroy_rhs_work_vectors(rhs_ctx));
    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    destroy_grouped_rhs_scratch_buffers(rhs_ctx);
    destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

PetscErrorCode run_ts_cuda_full_model_liouvillian_smoke_test(
    const Solver& solver,
    double start_time,
    Index row,
    Index col,
    Complex& value_out)
{
    TS ts = nullptr;
    Vec x = nullptr;

    PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
    PetscCall(TSSetType(ts, TSEULER));

    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, solver.layout.density_dim));
    PetscCall(VecSetType(x, VECCUDA));
    PetscCall(VecSet(x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCall(VecGetArray(x, &x_ptr));
    x_ptr[row * solver.layout.hilbert_dim + col] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x, &x_ptr));

    const std::vector<Index> target_sites = {0, 1, 2};

    GroupedStateLayout grouped_layout =
        make_grouped_state_layout(target_sites, solver.model.local_dims);

    CudaGroupedStateLayout cuda_grouped_layout{};
    if (!create_cuda_grouped_state_layout(grouped_layout, cuda_grouped_layout)) {
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    PetscCudaTsRhsContext rhs_ctx{
        &solver,
        1,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        build_cached_static_dissipators(solver),
        build_cached_grouped_layouts(solver, 1),
        GroupedRhsScratchBuffers{},
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    rhs_ctx.executor_cache.max_entries =
        estimate_executor_cache_entries(solver);

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

    {
        const PetscErrorCode scratch_ierr =
            create_grouped_rhs_scratch_buffers(rhs_ctx);
        if (scratch_ierr != 0) {
            destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
            (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
            if (rhs_ctx.elementwise_stream != nullptr) {
                cudaStreamDestroy(rhs_ctx.elementwise_stream);
                rhs_ctx.elementwise_stream = nullptr;
            }
            PetscCall(VecDestroy(&x));
            PetscCall(TSDestroy(&ts));
            return scratch_ierr;
        }
    }

    PetscCall(create_rhs_work_vectors(x, rhs_ctx));

    PetscCall(TSSetRHSFunction(ts, nullptr, ts_rhs_function_cuda_full_model_liouvillian, &rhs_ctx));
    PetscCall(TSSetTime(ts, start_time));
    PetscCall(TSSetTimeStep(ts, 1.0e-3));
    PetscCall(TSSetMaxSteps(ts, 1));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSolve(ts, x));

    PetscCall(VecGetArray(x, &x_ptr));
    value_out = reinterpret_cast<Complex*>(x_ptr)[row * solver.layout.hilbert_dim + col];
    PetscCall(VecRestoreArray(x, &x_ptr));

    PetscCall(destroy_rhs_work_vectors(rhs_ctx));
    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    destroy_grouped_rhs_scratch_buffers(rhs_ctx);
    destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

} // namespace culindblad
