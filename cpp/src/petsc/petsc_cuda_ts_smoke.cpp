#include "culindblad/petsc_cuda_ts_smoke.hpp"

#include <petscts.h>
#include <petscvec.h>
#include <stdexcept>

#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"
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
    const Solver& solver)
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
        nullptr
    };

    rhs_ctx.work_vec_a = nullptr;
    rhs_ctx.work_vec_b = nullptr;
    rhs_ctx.work_vec_c = nullptr;

    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_a));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_b));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_c));

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        if (rhs_ctx.work_vec_c != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
        if (rhs_ctx.work_vec_b != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
        if (rhs_ctx.work_vec_a != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

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

    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    if (rhs_ctx.work_vec_c != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
    }
    if (rhs_ctx.work_vec_b != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
    }
    if (rhs_ctx.work_vec_a != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
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
        nullptr
    };

    rhs_ctx.work_vec_a = nullptr;
    rhs_ctx.work_vec_b = nullptr;
    rhs_ctx.work_vec_c = nullptr;

    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_a));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_b));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_c));

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        if (rhs_ctx.work_vec_c != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
        if (rhs_ctx.work_vec_b != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
        if (rhs_ctx.work_vec_a != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

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

    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    if (rhs_ctx.work_vec_c != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
    }
    if (rhs_ctx.work_vec_b != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
    }
    if (rhs_ctx.work_vec_a != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
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
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        build_cached_static_dissipators(solver),
        build_cached_grouped_layouts(solver),
        nullptr
    };

    rhs_ctx.work_vec_a = nullptr;
    rhs_ctx.work_vec_b = nullptr;
    rhs_ctx.work_vec_c = nullptr;

    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_a));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_b));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_c));

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        if (rhs_ctx.work_vec_c != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
        if (rhs_ctx.work_vec_b != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
        if (rhs_ctx.work_vec_a != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

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

    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    if (rhs_ctx.work_vec_c != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
    }
    if (rhs_ctx.work_vec_b != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
    }
    if (rhs_ctx.work_vec_a != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
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
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        target_sites,
        grouped_layout,
        cuda_grouped_layout,
        CuTensorExecutorCache{},
        build_cached_static_dissipators(solver),
        build_cached_grouped_layouts(solver),
        nullptr
    };

    rhs_ctx.work_vec_a = nullptr;
    rhs_ctx.work_vec_b = nullptr;
    rhs_ctx.work_vec_c = nullptr;

    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_a));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_b));
    PetscCall(VecDuplicate(x, &rhs_ctx.work_vec_c));

    if (cudaStreamCreate(&rhs_ctx.elementwise_stream) != cudaSuccess) {
        destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
        (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);
        if (rhs_ctx.work_vec_c != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
        if (rhs_ctx.work_vec_b != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
        if (rhs_ctx.work_vec_a != nullptr) PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
        PetscCall(VecDestroy(&x));
        PetscCall(TSDestroy(&ts));
        return PETSC_ERR_LIB;
    }

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

    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    destroy_cached_grouped_layouts(rhs_ctx.cached_grouped_layouts);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

    if (rhs_ctx.elementwise_stream != nullptr) {
        cudaStreamDestroy(rhs_ctx.elementwise_stream);
    }

    if (rhs_ctx.work_vec_c != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_c));
    }
    if (rhs_ctx.work_vec_b != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_b));
    }
    if (rhs_ctx.work_vec_a != nullptr) {
        PetscCall(VecDestroy(&rhs_ctx.work_vec_a));
    }

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

} // namespace culindblad