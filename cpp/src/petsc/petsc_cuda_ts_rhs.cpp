#include "culindblad/petsc_cuda_ts_rhs.hpp"

#include <petscts.h>
#include <petscvec.h>

#include <cuda_runtime.h>

#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/local_operator_utils.hpp"
#include "culindblad/petsc_cuda_apply.hpp"
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

const CachedGroupedLayoutEntry* find_cached_grouped_layout(
    const PetscCudaTsRhsContext& rhs_ctx,
    const std::vector<Index>& sites)
{
    for (const CachedGroupedLayoutEntry& entry :
         rhs_ctx.cached_grouped_layouts) {
        if (same_sites(entry.sites, sites)) {
            return &entry;
        }
    }

    return nullptr;
}

PetscErrorCode zero_petsc_cuda_vec(Vec v, Index size)
{
    PetscScalar* d_v_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArray(v, &d_v_ptr));
#else
    PetscCall(VecGetArray(v, &d_v_ptr));
#endif

    const cudaError_t memset_status =
        cudaMemsetAsync(
            reinterpret_cast<void*>(d_v_ptr),
            0,
            size * sizeof(Complex),
            0);

    if (memset_status != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(v, &d_v_ptr));
#else
        PetscCall(VecRestoreArray(v, &d_v_ptr));
#endif
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(v, &d_v_ptr));
#else
    PetscCall(VecRestoreArray(v, &d_v_ptr));
#endif

    return 0;
}

PetscErrorCode add_petsc_cuda_vecs(
    Vec a,
    Vec b,
    Vec out,
    Index size)
{
    const PetscScalar* d_a_ptr = nullptr;
    const PetscScalar* d_b_ptr = nullptr;
    PetscScalar* d_out_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArrayRead(a, &d_a_ptr));
    PetscCall(VecCUDAGetArrayRead(b, &d_b_ptr));
    PetscCall(VecCUDAGetArray(out, &d_out_ptr));
#else
    PetscCall(VecGetArrayRead(a, &d_a_ptr));
    PetscCall(VecGetArrayRead(b, &d_b_ptr));
    PetscCall(VecGetArray(out, &d_out_ptr));
#endif

    const bool add_ok =
        launch_vector_add_kernel(
            reinterpret_cast<const void*>(d_a_ptr),
            reinterpret_cast<const void*>(d_b_ptr),
            reinterpret_cast<void*>(d_out_ptr),
            size,
            0);

    if (!add_ok || cudaDeviceSynchronize() != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
        PetscCall(VecCUDARestoreArrayRead(b, &d_b_ptr));
        PetscCall(VecCUDARestoreArrayRead(a, &d_a_ptr));
#else
        PetscCall(VecRestoreArray(out, &d_out_ptr));
        PetscCall(VecRestoreArrayRead(b, &d_b_ptr));
        PetscCall(VecRestoreArrayRead(a, &d_a_ptr));
#endif
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
    PetscCall(VecCUDARestoreArrayRead(b, &d_b_ptr));
    PetscCall(VecCUDARestoreArrayRead(a, &d_a_ptr));
#else
    PetscCall(VecRestoreArray(out, &d_out_ptr));
    PetscCall(VecRestoreArrayRead(b, &d_b_ptr));
    PetscCall(VecRestoreArrayRead(a, &d_a_ptr));
#endif

    return 0;
}

PetscErrorCode scale_petsc_cuda_vec(
    Vec in,
    double scale,
    Vec out,
    Index size)
{
    const PetscScalar* d_in_ptr = nullptr;
    PetscScalar* d_out_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArrayRead(in, &d_in_ptr));
    PetscCall(VecCUDAGetArray(out, &d_out_ptr));
#else
    PetscCall(VecGetArrayRead(in, &d_in_ptr));
    PetscCall(VecGetArray(out, &d_out_ptr));
#endif

    const bool scale_ok =
        launch_vector_scale_kernel(
            reinterpret_cast<const void*>(d_in_ptr),
            scale,
            reinterpret_cast<void*>(d_out_ptr),
            size,
            0);

    if (!scale_ok || cudaDeviceSynchronize() != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
        PetscCall(VecCUDARestoreArrayRead(in, &d_in_ptr));
#else
        PetscCall(VecRestoreArray(out, &d_out_ptr));
        PetscCall(VecRestoreArrayRead(in, &d_in_ptr));
#endif
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
    PetscCall(VecCUDARestoreArrayRead(in, &d_in_ptr));
#else
    PetscCall(VecRestoreArray(out, &d_out_ptr));
    PetscCall(VecRestoreArrayRead(in, &d_in_ptr));
#endif

    return 0;
}

} // namespace

PetscErrorCode ts_rhs_function_cuda_grouped_commutator(
    TS,
    PetscReal,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver || !rhs_ctx->h_local_op) {
        return PETSC_ERR_ARG_NULL;
    }

    PetscCall(apply_grouped_commutator_cuda_vec(
        *rhs_ctx->solver,
        "smoke_h",
        *rhs_ctx->h_local_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        f));

    return 0;
}

PetscErrorCode ts_rhs_function_cuda_grouped_liouvillian(
    TS,
    PetscReal,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver || !rhs_ctx->h_local_op ||
        !rhs_ctx->d_local_op || !rhs_ctx->d_local_op_dag || !rhs_ctx->d_local_op_dag_op) {
        return PETSC_ERR_ARG_NULL;
    }

    Vec temp_comm = nullptr;
    Vec temp_diss = nullptr;

    PetscCall(VecDuplicate(x, &temp_comm));
    PetscCall(VecDuplicate(x, &temp_diss));

    PetscCall(apply_grouped_commutator_cuda_vec(
        *rhs_ctx->solver,
        "smoke_h",
        *rhs_ctx->h_local_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        temp_comm));

    PetscCall(apply_grouped_dissipator_cuda_vec(
        *rhs_ctx->solver,
        "smoke_d",
        *rhs_ctx->d_local_op,
        *rhs_ctx->d_local_op_dag,
        *rhs_ctx->d_local_op_dag_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        temp_diss));

    PetscCall(add_petsc_cuda_vecs(
        temp_comm,
        temp_diss,
        f,
        rhs_ctx->solver->layout.density_dim));

    PetscCall(VecDestroy(&temp_diss));
    PetscCall(VecDestroy(&temp_comm));
    return 0;
}

PetscErrorCode ts_rhs_function_cuda_static_model_liouvillian(
    TS,
    PetscReal,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;

    PetscCall(zero_petsc_cuda_vec(f, solver.layout.density_dim));

    Vec term_out = nullptr;
    Vec accum = nullptr;

    PetscCall(VecDuplicate(x, &term_out));
    PetscCall(VecDuplicate(x, &accum));

    PetscCall(zero_petsc_cuda_vec(accum, solver.layout.density_dim));

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, h_term.sites);

        if (layout_entry == nullptr) {
            PetscCall(VecDestroy(&accum));
            PetscCall(VecDestroy(&term_out));
            return PETSC_ERR_LIB;
        }

        PetscCall(apply_grouped_commutator_cuda_vec(
            solver,
            h_term.name,
            h_term.matrix,
            h_term.sites,
            layout_entry->grouped_layout,
            layout_entry->cuda_grouped_layout,
            rhs_ctx->executor_cache,
            x,
            term_out));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim));

        PetscCall(VecCopy(f, accum));
    }

    for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx->cached_static_dissipators) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, d_aux.sites);

        if (layout_entry == nullptr) {
            PetscCall(VecDestroy(&accum));
            PetscCall(VecDestroy(&term_out));
            return PETSC_ERR_LIB;
        }

        PetscCall(apply_grouped_dissipator_cuda_vec(
            solver,
            d_aux.name,
            d_aux.l_op,
            d_aux.l_dag,
            d_aux.l_dag_l,
            d_aux.sites,
            layout_entry->grouped_layout,
            layout_entry->cuda_grouped_layout,
            rhs_ctx->executor_cache,
            x,
            term_out));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim));

        PetscCall(VecCopy(f, accum));
    }

    PetscCall(VecCopy(accum, f));

    PetscCall(VecDestroy(&accum));
    PetscCall(VecDestroy(&term_out));
    return 0;
}

PetscErrorCode ts_rhs_function_cuda_full_model_liouvillian(
    TS,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscCudaTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;

    PetscCall(zero_petsc_cuda_vec(f, solver.layout.density_dim));

    Vec term_out = nullptr;
    Vec scaled_term_out = nullptr;
    Vec accum = nullptr;

    PetscCall(VecDuplicate(x, &term_out));
    PetscCall(VecDuplicate(x, &scaled_term_out));
    PetscCall(VecDuplicate(x, &accum));

    PetscCall(zero_petsc_cuda_vec(accum, solver.layout.density_dim));

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, h_term.sites);

        if (layout_entry == nullptr) {
            PetscCall(VecDestroy(&accum));
            PetscCall(VecDestroy(&scaled_term_out));
            PetscCall(VecDestroy(&term_out));
            return PETSC_ERR_LIB;
        }

        PetscCall(apply_grouped_commutator_cuda_vec(
            solver,
            h_term.name,
            h_term.matrix,
            h_term.sites,
            layout_entry->grouped_layout,
            layout_entry->cuda_grouped_layout,
            rhs_ctx->executor_cache,
            x,
            term_out));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim));

        PetscCall(VecCopy(f, accum));
    }

    for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx->cached_static_dissipators) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, d_aux.sites);

        if (layout_entry == nullptr) {
            PetscCall(VecDestroy(&accum));
            PetscCall(VecDestroy(&scaled_term_out));
            PetscCall(VecDestroy(&term_out));
            return PETSC_ERR_LIB;
        }

        PetscCall(apply_grouped_dissipator_cuda_vec(
            solver,
            d_aux.name,
            d_aux.l_op,
            d_aux.l_dag,
            d_aux.l_dag_l,
            d_aux.sites,
            layout_entry->grouped_layout,
            layout_entry->cuda_grouped_layout,
            rhs_ctx->executor_cache,
            x,
            term_out));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim));

        PetscCall(VecCopy(f, accum));
    }

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, td_term.sites);

        if (layout_entry == nullptr) {
            PetscCall(VecDestroy(&accum));
            PetscCall(VecDestroy(&scaled_term_out));
            PetscCall(VecDestroy(&term_out));
            return PETSC_ERR_LIB;
        }

        const double coeff =
            evaluate_time_dependent_coefficient(td_term, static_cast<double>(t));

        PetscCall(apply_grouped_commutator_cuda_vec(
            solver,
            td_term.name,
            td_term.matrix,
            td_term.sites,
            layout_entry->grouped_layout,
            layout_entry->cuda_grouped_layout,
            rhs_ctx->executor_cache,
            x,
            term_out));

        PetscCall(scale_petsc_cuda_vec(
            term_out,
            coeff,
            scaled_term_out,
            solver.layout.density_dim));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            scaled_term_out,
            f,
            solver.layout.density_dim));

        PetscCall(VecCopy(f, accum));
    }

    PetscCall(VecCopy(accum, f));

    PetscCall(VecDestroy(&accum));
    PetscCall(VecDestroy(&scaled_term_out));
    PetscCall(VecDestroy(&term_out));
    return 0;
}

} // namespace culindblad