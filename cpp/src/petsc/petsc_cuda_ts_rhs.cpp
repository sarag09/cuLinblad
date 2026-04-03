#include "culindblad/petsc_cuda_ts_rhs.hpp"

#include <petscts.h>
#include <petscvec.h>

#include <cuda_runtime.h>

#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/petsc_cuda_apply.hpp"
#include "culindblad/time_dependent_term.hpp"

namespace culindblad {

namespace {

Index rhs_total_density_dim(const PetscCudaTsRhsContext& rhs_ctx)
{
    return rhs_ctx.batch_size * rhs_ctx.solver->layout.density_dim;
}

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
    for (const CachedGroupedLayoutEntry& entry : rhs_ctx.cached_grouped_layouts) {
        if (same_sites(entry.sites, sites)) {
            return &entry;
        }
    }

    return nullptr;
}

CachedGroupedLayoutEntry* find_cached_grouped_layout_mutable(
    PetscCudaTsRhsContext& rhs_ctx,
    const std::vector<Index>& sites)
{
    for (CachedGroupedLayoutEntry& entry : rhs_ctx.cached_grouped_layouts) {
        if (same_sites(entry.sites, sites)) {
            return &entry;
        }
    }

    return nullptr;
}

PetscErrorCode zero_petsc_cuda_vec(
    Vec v,
    Index size,
    cudaStream_t stream)
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
            stream);

    if (memset_status != cudaSuccess) {
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
    Index size,
    cudaStream_t stream)
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
            stream);

    if (!add_ok) {
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
    Index size,
    cudaStream_t stream)
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
            stream);

    if (!scale_ok) {
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

PetscErrorCode zero_grouped_layout_accumulator(
    const PetscCudaTsRhsContext& rhs_ctx,
    CachedGroupedLayoutEntry& layout_entry,
    Index batch_size,
    cudaStream_t stream)
{
    const bool zero_ok =
        launch_zero_batched_buffer_kernel(
            rhs_ctx.grouped_scratch.d_grouped_accum,
            batch_size * layout_entry.grouped_layout.grouped_size,
            stream);

    return zero_ok ? 0 : PETSC_ERR_LIB;
}

PetscErrorCode accumulate_grouped_layout_term(
    const PetscCudaTsRhsContext& rhs_ctx,
    CachedGroupedLayoutEntry& layout_entry,
    double scale,
    Index batch_size,
    cudaStream_t stream)
{
    const Index grouped_elements =
        batch_size * layout_entry.grouped_layout.grouped_size;

    const bool accum_ok =
        scale == 1.0
            ? launch_vector_accumulate_kernel(
                  rhs_ctx.grouped_scratch.d_grouped_term,
                  rhs_ctx.grouped_scratch.d_grouped_accum,
                  grouped_elements,
                  stream)
            : launch_vector_scaled_accumulate_kernel(
                  rhs_ctx.grouped_scratch.d_grouped_term,
                  scale,
                  rhs_ctx.grouped_scratch.d_grouped_accum,
                  grouped_elements,
                  stream);

    return accum_ok ? 0 : PETSC_ERR_LIB;
}

PetscErrorCode apply_grouped_layout_terms_to_rhs(
    const Solver& solver,
    PetscCudaTsRhsContext& rhs_ctx,
    Vec x,
    Vec f,
    CachedGroupedLayoutEntry& layout_entry,
    PetscReal t)
{
    cudaStream_t s = rhs_ctx.elementwise_stream;
    if (layout_entry.grouped_bytes > rhs_ctx.grouped_scratch.grouped_bytes) {
        return PETSC_ERR_ARG_SIZ;
    }

    PetscCall(regroup_petsc_cuda_vec_to_grouped_buffer(
        layout_entry.cuda_grouped_layout,
        x,
        rhs_ctx.grouped_scratch.d_grouped_input,
        rhs_ctx.batch_size,
        s));

    PetscCall(zero_grouped_layout_accumulator(
        rhs_ctx,
        layout_entry,
        rhs_ctx.batch_size,
        s));

    bool has_contribution = false;

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        if (!same_sites(layout_entry.sites, h_term.sites)) {
            continue;
        }

        PetscCall(apply_grouped_commutator_cuda_buffer(
            solver,
            h_term.name,
            h_term.matrix,
            h_term.sites,
            layout_entry.grouped_layout,
            rhs_ctx.executor_cache,
            rhs_ctx.grouped_scratch.d_grouped_input,
            rhs_ctx.grouped_scratch.d_grouped_term,
            rhs_ctx.batch_size,
            s,
            nullptr,
            nullptr));

        PetscCall(accumulate_grouped_layout_term(
            rhs_ctx,
            layout_entry,
            1.0,
            rhs_ctx.batch_size,
            s));
        has_contribution = true;
    }

    for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx.cached_static_dissipators) {
        if (!same_sites(layout_entry.sites, d_aux.sites)) {
            continue;
        }

        PetscCall(apply_grouped_dissipator_cuda_buffer(
            solver,
            d_aux.name,
            d_aux.l_op,
            d_aux.l_dag,
            d_aux.l_dag_l,
            d_aux.sites,
            layout_entry.grouped_layout,
            rhs_ctx.executor_cache,
            rhs_ctx.grouped_scratch.d_grouped_input,
            rhs_ctx.grouped_scratch.d_grouped_term,
            rhs_ctx.batch_size,
            s,
            nullptr,
            nullptr));

        PetscCall(accumulate_grouped_layout_term(
            rhs_ctx,
            layout_entry,
            1.0,
            rhs_ctx.batch_size,
            s));
        has_contribution = true;
    }

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        if (!same_sites(layout_entry.sites, td_term.sites)) {
            continue;
        }

        const double coeff =
            evaluate_time_dependent_coefficient(td_term, static_cast<double>(t));

        PetscCall(apply_grouped_commutator_cuda_buffer(
            solver,
            td_term.name,
            td_term.matrix,
            td_term.sites,
            layout_entry.grouped_layout,
            rhs_ctx.executor_cache,
            rhs_ctx.grouped_scratch.d_grouped_input,
            rhs_ctx.grouped_scratch.d_grouped_term,
            rhs_ctx.batch_size,
            s,
            nullptr,
            nullptr));

        PetscCall(accumulate_grouped_layout_term(
            rhs_ctx,
            layout_entry,
            coeff,
            rhs_ctx.batch_size,
            s));
        has_contribution = true;
    }

    if (!has_contribution) {
        return 0;
    }

    PetscCall(flatten_grouped_buffer_to_petsc_cuda_vec(
        layout_entry.cuda_grouped_layout,
        rhs_ctx.grouped_scratch.d_grouped_accum,
        rhs_ctx.work_vec_a,
        rhs_ctx.batch_size,
        s,
        s));

    PetscCall(add_petsc_cuda_vecs(
        f,
        rhs_ctx.work_vec_a,
        rhs_ctx.work_vec_b,
        rhs_total_density_dim(rhs_ctx),
        s));

    PetscCall(VecCopy(rhs_ctx.work_vec_b, f));

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
        f,
        rhs_ctx->batch_size,
        rhs_ctx->elementwise_stream));

    if (cudaStreamSynchronize(rhs_ctx->elementwise_stream) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

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

    cudaStream_t s = rhs_ctx->elementwise_stream;

    Vec temp_comm = rhs_ctx->work_vec_a;
    Vec temp_diss = rhs_ctx->work_vec_b;

    PetscCall(apply_grouped_commutator_cuda_vec(
        *rhs_ctx->solver,
        "smoke_h",
        *rhs_ctx->h_local_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        temp_comm,
        rhs_ctx->batch_size,
        s));

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
        temp_diss,
        rhs_ctx->batch_size,
        s));

    PetscCall(add_petsc_cuda_vecs(
        temp_comm,
        temp_diss,
        f,
        rhs_total_density_dim(*rhs_ctx),
        s));

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

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
    cudaStream_t s = rhs_ctx->elementwise_stream;

    const Index total_density_dim = rhs_total_density_dim(*rhs_ctx);
    PetscCall(zero_petsc_cuda_vec(f, total_density_dim, s));

    for (CachedGroupedLayoutEntry& layout_entry : rhs_ctx->cached_grouped_layouts) {
        PetscCall(apply_grouped_layout_terms_to_rhs(
            solver,
            *rhs_ctx,
            x,
            f,
            layout_entry,
            0.0));
    }

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }

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
    cudaStream_t s = rhs_ctx->elementwise_stream;

    const Index total_density_dim = rhs_total_density_dim(*rhs_ctx);
    PetscCall(zero_petsc_cuda_vec(f, total_density_dim, s));

    for (CachedGroupedLayoutEntry& layout_entry : rhs_ctx->cached_grouped_layouts) {
        PetscCall(apply_grouped_layout_terms_to_rhs(
            solver,
            *rhs_ctx,
            x,
            f,
            layout_entry,
            t));
    }

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }
        
    return 0;
}

} // namespace culindblad
