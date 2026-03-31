#include "culindblad/petsc_cuda_ts_rhs.hpp"

#include <petscts.h>
#include <petscvec.h>

#include <cuda_runtime.h>

#include "culindblad/cuda_elementwise.hpp"
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
    for (const CachedGroupedLayoutEntry& entry : rhs_ctx.cached_grouped_layouts) {
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
    bool restore_a_read = false;
    bool restore_b_read = false;
    bool restore_out_write = false;

#if defined(PETSC_HAVE_CUDA)
    if (a == out) {
        PetscCall(VecCUDAGetArray(out, &d_out_ptr));
        d_a_ptr = d_out_ptr;
        restore_out_write = true;
    } else {
        PetscCall(VecCUDAGetArrayRead(a, &d_a_ptr));
        restore_a_read = true;
    }

    if (b == out) {
        if (!restore_out_write) {
            PetscCall(VecCUDAGetArray(out, &d_out_ptr));
            restore_out_write = true;
        }
        d_b_ptr = d_out_ptr;
    } else {
        PetscCall(VecCUDAGetArrayRead(b, &d_b_ptr));
        restore_b_read = true;
    }

    if (!restore_out_write) {
        PetscCall(VecCUDAGetArray(out, &d_out_ptr));
        restore_out_write = true;
    }
#else
    if (a == out) {
        PetscCall(VecGetArray(out, &d_out_ptr));
        d_a_ptr = d_out_ptr;
        restore_out_write = true;
    } else {
        PetscCall(VecGetArrayRead(a, &d_a_ptr));
        restore_a_read = true;
    }

    if (b == out) {
        if (!restore_out_write) {
            PetscCall(VecGetArray(out, &d_out_ptr));
            restore_out_write = true;
        }
        d_b_ptr = d_out_ptr;
    } else {
        PetscCall(VecGetArrayRead(b, &d_b_ptr));
        restore_b_read = true;
    }

    if (!restore_out_write) {
        PetscCall(VecGetArray(out, &d_out_ptr));
        restore_out_write = true;
    }
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
        if (restore_out_write) {
            PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
        }
        if (restore_b_read) {
            PetscCall(VecCUDARestoreArrayRead(b, &d_b_ptr));
        }
        if (restore_a_read) {
            PetscCall(VecCUDARestoreArrayRead(a, &d_a_ptr));
        }
#else
        if (restore_out_write) {
            PetscCall(VecRestoreArray(out, &d_out_ptr));
        }
        if (restore_b_read) {
            PetscCall(VecRestoreArrayRead(b, &d_b_ptr));
        }
        if (restore_a_read) {
            PetscCall(VecRestoreArrayRead(a, &d_a_ptr));
        }
#endif
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    if (restore_out_write) {
        PetscCall(VecCUDARestoreArray(out, &d_out_ptr));
    }
    if (restore_b_read) {
        PetscCall(VecCUDARestoreArrayRead(b, &d_b_ptr));
    }
    if (restore_a_read) {
        PetscCall(VecCUDARestoreArrayRead(a, &d_a_ptr));
    }
#else
    if (restore_out_write) {
        PetscCall(VecRestoreArray(out, &d_out_ptr));
    }
    if (restore_b_read) {
        PetscCall(VecRestoreArrayRead(b, &d_b_ptr));
    }
    if (restore_a_read) {
        PetscCall(VecRestoreArrayRead(a, &d_a_ptr));
    }
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
        rhs_ctx->elementwise_stream));

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
        s));

    PetscCall(add_petsc_cuda_vecs(
        temp_comm,
        temp_diss,
        f,
        rhs_ctx->solver->layout.density_dim,
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

    Vec term_out = rhs_ctx->work_vec_a;
    Vec accum = rhs_ctx->work_vec_b;

    PetscCall(zero_petsc_cuda_vec(f, solver.layout.density_dim, s));
    PetscCall(zero_petsc_cuda_vec(accum, solver.layout.density_dim, s));

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, h_term.sites);

        if (layout_entry == nullptr) {
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
            term_out,
            s));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim,
            s));

        PetscCall(VecCopy(f, accum));
    }

    for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx->cached_static_dissipators) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, d_aux.sites);

        if (layout_entry == nullptr) {
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
            term_out,
            s));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim,
            s));

        PetscCall(VecCopy(f, accum));
    }

    PetscCall(VecCopy(accum, f));

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

    Vec term_out = rhs_ctx->work_vec_a;
    Vec scaled_term_out = rhs_ctx->work_vec_b;
    Vec accum = rhs_ctx->work_vec_c;

    PetscCall(zero_petsc_cuda_vec(f, solver.layout.density_dim, s));
    PetscCall(zero_petsc_cuda_vec(accum, solver.layout.density_dim, s));

    for (const OperatorTerm& h_term : solver.model.hamiltonian_terms) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, h_term.sites);

        if (layout_entry == nullptr) {
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
            term_out,
            s));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim,
            s));

        PetscCall(VecCopy(f, accum));
    }

    for (const CachedDissipatorAuxiliaries& d_aux : rhs_ctx->cached_static_dissipators) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, d_aux.sites);

        if (layout_entry == nullptr) {
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
            term_out,
            s));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            term_out,
            f,
            solver.layout.density_dim,
            s));

        PetscCall(VecCopy(f, accum));
    }

    for (const TimeDependentTerm& td_term : solver.model.time_dependent_hamiltonian_terms) {
        const CachedGroupedLayoutEntry* layout_entry =
            find_cached_grouped_layout(*rhs_ctx, td_term.sites);

        if (layout_entry == nullptr) {
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
            term_out,
            s));

        PetscCall(scale_petsc_cuda_vec(
            term_out,
            coeff,
            scaled_term_out,
            solver.layout.density_dim,
            s));

        PetscCall(add_petsc_cuda_vecs(
            accum,
            scaled_term_out,
            f,
            solver.layout.density_dim,
            s));

        PetscCall(VecCopy(f, accum));
    }

    PetscCall(VecCopy(accum, f));

    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return PETSC_ERR_LIB;
    }
        
    return 0;
}

} // namespace culindblad
