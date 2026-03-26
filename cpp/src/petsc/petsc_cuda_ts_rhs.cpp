#include "culindblad/petsc_cuda_ts_rhs.hpp"

#include <petscts.h>
#include <petscvec.h>

#include <cuda_runtime.h>

#include "culindblad/cuda_elementwise.hpp"
#include "culindblad/petsc_cuda_apply.hpp"

namespace culindblad {

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
        *rhs_ctx->h_local_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        temp_comm));

    PetscCall(apply_grouped_dissipator_cuda_vec(
        *rhs_ctx->solver,
        *rhs_ctx->d_local_op,
        *rhs_ctx->d_local_op_dag,
        *rhs_ctx->d_local_op_dag_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->cuda_grouped_layout,
        rhs_ctx->executor_cache,
        x,
        temp_diss));

    const PetscScalar* d_comm_ptr = nullptr;
    const PetscScalar* d_diss_ptr = nullptr;
    PetscScalar* d_f_ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArrayRead(temp_comm, &d_comm_ptr));
    PetscCall(VecCUDAGetArrayRead(temp_diss, &d_diss_ptr));
    PetscCall(VecCUDAGetArray(f, &d_f_ptr));
#else
    PetscCall(VecGetArrayRead(temp_comm, &d_comm_ptr));
    PetscCall(VecGetArrayRead(temp_diss, &d_diss_ptr));
    PetscCall(VecGetArray(f, &d_f_ptr));
#endif

    const bool add_ok =
        launch_vector_add_kernel(
            reinterpret_cast<const void*>(d_comm_ptr),
            reinterpret_cast<const void*>(d_diss_ptr),
            reinterpret_cast<void*>(d_f_ptr),
            rhs_ctx->solver->layout.density_dim,
            0);

    if (!add_ok) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(f, &d_f_ptr));
        PetscCall(VecCUDARestoreArrayRead(temp_diss, &d_diss_ptr));
        PetscCall(VecCUDARestoreArrayRead(temp_comm, &d_comm_ptr));
#else
        PetscCall(VecRestoreArray(f, &d_f_ptr));
        PetscCall(VecRestoreArrayRead(temp_diss, &d_diss_ptr));
        PetscCall(VecRestoreArrayRead(temp_comm, &d_comm_ptr));
#endif
        PetscCall(VecDestroy(&temp_diss));
        PetscCall(VecDestroy(&temp_comm));
        return PETSC_ERR_LIB;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(f, &d_f_ptr));
        PetscCall(VecCUDARestoreArrayRead(temp_diss, &d_diss_ptr));
        PetscCall(VecCUDARestoreArrayRead(temp_comm, &d_comm_ptr));
#else
        PetscCall(VecRestoreArray(f, &d_f_ptr));
        PetscCall(VecRestoreArrayRead(temp_diss, &d_diss_ptr));
        PetscCall(VecRestoreArrayRead(temp_comm, &d_comm_ptr));
#endif
        PetscCall(VecDestroy(&temp_diss));
        PetscCall(VecDestroy(&temp_comm));
        return PETSC_ERR_LIB;
    }

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDARestoreArray(f, &d_f_ptr));
    PetscCall(VecCUDARestoreArrayRead(temp_diss, &d_diss_ptr));
    PetscCall(VecCUDARestoreArrayRead(temp_comm, &d_comm_ptr));
#else
    PetscCall(VecRestoreArray(f, &d_f_ptr));
    PetscCall(VecRestoreArrayRead(temp_diss, &d_diss_ptr));
    PetscCall(VecRestoreArrayRead(temp_comm, &d_comm_ptr));
#endif

    PetscCall(VecDestroy(&temp_diss));
    PetscCall(VecDestroy(&temp_comm));
    return 0;
}

} // namespace culindblad