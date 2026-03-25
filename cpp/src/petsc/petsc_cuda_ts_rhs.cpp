#include "culindblad/petsc_cuda_ts_rhs.hpp"

#include <petscts.h>
#include <petscvec.h>

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
    if (!rhs_ctx || !rhs_ctx->solver || !rhs_ctx->local_op) {
        return PETSC_ERR_ARG_NULL;
    }

    PetscCall(apply_grouped_commutator_cuda_vec(
        *rhs_ctx->solver,
        *rhs_ctx->local_op,
        rhs_ctx->target_sites,
        rhs_ctx->grouped_layout,
        rhs_ctx->executor_cache,
        x,
        f));

    return 0;
}

} // namespace culindblad