#include "culindblad/petsc_cuda_ts_smoke.hpp"

#include <petscts.h>
#include <petscvec.h>

#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/petsc_cuda_ts_rhs.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

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

    PetscCudaTsRhsContext rhs_ctx{
        &solver,
        &local_op,
        target_sites,
        make_grouped_state_layout(target_sites, solver.model.local_dims),
        CuTensorExecutorCache{}
    };

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

    const bool cache_destroy_ok =
        destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    (void)cache_destroy_ok;

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

} // namespace culindblad
