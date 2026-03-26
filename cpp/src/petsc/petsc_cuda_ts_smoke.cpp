#include "culindblad/petsc_cuda_ts_smoke.hpp"

#include <petscts.h>
#include <petscvec.h>

#include "culindblad/cuda_grouped_layout.hpp"
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

    (void)destroy_cutensor_executor_cache(rhs_ctx.executor_cache);
    (void)destroy_cuda_grouped_state_layout(rhs_ctx.cuda_grouped_layout);

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
        CuTensorExecutorCache{}
    };

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

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

} // namespace culindblad