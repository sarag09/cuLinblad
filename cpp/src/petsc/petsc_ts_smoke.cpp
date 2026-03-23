#include "culindblad/petsc_ts_smoke.hpp"

#include <petscts.h>
#include <petscvec.h>

#include "culindblad/petsc_ts_rhs.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

PetscErrorCode run_ts_smoke_test(
    const Solver& solver,
    Index row,
    Index col,
    Complex& value_out)
{
    TS ts = nullptr;
    Vec x = nullptr;

    PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
    PetscCall(TSSetType(ts, TSEULER));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, solver.layout.density_dim, &x));
    PetscCall(VecSet(x, 0.0));

    PetscScalar* x_ptr = nullptr;
    PetscCall(VecGetArray(x, &x_ptr));
    x_ptr[row * solver.layout.hilbert_dim + col] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x, &x_ptr));

    PetscTsRhsContext rhs_ctx{&solver};

    PetscCall(TSSetRHSFunction(ts, nullptr, ts_rhs_function, &rhs_ctx));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetTimeStep(ts, 1.0e-3));
    PetscCall(TSSetMaxSteps(ts, 1));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSolve(ts, x));

    PetscCall(VecGetArray(x, &x_ptr));
    value_out = reinterpret_cast<Complex*>(x_ptr)[row * solver.layout.hilbert_dim + col];
    PetscCall(VecRestoreArray(x, &x_ptr));

    PetscCall(VecDestroy(&x));
    PetscCall(TSDestroy(&ts));
    return 0;
}

} // namespace culindblad