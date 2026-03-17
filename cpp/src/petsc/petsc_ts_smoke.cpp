#include <petscts.h>
#include <petscvec.h>

#include "culindblad/petsc_ts_rhs.hpp"
#include "culindblad/petsc_ts_smoke.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

PetscErrorCode run_ts_smoke_test(
    const Solver& solver,
    Index ket_index,
    Index bra_index,
    Complex& out_value)
{
    Vec x_ts = nullptr;
    TS ts = nullptr;

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, solver.layout.density_dim, &x_ts));
    PetscCall(VecSet(x_ts, 0.0));

    PetscScalar* x_ts_ptr = nullptr;
    PetscCall(VecGetArray(x_ts, &x_ts_ptr));
    x_ts_ptr[ket_index * solver.layout.hilbert_dim + bra_index] = PetscScalar(1.0);
    PetscCall(VecRestoreArray(x_ts, &x_ts_ptr));

    PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
    PetscCall(TSSetType(ts, TSRK));
    PetscCall(TSSetRHSFunction(ts, nullptr, ts_rhs_function, const_cast<Solver*>(&solver)));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetTimeStep(ts, 1.0e-3));
    PetscCall(TSSetMaxSteps(ts, 1));
    PetscCall(TSSetMaxTime(ts, 1.0e-3));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSolve(ts, x_ts));

    PetscCall(VecGetArray(x_ts, &x_ts_ptr));
    out_value = reinterpret_cast<Complex*>(x_ts_ptr)[ket_index * solver.layout.hilbert_dim + bra_index];
    PetscCall(VecRestoreArray(x_ts, &x_ts_ptr));

    PetscCall(TSDestroy(&ts));
    PetscCall(VecDestroy(&x_ts));

    return 0;
}

} // namespace culindblad
