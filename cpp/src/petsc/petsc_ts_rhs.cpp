#include <petscts.h>

#include "culindblad/petsc_apply.hpp"
#include "culindblad/petsc_ts_rhs.hpp"
#include "culindblad/solver.hpp"

namespace culindblad {

PetscErrorCode ts_rhs_function(TS ts, PetscReal t, Vec x, Vec f, void* ctx)
{
    Solver* solver = static_cast<Solver*>(ctx);
    if (!solver) {
        return PETSC_ERR_ARG_NULL;
    }

    PetscCall(apply_liouvillian_vec(*solver, x, f));
    return 0;
}

} // namespace culindblad