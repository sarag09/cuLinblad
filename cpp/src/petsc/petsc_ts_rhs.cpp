#include <petscvec.h>
#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/petsc_ts_rhs.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/state_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

PetscErrorCode ts_rhs_function(
    TS,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx)
{
    auto* rhs_ctx = static_cast<PetscTsRhsContext*>(ctx);
    if (!rhs_ctx || !rhs_ctx->solver) {
        return PETSC_ERR_ARG_NULL;
    }

    const Solver& solver = *rhs_ctx->solver;

    const PetscScalar* x_ptr = nullptr;
    PetscScalar* f_ptr = nullptr;

    PetscCall(VecGetArrayRead(x, &x_ptr));
    PetscCall(VecGetArray(f, &f_ptr));

    ConstStateBuffer in_buf{
        reinterpret_cast<const Complex*>(x_ptr),
        solver.layout.density_dim
    };

    StateBuffer out_buf{
        reinterpret_cast<Complex*>(f_ptr),
        solver.layout.density_dim
    };

    apply_liouvillian_at_time(
        solver,
        static_cast<double>(t),
        in_buf,
        out_buf);

    PetscCall(VecRestoreArray(f, &f_ptr));
    PetscCall(VecRestoreArrayRead(x, &x_ptr));

    return 0;
}

} // namespace culindblad