#include <petscmat.h>

#include "culindblad/petsc_apply.hpp"
#include "culindblad/petsc_shell.hpp"
#include "culindblad/solver.hpp"

namespace culindblad {

PetscErrorCode matshell_apply(Mat A, Vec x, Vec y)
{
    void* ctx_void = nullptr;
    PetscCall(MatShellGetContext(A, &ctx_void));

    Solver* solver = static_cast<Solver*>(ctx_void);
    if (!solver) {
        return PETSC_ERR_ARG_NULL;
    }

    PetscCall(apply_liouvillian_vec(*solver, x, y));
    return 0;
}

} // namespace culindblad
