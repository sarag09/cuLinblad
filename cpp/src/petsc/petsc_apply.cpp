#include <exception>
#include <iostream>

#include <petscvec.h>

#include "culindblad/backend.hpp"
#include "culindblad/petsc_apply.hpp"
#include "culindblad/state_buffer.hpp"
#include "culindblad/solver.hpp"

namespace culindblad {

PetscErrorCode apply_liouvillian_vec(
    const Solver& solver,
    Vec x,
    Vec y)
{
    PetscErrorCode ierr = 0;

    PetscInt x_size = 0;
    PetscInt y_size = 0;

    PetscCall(VecGetSize(x, &x_size));
    PetscCall(VecGetSize(y, &y_size));

    if (static_cast<Index>(x_size) != solver.layout.density_dim) {
        std::cerr << "apply_liouvillian_vec: x has wrong size." << std::endl;
        return PETSC_ERR_ARG_SIZ;
    }

    if (static_cast<Index>(y_size) != solver.layout.density_dim) {
        std::cerr << "apply_liouvillian_vec: y has wrong size." << std::endl;
        return PETSC_ERR_ARG_SIZ;
    }

    const PetscScalar* x_ptr = nullptr;
    PetscScalar* y_ptr = nullptr;

    PetscCall(VecGetArrayRead(x, &x_ptr));

    ierr = VecGetArray(y, &y_ptr);
    if (ierr != 0) {
        VecRestoreArrayRead(x, &x_ptr);
        return ierr;
    }

    try {
        ConstStateBuffer in_buf{
            reinterpret_cast<const Complex*>(x_ptr),
            solver.layout.density_dim
        };

        StateBuffer out_buf{
            reinterpret_cast<Complex*>(y_ptr),
            solver.layout.density_dim
        };

        apply_liouvillian(solver, in_buf, out_buf);
    } catch (const std::exception& ex) {
        std::cerr << "apply_liouvillian_vec exception: " << ex.what() << std::endl;
        VecRestoreArrayRead(x, &x_ptr);
        VecRestoreArray(y, &y_ptr);
        return PETSC_ERR_LIB;
    }

    ierr = VecRestoreArrayRead(x, &x_ptr);
    if (ierr != 0) {
        VecRestoreArray(y, &y_ptr);
        return ierr;
    }

    PetscCall(VecRestoreArray(y, &y_ptr));

    return 0;
}

} // namespace culindblad