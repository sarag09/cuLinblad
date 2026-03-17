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

    ierr = VecGetSize(x, &x_size);
    if (ierr != 0) {
        return ierr;
    }

    ierr = VecGetSize(y, &y_size);
    if (ierr != 0) {
        return ierr;
    }

    if (static_cast<Index>(x_size) != solver.layout.density_dim) {
        std::cerr << "apply_liouvillian_vec: x has wrong size." << std::endl;
        return PETSC_ERR_ARG_SIZ;
    }

    if (static_cast<Index>(y_size) != solver.layout.density_dim) {
        std::cerr << "apply_liouvillian_vec: y has wrong size." << std::endl;
        return PETSC_ERR_ARG_SIZ;
    }

    PetscScalar* x_ptr = nullptr;
    PetscScalar* y_ptr = nullptr;

    ierr = VecGetArray(x, &x_ptr);
    if (ierr != 0) {
        return ierr;
    }

    ierr = VecGetArray(y, &y_ptr);
    if (ierr != 0) {
        VecRestoreArray(x, &x_ptr);
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
        VecRestoreArray(x, &x_ptr);
        VecRestoreArray(y, &y_ptr);
        return PETSC_ERR_LIB;
    }

    ierr = VecRestoreArray(x, &x_ptr);
    if (ierr != 0) {
        VecRestoreArray(y, &y_ptr);
        return ierr;
    }

    ierr = VecRestoreArray(y, &y_ptr);
    if (ierr != 0) {
        return ierr;
    }

    return 0;
}

} // namespace culindblad
