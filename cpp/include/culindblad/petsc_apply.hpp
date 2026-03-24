#pragma once

#include <petscvec.h>

#include "culindblad/solver.hpp"

namespace culindblad {

PetscErrorCode apply_liouvillian_vec(
    const Solver& solver,
    Vec x,
    Vec y);

PetscErrorCode petsc_cuda_vec_smoke_test(
    const Solver& solver,
    Index row,
    Index col,
    Complex& value_out);    

} // namespace culindblad
