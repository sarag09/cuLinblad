#pragma once

#include <petscvec.h>

#include "culindblad/solver.hpp"

namespace culindblad {

PetscErrorCode apply_liouvillian_vec(
    const Solver& solver,
    Vec x,
    Vec y);

} // namespace culindblad
