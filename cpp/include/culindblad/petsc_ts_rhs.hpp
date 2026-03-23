#pragma once

#include <petscts.h>

#include "culindblad/solver.hpp"

namespace culindblad {

struct PetscTsRhsContext {
    const Solver* solver;
};

PetscErrorCode ts_rhs_function(
    TS ts,
    PetscReal t,
    Vec x,
    Vec f,
    void* ctx);

} // namespace culindblad