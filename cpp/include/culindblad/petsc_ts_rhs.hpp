#pragma once

#include <petscts.h>

namespace culindblad {

PetscErrorCode ts_rhs_function(TS ts, PetscReal t, Vec x, Vec f, void* ctx);

} // namespace culindblad
