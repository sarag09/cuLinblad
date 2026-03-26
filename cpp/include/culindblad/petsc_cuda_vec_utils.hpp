#pragma once

#include <petscvec.h>

namespace culindblad {

PetscErrorCode petsc_cuda_vec_device_access_smoke(Vec x);

} // namespace culindblad
