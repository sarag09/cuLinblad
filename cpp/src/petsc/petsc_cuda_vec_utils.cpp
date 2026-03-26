#include "culindblad/petsc_cuda_vec_utils.hpp"

#include <petscvec.h>

namespace culindblad {

PetscErrorCode petsc_cuda_vec_device_access_smoke(Vec x)
{
    PetscScalar* ptr = nullptr;

#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAGetArray(x, &ptr));
    PetscCall(VecCUDARestoreArray(x, &ptr));
#else
    (void)x;
#endif

    return 0;
}

} // namespace culindblad
