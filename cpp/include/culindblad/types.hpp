#pragma once

#include <complex>
#include <cstddef>

namespace culindblad {

// Scalar precision is a build-time switch. Default = fp64 (double). Define CULINDBLAD_FP32
// (CMake option, -DCULINDBLAD_FP32=ON) to build the whole solver in single precision (complex
// float): PetscScalar must then also be complex-single (build run_sim against a PETSc arch
// configured --with-precision=single --with-scalar-type=complex). See cuda_complex_compat.cuh
// for the matching device (cuComplex) type and cutensor_*_desc for the 32F tensor/compute types.
#ifdef CULINDBLAD_FP32
using Real = float;
using Complex = std::complex<float>;
#else
using Real = double;
using Complex = std::complex<double>;
#endif
using Index = std::size_t;

} // namespace culindblad
