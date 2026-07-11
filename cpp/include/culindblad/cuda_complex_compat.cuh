#pragma once
// Device complex type + arithmetic that follow the build precision (types.hpp). The CUDA kernels
// were written against cuDoubleComplex / make_cuDoubleComplex / cuCadd / cuCmul / cuCsub / cuConj;
// this shim maps those to the single-precision cuComplex intrinsics when CULINDBLAD_FP32 is set, so
// the same kernel source builds both fp64 and fp32. Include ONLY from .cu translation units.
//
// Convention (applied to the .cu files): cuDoubleComplex -> DevComplex, make_cuDoubleComplex -> dev_mkc,
// cuCadd -> dev_add, cuCmul -> dev_mul, cuCsub -> dev_sub, cuConj -> dev_conj. DevComplex is layout-
// compatible with culindblad::Complex (both 2x the scalar), so reinterpret_cast<DevComplex*>(Complex*)
// stays valid; .x/.y members exist on both cuDoubleComplex and cuFloatComplex.

#include <cuComplex.h>

namespace culindblad {

#ifdef CULINDBLAD_FP32
using DevComplex = cuFloatComplex;
__host__ __device__ inline DevComplex dev_mkc(double r, double i) {
    return make_cuFloatComplex(static_cast<float>(r), static_cast<float>(i));
}
__host__ __device__ inline DevComplex dev_add(DevComplex a, DevComplex b) { return cuCaddf(a, b); }
__host__ __device__ inline DevComplex dev_mul(DevComplex a, DevComplex b) { return cuCmulf(a, b); }
__host__ __device__ inline DevComplex dev_sub(DevComplex a, DevComplex b) { return cuCsubf(a, b); }
__host__ __device__ inline DevComplex dev_conj(DevComplex a) { return cuConjf(a); }
#else
using DevComplex = cuDoubleComplex;
__host__ __device__ inline DevComplex dev_mkc(double r, double i) { return make_cuDoubleComplex(r, i); }
__host__ __device__ inline DevComplex dev_add(DevComplex a, DevComplex b) { return cuCadd(a, b); }
__host__ __device__ inline DevComplex dev_mul(DevComplex a, DevComplex b) { return cuCmul(a, b); }
__host__ __device__ inline DevComplex dev_sub(DevComplex a, DevComplex b) { return cuCsub(a, b); }
__host__ __device__ inline DevComplex dev_conj(DevComplex a) { return cuConj(a); }
#endif

} // namespace culindblad
