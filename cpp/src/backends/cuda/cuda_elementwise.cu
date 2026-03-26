#include "culindblad/cuda_elementwise.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "culindblad/types.hpp"

namespace culindblad {

namespace {

static_assert(sizeof(Complex) == sizeof(cuDoubleComplex),
              "Complex must match cuDoubleComplex layout");

__global__ void commutator_combine_kernel(
    const cuDoubleComplex* left,
    const cuDoubleComplex* right,
    cuDoubleComplex* out,
    std::size_t n)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const cuDoubleComplex minus_i = make_cuDoubleComplex(0.0, -1.0);
    out[idx] = cuCmul(minus_i, cuCsub(left[idx], right[idx]));
}

__global__ void dissipator_combine_kernel(
    const cuDoubleComplex* jump,
    const cuDoubleComplex* left,
    const cuDoubleComplex* right,
    cuDoubleComplex* out,
    std::size_t n)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const cuDoubleComplex half = make_cuDoubleComplex(0.5, 0.0);
    out[idx] = cuCsub(
        jump[idx],
        cuCmul(half, cuCadd(left[idx], right[idx])));
}

} // namespace

bool launch_commutator_combine_kernel(
    const void* d_left,
    const void* d_right,
    void* d_out,
    std::size_t num_elements,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((num_elements + block_size - 1) / block_size);

    commutator_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(d_left),
        reinterpret_cast<const cuDoubleComplex*>(d_right),
        reinterpret_cast<cuDoubleComplex*>(d_out),
        num_elements);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_dissipator_combine_kernel(
    const void* d_jump,
    const void* d_left,
    const void* d_right,
    void* d_out,
    std::size_t num_elements,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((num_elements + block_size - 1) / block_size);

    dissipator_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(d_jump),
        reinterpret_cast<const cuDoubleComplex*>(d_left),
        reinterpret_cast<const cuDoubleComplex*>(d_right),
        reinterpret_cast<cuDoubleComplex*>(d_out),
        num_elements);

    return cudaGetLastError() == cudaSuccess;
}

__global__ void vector_add_kernel(
    const culindblad::Complex* a,
    const culindblad::Complex* b,
    culindblad::Complex* out,
    culindblad::Index size)
{
    const culindblad::Index idx =
        static_cast<culindblad::Index>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    const cuDoubleComplex a_val =
        reinterpret_cast<const cuDoubleComplex*>(a)[idx];
    const cuDoubleComplex b_val =
        reinterpret_cast<const cuDoubleComplex*>(b)[idx];

    reinterpret_cast<cuDoubleComplex*>(out)[idx] =
        cuCadd(a_val, b_val);
}

bool launch_vector_add_kernel(
    const void* d_a,
    const void* d_b,
    void* d_out,
    Index size,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((size + block_size - 1) / block_size);

    vector_add_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_a),
        reinterpret_cast<const Complex*>(d_b),
        reinterpret_cast<Complex*>(d_out),
        size);

    return cudaGetLastError() == cudaSuccess;
}

} // namespace culindblad