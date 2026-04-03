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

__global__ void anti_commutator_combine_kernel(
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

    const cuDoubleComplex minus_half = make_cuDoubleComplex(-0.5, 0.0);
    out[idx] = cuCmul(minus_half, cuCadd(left[idx], right[idx]));
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

bool launch_anti_commutator_combine_kernel(
    const void* d_left,
    const void* d_right,
    void* d_out,
    std::size_t num_elements,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((num_elements + block_size - 1) / block_size);

    anti_commutator_combine_kernel<<<grid_size, block_size, 0, stream>>>(
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

__global__ void vector_accumulate_kernel(
    const culindblad::Complex* src,
    culindblad::Complex* dst,
    culindblad::Index size)
{
    const culindblad::Index idx =
        static_cast<culindblad::Index>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    const cuDoubleComplex src_val =
        reinterpret_cast<const cuDoubleComplex*>(src)[idx];
    const cuDoubleComplex dst_val =
        reinterpret_cast<const cuDoubleComplex*>(dst)[idx];

    reinterpret_cast<cuDoubleComplex*>(dst)[idx] =
        cuCadd(dst_val, src_val);
}

bool launch_vector_accumulate_kernel(
    const void* d_src,
    void* d_dst,
    Index size,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((size + block_size - 1) / block_size);

    vector_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_src),
        reinterpret_cast<Complex*>(d_dst),
        size);

    return cudaGetLastError() == cudaSuccess;
}

__global__ void vector_scale_kernel(
    const culindblad::Complex* in,
    double scale,
    culindblad::Complex* out,
    culindblad::Index size)
{
    const culindblad::Index idx =
        static_cast<culindblad::Index>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    const cuDoubleComplex in_val =
        reinterpret_cast<const cuDoubleComplex*>(in)[idx];
    const cuDoubleComplex s =
        make_cuDoubleComplex(scale, 0.0);

    reinterpret_cast<cuDoubleComplex*>(out)[idx] =
        cuCmul(s, in_val);
}

bool launch_vector_scale_kernel(
    const void* d_in,
    double scale,
    void* d_out,
    Index size,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((size + block_size - 1) / block_size);

    vector_scale_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_in),
        scale,
        reinterpret_cast<Complex*>(d_out),
        size);

    return cudaGetLastError() == cudaSuccess;
}

__global__ void vector_scaled_accumulate_kernel(
    const culindblad::Complex* src,
    double scale,
    culindblad::Complex* dst,
    culindblad::Index size)
{
    const culindblad::Index idx =
        static_cast<culindblad::Index>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    const cuDoubleComplex src_val =
        reinterpret_cast<const cuDoubleComplex*>(src)[idx];
    const cuDoubleComplex dst_val =
        reinterpret_cast<const cuDoubleComplex*>(dst)[idx];
    const cuDoubleComplex s =
        make_cuDoubleComplex(scale, 0.0);

    reinterpret_cast<cuDoubleComplex*>(dst)[idx] =
        cuCadd(dst_val, cuCmul(s, src_val));
}

bool launch_vector_scaled_accumulate_kernel(
    const void* d_src,
    double scale,
    void* d_dst,
    Index size,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size =
        static_cast<int>((size + block_size - 1) / block_size);

    vector_scaled_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const Complex*>(d_src),
        scale,
        reinterpret_cast<Complex*>(d_dst),
        size);

    return cudaGetLastError() == cudaSuccess;
}

__global__ void zero_batched_buffer_kernel(
    cuDoubleComplex* buffer,
    std::size_t total_elements)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        buffer[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

bool launch_zero_batched_buffer_kernel(
    void* buffer,
    Index total_elements,
    cudaStream_t stream)
{
    if (buffer == nullptr) {
        return false;
    }

    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block);

    zero_batched_buffer_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<cuDoubleComplex*>(buffer),
        static_cast<std::size_t>(total_elements));

    return cudaGetLastError() == cudaSuccess;
}

__global__ void batched_grouped_left_diagonal_kernel(
    const cuDoubleComplex* diagonal_op,
    const cuDoubleComplex* grouped_input,
    cuDoubleComplex* grouped_output,
    std::size_t target_hilbert_dim,
    std::size_t complement_dim,
    std::size_t batch_size)
{
    const std::size_t per_state_size =
        target_hilbert_dim * complement_dim * target_hilbert_dim * complement_dim;

    const std::size_t total_elements =
        per_state_size * batch_size;

    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    const std::size_t state_offset = idx % per_state_size;

    const std::size_t bra_comp = state_offset % complement_dim;
    const std::size_t tmp0 = state_offset / complement_dim;

    const std::size_t bra_target = tmp0 % target_hilbert_dim;
    const std::size_t tmp1 = tmp0 / target_hilbert_dim;

    const std::size_t ket_comp = tmp1 % complement_dim;
    const std::size_t ket_target = tmp1 / complement_dim;

    (void)bra_comp;
    (void)bra_target;
    (void)ket_comp;

    const cuDoubleComplex op_val = diagonal_op[ket_target];
    const cuDoubleComplex rho_val = grouped_input[idx];

    grouped_output[idx] = cuCmul(op_val, rho_val);
}

__global__ void batched_grouped_diagonal_dissipator_jump_kernel(
    const cuDoubleComplex* diagonal_op,
    const cuDoubleComplex* grouped_input,
    cuDoubleComplex* grouped_output,
    std::size_t target_hilbert_dim,
    std::size_t complement_dim,
    std::size_t batch_size)
{
    const std::size_t per_state_size =
        target_hilbert_dim * complement_dim * target_hilbert_dim * complement_dim;

    const std::size_t total_elements =
        per_state_size * batch_size;

    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    const std::size_t state_offset = idx % per_state_size;

    const std::size_t bra_comp = state_offset % complement_dim;
    const std::size_t tmp0 = state_offset / complement_dim;

    const std::size_t bra_target = tmp0 % target_hilbert_dim;
    const std::size_t tmp1 = tmp0 / target_hilbert_dim;

    const std::size_t ket_comp = tmp1 % complement_dim;
    const std::size_t ket_target = tmp1 / complement_dim;

    (void)bra_comp;
    (void)ket_comp;

    const cuDoubleComplex ket_factor = diagonal_op[ket_target];
    const cuDoubleComplex bra_factor = cuConj(diagonal_op[bra_target]);
    const cuDoubleComplex rho_val = grouped_input[idx];

    grouped_output[idx] = cuCmul(cuCmul(ket_factor, rho_val), bra_factor);
}

bool launch_batched_grouped_left_diagonal_kernel(
    const void* diagonal_op,
    const void* grouped_input,
    void* grouped_output,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    cudaStream_t stream)
{
    if (diagonal_op == nullptr || grouped_input == nullptr || grouped_output == nullptr) {
        return false;
    }

    const std::size_t per_state_size =
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim);

    const std::size_t total_elements =
        per_state_size * static_cast<std::size_t>(batch_size);

    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block);

    batched_grouped_left_diagonal_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const cuDoubleComplex*>(diagonal_op),
        static_cast<const cuDoubleComplex*>(grouped_input),
        static_cast<cuDoubleComplex*>(grouped_output),
        static_cast<std::size_t>(target_hilbert_dim),
        static_cast<std::size_t>(complement_dim),
        static_cast<std::size_t>(batch_size));

    return cudaGetLastError() == cudaSuccess;
}

bool launch_batched_grouped_diagonal_dissipator_jump_kernel(
    const void* diagonal_op,
    const void* grouped_input,
    void* grouped_output,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    cudaStream_t stream)
{
    if (diagonal_op == nullptr || grouped_input == nullptr || grouped_output == nullptr) {
        return false;
    }

    const std::size_t per_state_size =
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim);

    const std::size_t total_elements =
        per_state_size * static_cast<std::size_t>(batch_size);

    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block);

    batched_grouped_diagonal_dissipator_jump_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const cuDoubleComplex*>(diagonal_op),
        static_cast<const cuDoubleComplex*>(grouped_input),
        static_cast<cuDoubleComplex*>(grouped_output),
        static_cast<std::size_t>(target_hilbert_dim),
        static_cast<std::size_t>(complement_dim),
        static_cast<std::size_t>(batch_size));

    return cudaGetLastError() == cudaSuccess;
}

} // namespace culindblad
