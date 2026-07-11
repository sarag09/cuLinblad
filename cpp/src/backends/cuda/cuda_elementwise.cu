#include "culindblad/cuda_elementwise.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "culindblad/types.hpp"
#include "culindblad/cuda_complex_compat.cuh"

namespace culindblad {

namespace {

static_assert(sizeof(Complex) == sizeof(DevComplex),
              "Complex must match DevComplex layout");

__global__ void commutator_combine_kernel(
    const DevComplex* left,
    const DevComplex* right,
    DevComplex* out,
    std::size_t n)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const DevComplex minus_i = dev_mkc(0.0, -1.0);
    out[idx] = dev_mul(minus_i, dev_sub(left[idx], right[idx]));
}

__global__ void dissipator_combine_kernel(
    const DevComplex* jump,
    const DevComplex* left,
    const DevComplex* right,
    DevComplex* out,
    std::size_t n)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const DevComplex half = dev_mkc(0.5, 0.0);
    out[idx] = dev_sub(
        jump[idx],
        dev_mul(half, dev_add(left[idx], right[idx])));
}

__global__ void anti_commutator_combine_kernel(
    const DevComplex* left,
    const DevComplex* right,
    DevComplex* out,
    std::size_t n)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const DevComplex minus_half = dev_mkc(-0.5, 0.0);
    out[idx] = dev_mul(minus_half, dev_add(left[idx], right[idx]));
}

__global__ void batched_grouped_sparse_commutator_kernel(
    const DevComplex* sparse_values,
    const Index* sparse_rows,
    const Index* sparse_cols,
    Index sparse_nnz,
    const DevComplex* grouped_input,
    DevComplex* grouped_output,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size)
{
    const std::size_t total_elements =
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    const std::size_t grouped_size =
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim);
    std::size_t local_idx = idx % grouped_size;

    const Index bra_comp = static_cast<Index>(local_idx % complement_dim);
    local_idx /= complement_dim;
    const Index bra_target_out = static_cast<Index>(local_idx % target_hilbert_dim);
    local_idx /= target_hilbert_dim;
    const Index ket_comp = static_cast<Index>(local_idx % complement_dim);
    local_idx /= complement_dim;
    const Index ket_target_out = static_cast<Index>(local_idx);

    DevComplex left_sum = dev_mkc(0.0, 0.0);
    DevComplex right_sum = dev_mkc(0.0, 0.0);

    const std::size_t batch_offset = (idx / grouped_size) * grouped_size;
    const std::size_t left_base =
        batch_offset +
        (static_cast<std::size_t>(ket_comp) * target_hilbert_dim + bra_target_out) *
            complement_dim +
        bra_comp;
    const std::size_t right_base =
        batch_offset +
        (static_cast<std::size_t>(ket_target_out) * complement_dim + ket_comp) *
            target_hilbert_dim * complement_dim +
        bra_comp;
    const std::size_t left_stride =
        static_cast<std::size_t>(complement_dim) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim);
    const std::size_t right_stride = complement_dim;

    for (Index nz = 0; nz < sparse_nnz; ++nz) {
        const Index row = sparse_rows[nz];
        const Index col = sparse_cols[nz];
        const DevComplex value = sparse_values[nz];

        if (row == ket_target_out) {
            left_sum = dev_add(
                left_sum,
                dev_mul(
                    value,
                    grouped_input[left_base + static_cast<std::size_t>(col) * left_stride]));
        }

        if (col == bra_target_out) {
            right_sum = dev_add(
                right_sum,
                dev_mul(
                    grouped_input[right_base + static_cast<std::size_t>(row) * right_stride],
                    value));
        }
    }

    const DevComplex minus_i = dev_mkc(0.0, -1.0);
    // The commutator is the first term applied on every layer and writes every output
    // element, so it OVERWRITES the accumulator (the per-layer zeroing is skipped in the
    // RHS whenever a commutator is present). 0 += X is identical to = X, so bit-for-bit.
    grouped_output[idx] = dev_mul(minus_i, dev_sub(left_sum, right_sum));
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
        reinterpret_cast<const DevComplex*>(d_left),
        reinterpret_cast<const DevComplex*>(d_right),
        reinterpret_cast<DevComplex*>(d_out),
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
        reinterpret_cast<const DevComplex*>(d_jump),
        reinterpret_cast<const DevComplex*>(d_left),
        reinterpret_cast<const DevComplex*>(d_right),
        reinterpret_cast<DevComplex*>(d_out),
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
        reinterpret_cast<const DevComplex*>(d_left),
        reinterpret_cast<const DevComplex*>(d_right),
        reinterpret_cast<DevComplex*>(d_out),
        num_elements);

    return cudaGetLastError() == cudaSuccess;
}

bool launch_batched_grouped_sparse_commutator_kernel(
    const void* sparse_values,
    const Index* sparse_rows,
    const Index* sparse_cols,
    Index sparse_nnz,
    const void* grouped_input,
    void* grouped_output,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    cudaStream_t stream)
{
    if (sparse_values == nullptr || sparse_rows == nullptr || sparse_cols == nullptr ||
        grouped_input == nullptr || grouped_output == nullptr ||
        sparse_nnz <= 0 || target_hilbert_dim <= 0 || complement_dim <= 0 || batch_size <= 0) {
        return false;
    }

    constexpr int threads_per_block = 256;
    const std::size_t total_elements =
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim) *
        static_cast<std::size_t>(target_hilbert_dim) *
        static_cast<std::size_t>(complement_dim);
    const int blocks =
        static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block);

    batched_grouped_sparse_commutator_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DevComplex*>(sparse_values),
        sparse_rows,
        sparse_cols,
        sparse_nnz,
        static_cast<const DevComplex*>(grouped_input),
        static_cast<DevComplex*>(grouped_output),
        target_hilbert_dim,
        complement_dim,
        batch_size);

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

    const DevComplex a_val =
        reinterpret_cast<const DevComplex*>(a)[idx];
    const DevComplex b_val =
        reinterpret_cast<const DevComplex*>(b)[idx];

    reinterpret_cast<DevComplex*>(out)[idx] =
        dev_add(a_val, b_val);
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

    const DevComplex src_val =
        reinterpret_cast<const DevComplex*>(src)[idx];
    const DevComplex dst_val =
        reinterpret_cast<const DevComplex*>(dst)[idx];

    reinterpret_cast<DevComplex*>(dst)[idx] =
        dev_add(dst_val, src_val);
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

    const DevComplex in_val =
        reinterpret_cast<const DevComplex*>(in)[idx];
    const DevComplex s =
        dev_mkc(scale, 0.0);

    reinterpret_cast<DevComplex*>(out)[idx] =
        dev_mul(s, in_val);
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

    const DevComplex src_val =
        reinterpret_cast<const DevComplex*>(src)[idx];
    const DevComplex dst_val =
        reinterpret_cast<const DevComplex*>(dst)[idx];
    const DevComplex s =
        dev_mkc(scale, 0.0);

    reinterpret_cast<DevComplex*>(dst)[idx] =
        dev_add(dst_val, dev_mul(s, src_val));
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
    DevComplex* buffer,
    std::size_t total_elements)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        buffer[idx] = dev_mkc(0.0, 0.0);
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
        static_cast<DevComplex*>(buffer),
        static_cast<std::size_t>(total_elements));

    return cudaGetLastError() == cudaSuccess;
}

__global__ void batched_grouped_left_diagonal_kernel(
    const DevComplex* diagonal_op,
    const DevComplex* grouped_input,
    DevComplex* grouped_output,
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

    const DevComplex op_val = diagonal_op[ket_target];
    const DevComplex rho_val = grouped_input[idx];

    grouped_output[idx] = dev_mul(op_val, rho_val);
}

__global__ void batched_grouped_diagonal_dissipator_jump_kernel(
    const DevComplex* diagonal_op,
    const DevComplex* grouped_input,
    DevComplex* grouped_output,
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

    const DevComplex ket_factor = diagonal_op[ket_target];
    const DevComplex bra_factor = dev_conj(diagonal_op[bra_target]);
    const DevComplex rho_val = grouped_input[idx];

    grouped_output[idx] = dev_add(
        grouped_output[idx],
        dev_mul(dev_mul(ket_factor, rho_val), bra_factor));
}

__global__ void batched_grouped_diagonal_commutator_kernel(
    const DevComplex* diagonal_op,
    const DevComplex* grouped_input,
    DevComplex* grouped_output,
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

    const DevComplex ket_factor = diagonal_op[ket_target];
    const DevComplex bra_factor = diagonal_op[bra_target];
    const DevComplex rho_val = grouped_input[idx];
    const DevComplex minus_i = dev_mkc(0.0, -1.0);

    // Commutator overwrites the accumulator (first term per layer; zero skipped in the RHS).
    grouped_output[idx] = dev_mul(
        minus_i,
        dev_mul(
            dev_sub(ket_factor, bra_factor),
            rho_val));
}

// Anti-commutator -1/2 {D, rho} for a DIAGONAL operator D (= sum_k L_k^dag L_k):
// (-1/2){D,rho}_rc = -1/2 (D_r + D_c) rho_rc. Drop-in for the general cuTENSOR
// anti-commutator (same -1/2 (left+right) convention as anti_commutator_combine_kernel).
__global__ void batched_grouped_diagonal_anti_commutator_kernel(
    const DevComplex* diagonal_op,
    const DevComplex* grouped_input,
    DevComplex* grouped_output,
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

    const DevComplex ket_factor = diagonal_op[ket_target];
    const DevComplex bra_factor = diagonal_op[bra_target];
    const DevComplex rho_val = grouped_input[idx];
    const DevComplex minus_half = dev_mkc(-0.5, 0.0);

    grouped_output[idx] = dev_add(
        grouped_output[idx],
        dev_mul(
            minus_half,
            dev_mul(
                dev_add(ket_factor, bra_factor),
                rho_val)));
}

// Dissipator jump L rho L^dag for a (small) dense single-site collapse operator L that
// acts only on the target block (e.g. relaxation L = sqrt(1/T1) a):
//   (L rho L^dag)[i,ic,j,jc] = sum_{k,l} L[i,k] rho[k,ic,l,jc] conj(L[j,l]).
// The complement indices (ic, jc) pass through unchanged. Drop-in for the general
// cuTENSOR jump path; cheap when target_hilbert_dim is small (single-site qutrit = 3).
__global__ void batched_grouped_dense_jump_kernel(
    const DevComplex* l_op,
    const DevComplex* grouped_input,
    DevComplex* grouped_output,
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

    const std::size_t batch_idx = idx / per_state_size;
    const std::size_t state_offset = idx % per_state_size;

    const std::size_t bra_comp = state_offset % complement_dim;
    const std::size_t tmp0 = state_offset / complement_dim;

    const std::size_t bra_target = tmp0 % target_hilbert_dim;
    const std::size_t tmp1 = tmp0 / target_hilbert_dim;

    const std::size_t ket_comp = tmp1 % complement_dim;
    const std::size_t ket_target = tmp1 / complement_dim;

    const std::size_t base = batch_idx * per_state_size;
    DevComplex acc = dev_mkc(0.0, 0.0);

    for (std::size_t k = 0; k < target_hilbert_dim; ++k) {
        const DevComplex l_ik = l_op[ket_target * target_hilbert_dim + k];
        if (l_ik.x == 0.0 && l_ik.y == 0.0) {
            continue;
        }
        for (std::size_t l = 0; l < target_hilbert_dim; ++l) {
            const DevComplex l_jl = l_op[bra_target * target_hilbert_dim + l];
            if (l_jl.x == 0.0 && l_jl.y == 0.0) {
                continue;
            }
            const std::size_t rho_idx =
                base +
                ((k * complement_dim + ket_comp) * target_hilbert_dim + l) * complement_dim +
                bra_comp;
            acc = dev_add(
                acc,
                dev_mul(dev_mul(l_ik, grouped_input[rho_idx]), dev_conj(l_jl)));
        }
    }

    grouped_output[idx] = dev_add(grouped_output[idx], acc);
}

// Fused single-site DIAGONAL Liouvillian: combines the commutator, the anti-commutator,
// and one diagonal dissipator jump into a single elementwise pass that OVERWRITES the
// accumulator. factor = -i(H_ket - H_bra) - 1/2(N_ket + N_bra) + J_ket*conj(J_bra); any of
// h_diag/norm_diag/jump_diag == nullptr skips that term. Mathematically identical to
// applying the three diagonal kernels in sequence (only the float summation order differs,
// so rounding-level not bit-identical). The non-diagonal relaxation jump is applied after.
__global__ void batched_grouped_fused_diagonal_kernel(
    const DevComplex* h_diag,
    const DevComplex* norm_diag,
    const DevComplex* jump_diag,
    const DevComplex* grouped_input,
    DevComplex* grouped_output,
    std::size_t target_hilbert_dim,
    std::size_t complement_dim,
    std::size_t batch_size)
{
    const std::size_t per_state_size =
        target_hilbert_dim * complement_dim * target_hilbert_dim * complement_dim;
    const std::size_t total_elements = per_state_size * batch_size;
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

    DevComplex factor = dev_mkc(0.0, 0.0);
    if (h_diag != nullptr) {
        const DevComplex minus_i = dev_mkc(0.0, -1.0);
        factor = dev_add(
            factor,
            dev_mul(minus_i, dev_sub(h_diag[ket_target], h_diag[bra_target])));
    }
    if (norm_diag != nullptr) {
        const DevComplex minus_half = dev_mkc(-0.5, 0.0);
        factor = dev_add(
            factor,
            dev_mul(minus_half, dev_add(norm_diag[ket_target], norm_diag[bra_target])));
    }
    if (jump_diag != nullptr) {
        factor = dev_add(
            factor,
            dev_mul(jump_diag[ket_target], dev_conj(jump_diag[bra_target])));
    }
    grouped_output[idx] = dev_mul(factor, grouped_input[idx]);
}

bool launch_batched_grouped_fused_diagonal_kernel(
    const void* h_diag,
    const void* norm_diag,
    const void* jump_diag,
    const void* grouped_input,
    void* grouped_output,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    cudaStream_t stream)
{
    if (grouped_input == nullptr || grouped_output == nullptr ||
        target_hilbert_dim <= 0 || complement_dim <= 0 || batch_size <= 0) {
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
    batched_grouped_fused_diagonal_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DevComplex*>(h_diag),
        static_cast<const DevComplex*>(norm_diag),
        static_cast<const DevComplex*>(jump_diag),
        static_cast<const DevComplex*>(grouped_input),
        static_cast<DevComplex*>(grouped_output),
        target_hilbert_dim,
        complement_dim,
        batch_size);
    return cudaGetLastError() == cudaSuccess;
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
        static_cast<const DevComplex*>(diagonal_op),
        static_cast<const DevComplex*>(grouped_input),
        static_cast<DevComplex*>(grouped_output),
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
        static_cast<const DevComplex*>(diagonal_op),
        static_cast<const DevComplex*>(grouped_input),
        static_cast<DevComplex*>(grouped_output),
        static_cast<std::size_t>(target_hilbert_dim),
        static_cast<std::size_t>(complement_dim),
        static_cast<std::size_t>(batch_size));

    return cudaGetLastError() == cudaSuccess;
}

bool launch_batched_grouped_diagonal_commutator_kernel(
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

    batched_grouped_diagonal_commutator_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DevComplex*>(diagonal_op),
        static_cast<const DevComplex*>(grouped_input),
        static_cast<DevComplex*>(grouped_output),
        static_cast<std::size_t>(target_hilbert_dim),
        static_cast<std::size_t>(complement_dim),
        static_cast<std::size_t>(batch_size));

    return cudaGetLastError() == cudaSuccess;
}

bool launch_batched_grouped_diagonal_anti_commutator_kernel(
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

    batched_grouped_diagonal_anti_commutator_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DevComplex*>(diagonal_op),
        static_cast<const DevComplex*>(grouped_input),
        static_cast<DevComplex*>(grouped_output),
        static_cast<std::size_t>(target_hilbert_dim),
        static_cast<std::size_t>(complement_dim),
        static_cast<std::size_t>(batch_size));

    return cudaGetLastError() == cudaSuccess;
}

bool launch_batched_grouped_dense_jump_kernel(
    const void* l_op,
    const void* grouped_input,
    void* grouped_output,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    cudaStream_t stream)
{
    if (l_op == nullptr || grouped_input == nullptr || grouped_output == nullptr) {
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

    batched_grouped_dense_jump_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DevComplex*>(l_op),
        static_cast<const DevComplex*>(grouped_input),
        static_cast<DevComplex*>(grouped_output),
        static_cast<std::size_t>(target_hilbert_dim),
        static_cast<std::size_t>(complement_dim),
        static_cast<std::size_t>(batch_size));

    return cudaGetLastError() == cudaSuccess;
}

// ---------------------------------------------------------------------------
// fp32 open-system stabilizer: per-density-matrix TRACE RENORMALIZATION.
// After each RK4 step, physical density matrices (Re Tr ~ 1) are rescaled rho <- rho / Re(Tr rho) to
// cancel the coherent fp32 trace drift (the dominant single-precision open error). The dressed channel
// basis also carries OFF-DIAGONAL inputs |D_a><D_b| (a!=b) whose trace is ~0 and must NOT be rescaled;
// the (trace > 0.5) guard renormalizes ONLY the trace~1 physical matrices (matches the Julia
// characterization's `renorm && a==b`). Trace summed in double for accuracy. One block per matrix.
__global__ void density_trace_renorm_kernel(
    culindblad::Complex* state,
    culindblad::Index batch,
    culindblad::Index D)
{
    const culindblad::Index b = static_cast<culindblad::Index>(blockIdx.x);
    if (b >= batch) {
        return;
    }
    DevComplex* rho = reinterpret_cast<DevComplex*>(state) + b * D * D;

    __shared__ double sdata[256];
    double partial = 0.0;
    for (culindblad::Index i = threadIdx.x; i < D; i += blockDim.x) {
        partial += static_cast<double>(rho[i * D + i].x);   // real part of the diagonal
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    const double trace = sdata[0];
    if (trace <= 0.5) {
        return;   // off-diagonal channel input (Tr ~ 0) or degenerate -> leave unchanged
    }
    const double inv = 1.0 / trace;
    for (culindblad::Index idx = threadIdx.x; idx < D * D; idx += blockDim.x) {
        rho[idx].x *= inv;
        rho[idx].y *= inv;
    }
}

bool launch_density_trace_renorm_kernel(
    void* d_state,
    Index batch,
    Index D,
    cudaStream_t stream)
{
    if (batch == 0 || D == 0) {
        return true;
    }
    constexpr int block_size = 256;   // matches sdata[256]
    density_trace_renorm_kernel<<<static_cast<int>(batch), block_size, 0, stream>>>(
        reinterpret_cast<Complex*>(d_state), batch, D);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace culindblad
