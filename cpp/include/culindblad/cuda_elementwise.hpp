#pragma once

#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include "culindblad/types.hpp"

namespace culindblad {

constexpr Index kMaxTinyDenseHilbertDim = 9;
constexpr Index kMaxTinyDenseOperatorElements =
    kMaxTinyDenseHilbertDim * kMaxTinyDenseHilbertDim;

struct TinyDenseOperatorKernelArg {
    Index dim;
    Complex data[kMaxTinyDenseOperatorElements];
};

bool launch_commutator_combine_kernel(
    const void* d_left,
    const void* d_right,
    void* d_out,
    std::size_t num_elements,
    cudaStream_t stream);

bool launch_dissipator_combine_kernel(
    const void* d_jump,
    const void* d_left,
    const void* d_right,
    void* d_out,
    std::size_t num_elements,
    cudaStream_t stream);

bool launch_vector_add_kernel(
    const void* d_a,
    const void* d_b,
    void* d_out,
    Index size,
    cudaStream_t stream);
    
bool launch_vector_scale_kernel(
    const void* d_in,
    double scale,
    void* d_out,
    Index size,
    cudaStream_t stream);  

bool launch_tiny_dense_commutator_kernel(
    const TinyDenseOperatorKernelArg& op,
    const void* d_rho,
    void* d_out,
    cudaStream_t stream);

bool launch_batched_tiny_dense_commutator_kernel(
    const TinyDenseOperatorKernelArg& op,
    const void* d_rho,
    void* d_out,
    Index batch_size,
    cudaStream_t stream);

bool launch_tiny_dense_dissipator_kernel(
    const TinyDenseOperatorKernelArg& jump_op,
    const TinyDenseOperatorKernelArg& norm_op,
    const void* d_rho,
    void* d_out,
    cudaStream_t stream);

bool launch_batched_tiny_dense_dissipator_kernel(
    const TinyDenseOperatorKernelArg& jump_op,
    const TinyDenseOperatorKernelArg& norm_op,
    const void* d_rho,
    void* d_out,
    Index batch_size,
    cudaStream_t stream);
  
bool launch_zero_batched_buffer_kernel(
    void* buffer,
    Index total_elements,
    cudaStream_t stream);    

bool launch_batched_grouped_left_diagonal_kernel(
    const void* diagonal_op,
    const void* grouped_input,
    void* grouped_output,
    Index target_hilbert_dim,
    Index complement_dim,
    Index batch_size,
    cudaStream_t stream);

} // namespace culindblad
