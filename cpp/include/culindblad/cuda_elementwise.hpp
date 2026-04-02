#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "culindblad/types.hpp"

namespace culindblad {

bool launch_commutator_combine_kernel(
    const void* d_left,
    const void* d_right,
    void* d_out,
    std::size_t num_elements,
    cudaStream_t stream);

bool launch_commutator_scaled_accumulate_kernel(
    const void* d_left,
    const void* d_right,
    double scale,
    void* d_accum,
    std::size_t num_elements,
    cudaStream_t stream);

bool launch_dissipator_combine_kernel(
    const void* d_jump,
    const void* d_left,
    const void* d_right,
    void* d_out,
    std::size_t num_elements,
    cudaStream_t stream);

bool launch_dissipator_scaled_accumulate_kernel(
    const void* d_jump,
    const void* d_left,
    const void* d_right,
    double scale,
    void* d_accum,
    std::size_t num_elements,
    cudaStream_t stream);

bool launch_vector_add_kernel(
    const void* d_a,
    const void* d_b,
    void* d_out,
    Index size,
    cudaStream_t stream);

bool launch_vector_accumulate_kernel(
    const void* d_src,
    void* d_dst,
    Index size,
    cudaStream_t stream);
    
bool launch_vector_scale_kernel(
    const void* d_in,
    double scale,
    void* d_out,
    Index size,
    cudaStream_t stream);  

bool launch_vector_scaled_accumulate_kernel(
    const void* d_src,
    double scale,
    void* d_dst,
    Index size,
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
