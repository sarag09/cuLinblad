#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace culindblad {

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

} // namespace culindblad
