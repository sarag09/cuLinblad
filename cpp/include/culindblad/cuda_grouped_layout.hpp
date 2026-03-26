#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct CudaGroupedStateLayout {
    Index flat_size;
    Index grouped_size;

    Index* d_flat_to_grouped;
    Index* d_grouped_to_flat;
};

bool create_cuda_grouped_state_layout(
    const GroupedStateLayout& host_layout,
    CudaGroupedStateLayout& cuda_layout);

bool destroy_cuda_grouped_state_layout(
    CudaGroupedStateLayout& cuda_layout);

bool launch_flat_to_grouped_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* d_flat_input,
    void* d_grouped_output,
    cudaStream_t stream);

bool launch_grouped_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

} // namespace culindblad
