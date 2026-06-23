#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct CudaGroupedStateLayout {
    Index flat_size;     // density_dim = D*D
    Index grouped_size;  // = flat_size (same element count, reordered)

    // The flat<->grouped permutation is a deterministic function of the target sites and
    // mixed-radix strides, so it is computed arithmetically in the relayout kernels from
    // this small descriptor instead of being stored as two density_dim-sized index tables
    // (which cost ~688 MB/layout at D=6561, i.e. tens of GB across an all-to-all model).
    Index hilbert_dim;      // D
    Index target_dim;       // product of target-site dims
    Index complement_dim;   // product of complement-site dims
    Index num_target;       // number of target sites
    Index num_complement;   // number of complement sites
    // Packed per-position descriptors, length 3*num_target / 3*num_complement; each
    // position contributes [full_hilbert_stride, local_dim, packed_weight].
    Index* d_target_desc;
    Index* d_complement_desc;
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

bool launch_flat_to_grouped_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_flat_input,
    void* d_grouped_output,
    cudaStream_t stream);

bool launch_grouped_to_flat_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

// Accumulating variant: flat_output[f] += grouped_input[g] (flat_output not overwritten).
bool launch_grouped_to_flat_accumulate_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

} // namespace culindblad
