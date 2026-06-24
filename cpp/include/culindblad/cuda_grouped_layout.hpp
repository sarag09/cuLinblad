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

// State-vector (ket-only) layout: same descriptors as the density layout, but flat_size =
// grouped_size = hilbert_dim (D), not D^2. Use the *_vec_* launchers below with this layout.
bool create_cuda_psi_grouped_state_layout(
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

// State-vector (ket-only, D-sized) relayout. Use with a layout from
// create_cuda_psi_grouped_state_layout; flat IS ket_full, grouped is g = ket_t*comp + ket_c.
bool launch_flat_to_grouped_vec_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_flat_input,
    void* d_grouped_output,
    cudaStream_t stream);

bool launch_grouped_to_flat_vec_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

// One-sided left action -i*H|psi> accumulated into the flat RHS f (state-vector solver).
// sparse: any k-site sparse term; diagonal: a single-site diagonal term (h_diag over the target).
bool launch_grouped_sparse_left_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* sparse_values,
    const void* sparse_rows,
    const void* sparse_cols,
    Index sparse_nnz,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

// CSR variant of the sparse left action (each thread visits only its row's nonzeros via
// sparse_row_ptr; bit-identical, far fewer instructions on the gather-bound kernel).
bool launch_grouped_sparse_left_csr_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* sparse_values,
    const void* sparse_cols,
    const void* sparse_row_ptr,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

bool launch_grouped_diagonal_left_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* h_diag,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

// Fused single-site diagonal: f[b*D+k] += -i*(sum_s site_diags[off_s+(k/stride_s)%dim_s])*psi,
// building the diagonal on the GPU from the tiny per-site diagonals + descriptor (no host
// D-loop). No layout / no regroup.
bool launch_psi_fused_diagonal_kernel(
    const void* site_diags,
    const void* site_desc,
    Index num_folded,
    const void* psi_flat,
    void* flat_output,
    Index hilbert_dim,
    Index batch_size,
    cudaStream_t stream);

// Open-system fused single-site diagonal: f[idx] += (sum_s F_s[ket_s,bra_s]) * rho[idx] over the
// full flat density vector (F_s = per-site dim x dim Liouvillian-diagonal factor, -i baked in).
bool launch_rho_fused_diagonal_kernel(
    const void* site_factors,
    const void* site_desc,
    Index num_folded,
    const void* rho_flat,
    void* flat_output,
    Index hilbert_dim,
    Index batch_size,
    cudaStream_t stream);

// Accumulating variant: flat_output[f] += grouped_input[g] (flat_output not overwritten).
bool launch_grouped_to_flat_accumulate_batched_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

// Fused 2-site sparse commutator -i[H,rho] accumulated straight into the flat output (f),
// skipping the grouped accumulator + the separate flatten pass. For sole-contributor
// coupling layers only.
bool launch_grouped_sparse_commutator_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* sparse_values,
    const void* sparse_rows,
    const void* sparse_cols,
    Index sparse_nnz,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

// Single-site fused diagonal / dense jump accumulated straight into the flat output f.
bool launch_grouped_fused_diagonal_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* h_diag,
    const void* norm_diag,
    const void* jump_diag,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

bool launch_grouped_dense_jump_to_flat_kernel(
    const CudaGroupedStateLayout& cuda_layout,
    const void* l_op,
    Index batch_size,
    const void* d_grouped_input,
    void* d_flat_output,
    cudaStream_t stream);

} // namespace culindblad
