#pragma once

#include <petscvec.h>

#include <string>

#include "culindblad/cuda_grouped_layout.hpp"
#include "culindblad/cutensor_executor_cache.hpp"
#include "culindblad/grouped_state_layout.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

PetscErrorCode apply_grouped_left_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    Index batch_size = 1,
    cudaStream_t consumer_stream = nullptr);

PetscErrorCode apply_grouped_right_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    Index batch_size = 1,
    cudaStream_t consumer_stream = nullptr);

PetscErrorCode apply_grouped_commutator_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    Index batch_size = 1,
    cudaStream_t consumer_stream = nullptr);

PetscErrorCode apply_grouped_dissipator_cuda_vec(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    const CudaGroupedStateLayout& cuda_grouped_layout,
    CuTensorExecutorCache& executor_cache,
    Vec x,
    Vec y,
    Index batch_size = 1,
    cudaStream_t consumer_stream = nullptr);

PetscErrorCode regroup_petsc_cuda_vec_to_grouped_buffer(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Vec x,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t stream);

// State-vector (|psi>, D-sized) regroup + one-sided left actions for the Schrodinger solver.
PetscErrorCode regroup_petsc_cuda_vec_to_psi_grouped_buffer(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    Vec x,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode apply_grouped_sparse_left_to_psi_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* sparse_values,
    const void* sparse_rows,
    const void* sparse_cols,
    Index sparse_nnz,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode apply_grouped_diagonal_left_to_psi_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* h_diag,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode apply_grouped_sparse_left_csr_to_psi_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* sparse_values,
    const void* sparse_cols,
    const void* sparse_row_ptr,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode apply_psi_fused_diagonal_to_petsc_vec(
    const void* site_diags,
    const void* site_desc,
    Index num_folded,
    Vec x,
    Vec y,
    Index hilbert_dim,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode apply_rho_fused_diagonal_to_petsc_vec(
    const void* site_factors,
    const void* site_desc,
    Index num_folded,
    Vec x,
    Vec y,
    Index hilbert_dim,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode flatten_grouped_buffer_to_petsc_cuda_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaStream_t producer_stream);

PetscErrorCode flatten_grouped_buffer_accumulate_into_petsc_cuda_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t stream);

// Fused 2-site sparse commutator applied straight into the flat RHS vector y (accumulating),
// skipping the grouped accumulator + the separate flatten. Sole-contributor coupling layers.
PetscErrorCode apply_grouped_sparse_commutator_to_petsc_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* sparse_values,
    const void* sparse_rows,
    const void* sparse_cols,
    Index sparse_nnz,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t stream);

// Single-site fused diagonal / dense jump applied straight into the flat RHS vector y.
PetscErrorCode apply_grouped_fused_diagonal_to_petsc_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* h_diag,
    const void* norm_diag,
    const void* jump_diag,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode apply_grouped_dense_jump_to_petsc_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* l_op,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t stream);

PetscErrorCode apply_grouped_commutator_cuda_buffer(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event = nullptr,
    cudaEvent_t output_ready_event = nullptr);

PetscErrorCode apply_grouped_dissipator_cuda_buffer(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Complex>& local_op_dag_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event = nullptr,
    cudaEvent_t output_ready_event = nullptr);

PetscErrorCode apply_grouped_dissipator_jump_cuda_buffer(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& local_op_dag,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event = nullptr,
    cudaEvent_t output_ready_event = nullptr);

PetscErrorCode apply_grouped_anti_commutator_cuda_buffer(
    const Solver& solver,
    const std::string& term_label,
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const GroupedStateLayout& grouped_layout,
    CuTensorExecutorCache& executor_cache,
    const void* d_grouped_input,
    void* d_grouped_output,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaEvent_t input_ready_event = nullptr,
    cudaEvent_t output_ready_event = nullptr);

} // namespace culindblad
