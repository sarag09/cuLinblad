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

PetscErrorCode flatten_grouped_buffer_to_petsc_cuda_vec(
    const CudaGroupedStateLayout& cuda_grouped_layout,
    const void* d_grouped_input,
    Vec y,
    Index batch_size,
    cudaStream_t consumer_stream,
    cudaStream_t producer_stream);

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
