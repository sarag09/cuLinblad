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

} // namespace culindblad
