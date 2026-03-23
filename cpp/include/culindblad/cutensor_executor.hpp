#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_plan.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct CuTensorExecutor {
    CuTensorContractionDesc desc;
    CuTensorPlanBundle plan_bundle;
    cudaStream_t stream;

    void* d_op;
    void* d_input;
    void* d_output;
    void* d_workspace;

    size_t op_bytes;
    size_t input_bytes;
    size_t output_bytes;
};

bool create_cutensor_executor(
    const CuTensorContractionDesc& desc,
    size_t op_bytes,
    size_t input_bytes,
    size_t output_bytes,
    CuTensorExecutor& executor);

bool destroy_cutensor_executor(
    CuTensorExecutor& executor);

bool upload_cutensor_executor_operator(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op);

bool upload_cutensor_executor_input(
    CuTensorExecutor& executor,
    const std::vector<Complex>& input_tensor);

bool upload_cutensor_executor_inputs(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor);

bool copy_cutensor_executor_output_to_input(
    CuTensorExecutor& src_executor,
    CuTensorExecutor& dst_executor);

bool execute_cutensor_executor_device(
    CuTensorExecutor& executor);

bool download_cutensor_executor_output(
    CuTensorExecutor& executor,
    std::vector<Complex>& output_tensor);

bool execute_cutensor_executor(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor);

} // namespace culindblad