#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_plan.hpp"
#include "culindblad/pinned_host_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct CuTensorExecutor {
    CuTensorContractionDesc desc;
    CuTensorPlanBundle plan_bundle;

    cudaStream_t stream;
    cudaEvent_t completion_event;
    cudaStream_t completion_stream;
    bool completion_recorded;

    void* d_op;
    void* d_input;
    void* d_output;
    void* d_workspace;

    std::size_t op_bytes;
    std::size_t input_bytes;
    std::size_t output_bytes;

    bool operator_resident;
    std::string resident_operator_tag;
};

bool create_cutensor_executor(
    const CuTensorContractionDesc& desc,
    size_t op_bytes,
    size_t input_bytes,
    size_t output_bytes,
    CuTensorExecutor& executor);

bool destroy_cutensor_executor(
    CuTensorExecutor& executor);

bool release_cutensor_executor_device_buffers(
    CuTensorExecutor& executor);

bool upload_cutensor_executor_operator(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op);

bool upload_cutensor_executor_operator_on_stream(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    cudaStream_t stream);

bool ensure_cutensor_executor_operator(
    CuTensorExecutor& executor,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op);

bool ensure_cutensor_executor_operator_on_stream(
    CuTensorExecutor& executor,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op,
    cudaStream_t stream);

bool upload_cutensor_executor_input(
    CuTensorExecutor& executor,
    const std::vector<Complex>& input_tensor);

bool upload_cutensor_executor_input_on_stream(
    CuTensorExecutor& executor,
    const std::vector<Complex>& input_tensor,
    cudaStream_t stream);

bool upload_cutensor_executor_inputs(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor);

bool copy_cutensor_executor_output_to_input(
    CuTensorExecutor& src_executor,
    CuTensorExecutor& dst_executor);

bool execute_cutensor_executor_device(
    CuTensorExecutor& executor);

bool execute_cutensor_executor_device_on_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream);

bool execute_cutensor_executor_device_no_completion(
    CuTensorExecutor& executor);

bool execute_cutensor_executor_device_no_completion_on_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream);

bool download_cutensor_executor_output(
    CuTensorExecutor& executor,
    std::vector<Complex>& output_tensor);

bool execute_cutensor_executor(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor);

bool execute_cutensor_executor_with_resident_operator(
    CuTensorExecutor& executor,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor);

bool execute_cutensor_executor_with_resident_operator_pinned(
    CuTensorExecutor& executor,
    const PinnedComplexBuffer& input_buffer,
    PinnedComplexBuffer& output_buffer);

bool record_cutensor_executor_completion(
    CuTensorExecutor& executor);

bool record_cutensor_executor_completion_on_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream);

bool wait_for_cutensor_executor_completion(
    CuTensorExecutor& producer,
    cudaStream_t consumer_stream);

} // namespace culindblad
