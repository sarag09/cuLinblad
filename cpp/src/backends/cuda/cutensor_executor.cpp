#include "culindblad/cutensor_executor.hpp"

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "culindblad/cutensor_plan.hpp"
#include "culindblad/pinned_host_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

cudaStream_t resolve_execution_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream)
{
    return stream != nullptr ? stream : executor.stream;
}

bool prepare_cutensor_executor_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream)
{
    cudaStream_t execution_stream = resolve_execution_stream(executor, stream);
    if (execution_stream == nullptr) {
        return false;
    }

    if (!executor.completion_recorded ||
        executor.completion_stream == nullptr ||
        executor.completion_stream == execution_stream) {
        return true;
    }

    if (executor.completion_event == nullptr) {
        return false;
    }

    return cudaStreamWaitEvent(
               execution_stream,
               executor.completion_event,
               0) == cudaSuccess;
}

bool cuda_malloc_bytes(void** ptr, size_t bytes)
{
    return cudaMalloc(ptr, bytes) == cudaSuccess;
}

bool ensure_cuda_buffer(void** ptr, size_t bytes)
{
    if (bytes == 0) {
        *ptr = nullptr;
        return true;
    }

    if (*ptr != nullptr) {
        return true;
    }

    return cuda_malloc_bytes(ptr, bytes);
}

bool cuda_copy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream)
{
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

bool cuda_copy_d2h_async(void* dst, const void* src, size_t bytes, cudaStream_t stream)
{
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

bool cuda_copy_d2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream)
{
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
}

bool execute_cutensor_executor_device_impl(
    CuTensorExecutor& executor,
    cudaStream_t stream,
    bool record_completion)
{
    cudaStream_t execution_stream = resolve_execution_stream(executor, stream);
    if (!prepare_cutensor_executor_stream(executor, execution_stream)) {
        return false;
    }

    if (executor.d_input == nullptr) {
        return false;
    }

    if (!ensure_cuda_buffer(&executor.d_output, executor.output_bytes) ||
        !ensure_cuda_buffer(
            &executor.d_workspace,
            static_cast<size_t>(executor.plan_bundle.workspace_size))) {
        return false;
    }

    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};

    const cutensorStatus_t exec_status =
        cutensorContract(
            executor.plan_bundle.op_bundle.tensor_descs.handle,
            executor.plan_bundle.plan,
            reinterpret_cast<const void*>(&alpha),
            executor.d_op,
            executor.d_input,
            reinterpret_cast<const void*>(&beta),
            executor.d_output,
            executor.d_output,
            executor.d_workspace,
            executor.plan_bundle.workspace_size,
            execution_stream);

    if (exec_status != CUTENSOR_STATUS_SUCCESS) {
        return false;
    }

    if (record_completion &&
        !record_cutensor_executor_completion_on_stream(executor, execution_stream)) {
        return false;
    }

    return true;
}

bool finalize_cutensor_executor_preparation(
    CuTensorExecutor& executor,
    cudaStream_t stream)
{
    return record_cutensor_executor_completion_on_stream(executor, stream);
}

} // namespace

bool create_cutensor_executor(
    const CuTensorContractionDesc& desc,
    size_t op_bytes,
    size_t input_bytes,
    size_t output_bytes,
    CuTensorExecutor& executor)
{
    executor.desc = desc;
    executor.stream = nullptr;
    executor.completion_event = nullptr;
    executor.completion_stream = nullptr;
    executor.completion_recorded = false;
    executor.d_op = nullptr;
    executor.d_input = nullptr;
    executor.d_output = nullptr;
    executor.d_workspace = nullptr;
    executor.op_bytes = op_bytes;
    executor.input_bytes = input_bytes;
    executor.output_bytes = output_bytes;
    executor.operator_resident = false;
    executor.resident_operator_tag.clear();

    if (!create_cutensor_plan(desc, executor.plan_bundle)) {
        return false;
    }

    if (cudaStreamCreate(&executor.stream) != cudaSuccess) {
        destroy_cutensor_plan(executor.plan_bundle);
        return false;
    }

    if (cudaEventCreateWithFlags(&executor.completion_event, cudaEventDisableTiming) != cudaSuccess) {
        cudaStreamDestroy(executor.stream);
        destroy_cutensor_plan(executor.plan_bundle);
        return false;
    }

    if (!cuda_malloc_bytes(&executor.d_op, op_bytes)) {
        cudaStreamDestroy(executor.stream);
        destroy_cutensor_plan(executor.plan_bundle);
        return false;
    }

    return true;
}

bool destroy_cutensor_executor(
    CuTensorExecutor& executor)
{
    bool ok = true;

    if (executor.d_workspace != nullptr && cudaFree(executor.d_workspace) != cudaSuccess) {
        ok = false;
    }

    if (executor.d_output != nullptr && cudaFree(executor.d_output) != cudaSuccess) {
        ok = false;
    }

    if (executor.d_input != nullptr && cudaFree(executor.d_input) != cudaSuccess) {
        ok = false;
    }

    if (executor.d_op != nullptr && cudaFree(executor.d_op) != cudaSuccess) {
        ok = false;
    }

    if (executor.completion_event != nullptr &&
        cudaEventDestroy(executor.completion_event) != cudaSuccess) {
        ok = false;
    }

    if (executor.stream != nullptr && cudaStreamDestroy(executor.stream) != cudaSuccess) {
        ok = false;
    }

    if (!destroy_cutensor_plan(executor.plan_bundle)) {
        ok = false;
    }

    executor.operator_resident = false;
    executor.resident_operator_tag.clear();
    executor.completion_stream = nullptr;
    executor.completion_recorded = false;

    return ok;
}

bool release_cutensor_executor_device_buffers(
    CuTensorExecutor& executor)
{
    bool ok = true;

    if (executor.d_workspace != nullptr && cudaFree(executor.d_workspace) != cudaSuccess) {
        ok = false;
    }

    if (executor.d_output != nullptr && cudaFree(executor.d_output) != cudaSuccess) {
        ok = false;
    }

    if (executor.d_input != nullptr && cudaFree(executor.d_input) != cudaSuccess) {
        ok = false;
    }

    executor.d_workspace = nullptr;
    executor.d_output = nullptr;
    executor.d_input = nullptr;
    executor.completion_stream = nullptr;
    executor.completion_recorded = false;
    return ok;
}

bool upload_cutensor_executor_operator(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op)
{
    return upload_cutensor_executor_operator_on_stream(
        executor,
        local_op,
        executor.stream);
}

bool upload_cutensor_executor_operator_on_stream(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    cudaStream_t stream)
{
    if (local_op.size() * sizeof(Complex) != executor.op_bytes) {
        return false;
    }

    cudaStream_t execution_stream = resolve_execution_stream(executor, stream);
    if (!prepare_cutensor_executor_stream(executor, execution_stream)) {
        return false;
    }

    if (!cuda_copy_h2d_async(executor.d_op, local_op.data(), executor.op_bytes, execution_stream)) {
        return false;
    }

    executor.operator_resident = true;
    executor.resident_operator_tag.clear();
    return finalize_cutensor_executor_preparation(executor, execution_stream);
}

bool ensure_cutensor_executor_operator(
    CuTensorExecutor& executor,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op)
{
    return ensure_cutensor_executor_operator_on_stream(
        executor,
        operator_tag,
        local_op,
        executor.stream);
}

bool ensure_cutensor_executor_operator_on_stream(
    CuTensorExecutor& executor,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op,
    cudaStream_t stream)
{
    if (executor.operator_resident &&
        executor.resident_operator_tag == operator_tag) {
        return true;
    }

    if (!upload_cutensor_executor_operator_on_stream(executor, local_op, stream)) {
        return false;
    }

    executor.operator_resident = true;
    executor.resident_operator_tag = operator_tag;
    return true;
}

bool upload_cutensor_executor_input(
    CuTensorExecutor& executor,
    const std::vector<Complex>& input_tensor)
{
    return upload_cutensor_executor_input_on_stream(
        executor,
        input_tensor,
        executor.stream);
}

bool upload_cutensor_executor_input_on_stream(
    CuTensorExecutor& executor,
    const std::vector<Complex>& input_tensor,
    cudaStream_t stream)
{
    if (input_tensor.size() * sizeof(Complex) != executor.input_bytes) {
        return false;
    }

    cudaStream_t execution_stream = resolve_execution_stream(executor, stream);
    if (!prepare_cutensor_executor_stream(executor, execution_stream)) {
        return false;
    }

    if (!ensure_cuda_buffer(&executor.d_input, executor.input_bytes) ||
        !ensure_cuda_buffer(&executor.d_output, executor.output_bytes)) {
        return false;
    }

    if (!cuda_copy_h2d_async(executor.d_input, input_tensor.data(), executor.input_bytes, execution_stream)) {
        return false;
    }

    if (cudaMemsetAsync(executor.d_output, 0, executor.output_bytes, execution_stream) != cudaSuccess) {
        return false;
    }

    return finalize_cutensor_executor_preparation(executor, execution_stream);
}

bool upload_cutensor_executor_inputs(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor)
{
    if (!upload_cutensor_executor_operator(executor, local_op)) {
        return false;
    }

    if (!upload_cutensor_executor_input(executor, input_tensor)) {
        return false;
    }

    return true;
}

bool copy_cutensor_executor_output_to_input(
    CuTensorExecutor& src_executor,
    CuTensorExecutor& dst_executor)
{
    if (src_executor.output_bytes != dst_executor.input_bytes) {
        return false;
    }

    if (!prepare_cutensor_executor_stream(dst_executor, dst_executor.stream)) {
        return false;
    }

    if (!ensure_cuda_buffer(&dst_executor.d_input, dst_executor.input_bytes) ||
        !ensure_cuda_buffer(&dst_executor.d_output, dst_executor.output_bytes)) {
        return false;
    }

    if (!wait_for_cutensor_executor_completion(src_executor, dst_executor.stream)) {
        return false;
    }

    if (cudaMemcpyAsync(
            dst_executor.d_input,
            src_executor.d_output,
            src_executor.output_bytes,
            cudaMemcpyDeviceToDevice,
            dst_executor.stream) != cudaSuccess) {
        return false;
    }

    if (cudaMemsetAsync(
            dst_executor.d_output,
            0,
            dst_executor.output_bytes,
            dst_executor.stream) != cudaSuccess) {
        return false;
    }

    return finalize_cutensor_executor_preparation(dst_executor, dst_executor.stream);
}

bool execute_cutensor_executor_device(
    CuTensorExecutor& executor)
{
    return execute_cutensor_executor_device_impl(executor, executor.stream, true);
}

bool execute_cutensor_executor_device_on_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream)
{
    return execute_cutensor_executor_device_impl(executor, stream, true);
}

bool execute_cutensor_executor_device_no_completion(
    CuTensorExecutor& executor)
{
    return execute_cutensor_executor_device_impl(executor, executor.stream, false);
}

bool execute_cutensor_executor_device_no_completion_on_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream)
{
    return execute_cutensor_executor_device_impl(executor, stream, false);
}

bool download_cutensor_executor_output(
    CuTensorExecutor& executor,
    std::vector<Complex>& output_tensor)
{
    if (output_tensor.size() * sizeof(Complex) != executor.output_bytes) {
        return false;
    }

    if (!prepare_cutensor_executor_stream(executor, executor.stream)) {
        return false;
    }

    if (!cuda_copy_d2h_async(output_tensor.data(), executor.d_output, executor.output_bytes, executor.stream)) {
        return false;
    }

    if (cudaStreamSynchronize(executor.stream) != cudaSuccess) {
        return false;
    }

    return true;
}

bool execute_cutensor_executor(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor)
{
    if (!upload_cutensor_executor_inputs(executor, local_op, input_tensor)) {
        return false;
    }

    if (!execute_cutensor_executor_device(executor)) {
        return false;
    }

    if (!download_cutensor_executor_output(executor, output_tensor)) {
        return false;
    }

    return true;
}

bool execute_cutensor_executor_with_resident_operator(
    CuTensorExecutor& executor,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor)
{
    if (!executor.operator_resident) {
        return false;
    }

    if (!upload_cutensor_executor_input(executor, input_tensor)) {
        return false;
    }

    if (!execute_cutensor_executor_device(executor)) {
        return false;
    }

    if (!download_cutensor_executor_output(executor, output_tensor)) {
        return false;
    }

    return true;
}

bool execute_cutensor_executor_with_resident_operator_pinned(
    CuTensorExecutor& executor,
    const PinnedComplexBuffer& input_buffer,
    PinnedComplexBuffer& output_buffer)
{
    if (!executor.operator_resident) {
        return false;
    }

    if (input_buffer.size * sizeof(Complex) != executor.input_bytes) {
        return false;
    }

    if (output_buffer.size * sizeof(Complex) != executor.output_bytes) {
        return false;
    }

    if (!prepare_cutensor_executor_stream(executor, executor.stream)) {
        return false;
    }

    if (!ensure_cuda_buffer(&executor.d_input, executor.input_bytes) ||
        !ensure_cuda_buffer(&executor.d_output, executor.output_bytes) ||
        !ensure_cuda_buffer(
            &executor.d_workspace,
            static_cast<size_t>(executor.plan_bundle.workspace_size))) {
        return false;
    }

    if (!cuda_copy_h2d_async(
            executor.d_input,
            input_buffer.data,
            executor.input_bytes,
            executor.stream)) {
        return false;
    }

    if (cudaMemsetAsync(
            executor.d_output,
            0,
            executor.output_bytes,
            executor.stream) != cudaSuccess) {
        return false;
    }

    if (!execute_cutensor_executor_device(executor)) {
        return false;
    }

    if (!cuda_copy_d2h_async(
            output_buffer.data,
            executor.d_output,
            executor.output_bytes,
            executor.stream)) {
        return false;
    }

    if (cudaStreamSynchronize(executor.stream) != cudaSuccess) {
        return false;
    }

    return true;
}

bool record_cutensor_executor_completion(
    CuTensorExecutor& executor)
{
    return record_cutensor_executor_completion_on_stream(
        executor,
        executor.stream);
}

bool record_cutensor_executor_completion_on_stream(
    CuTensorExecutor& executor,
    cudaStream_t stream)
{
    cudaStream_t completion_stream = resolve_execution_stream(executor, stream);
    if (executor.completion_event == nullptr || completion_stream == nullptr) {
        return false;
    }

    if (cudaEventRecord(executor.completion_event, completion_stream) != cudaSuccess) {
        return false;
    }

    executor.completion_stream = completion_stream;
    executor.completion_recorded = true;
    return true;
}

bool wait_for_cutensor_executor_completion(
    CuTensorExecutor& producer,
    cudaStream_t consumer_stream)
{
    if (!producer.completion_recorded || producer.completion_event == nullptr) {
        return true;
    }

    if (consumer_stream == nullptr) {
        return cudaEventSynchronize(producer.completion_event) == cudaSuccess;
    }

    if (producer.completion_stream == consumer_stream) {
        return true;
    }

    return cudaStreamWaitEvent(
        consumer_stream,
        producer.completion_event,
        0) == cudaSuccess;
}

} // namespace culindblad
