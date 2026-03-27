#include "culindblad/cutensor_executor.hpp"

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "culindblad/cutensor_plan.hpp"
#include "culindblad/pinned_host_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

bool cuda_malloc_bytes(void** ptr, size_t bytes)
{
    return cudaMalloc(ptr, bytes) == cudaSuccess;
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

    if (!cuda_malloc_bytes(&executor.d_input, input_bytes)) {
        cudaFree(executor.d_op);
        cudaStreamDestroy(executor.stream);
        destroy_cutensor_plan(executor.plan_bundle);
        return false;
    }

    if (!cuda_malloc_bytes(&executor.d_output, output_bytes)) {
        cudaFree(executor.d_input);
        cudaFree(executor.d_op);
        cudaStreamDestroy(executor.stream);
        destroy_cutensor_plan(executor.plan_bundle);
        return false;
    }

    if (executor.plan_bundle.workspace_size > 0) {
        if (!cuda_malloc_bytes(
                &executor.d_workspace,
                static_cast<size_t>(executor.plan_bundle.workspace_size))) {
            cudaFree(executor.d_output);
            cudaFree(executor.d_input);
            cudaFree(executor.d_op);
            cudaStreamDestroy(executor.stream);
            destroy_cutensor_plan(executor.plan_bundle);
            return false;
        }
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

    return ok;
}

bool upload_cutensor_executor_operator(
    CuTensorExecutor& executor,
    const std::vector<Complex>& local_op)
{
    if (local_op.size() * sizeof(Complex) != executor.op_bytes) {
        return false;
    }

    if (!cuda_copy_h2d_async(executor.d_op, local_op.data(), executor.op_bytes, executor.stream)) {
        return false;
    }

    executor.operator_resident = true;
    executor.resident_operator_tag.clear();
    return true;
}

bool ensure_cutensor_executor_operator(
    CuTensorExecutor& executor,
    const std::string& operator_tag,
    const std::vector<Complex>& local_op)
{
    if (executor.operator_resident &&
        executor.resident_operator_tag == operator_tag) {
        return true;
    }

    if (!upload_cutensor_executor_operator(executor, local_op)) {
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
    if (input_tensor.size() * sizeof(Complex) != executor.input_bytes) {
        return false;
    }

    if (!cuda_copy_h2d_async(executor.d_input, input_tensor.data(), executor.input_bytes, executor.stream)) {
        return false;
    }

    if (cudaMemsetAsync(executor.d_output, 0, executor.output_bytes, executor.stream) != cudaSuccess) {
        return false;
    }

    return true;
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

    return true;
}

bool execute_cutensor_executor_device(
    CuTensorExecutor& executor)
{
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
            executor.stream);

    if (exec_status != CUTENSOR_STATUS_SUCCESS) {
        return false;
    }

    if (!record_cutensor_executor_completion(executor)) {
        return false;
    }    

    return true;
}

bool download_cutensor_executor_output(
    CuTensorExecutor& executor,
    std::vector<Complex>& output_tensor)
{
    if (output_tensor.size() * sizeof(Complex) != executor.output_bytes) {
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
    if (executor.completion_event == nullptr || executor.stream == nullptr) {
        return false;
    }

    return cudaEventRecord(executor.completion_event, executor.stream) == cudaSuccess;
}

bool wait_for_cutensor_executor_completion(
    CuTensorExecutor& producer,
    cudaStream_t consumer_stream)
{
    if (producer.completion_event == nullptr) {
        return false;
    }

    return cudaStreamWaitEvent(
        consumer_stream,
        producer.completion_event,
        0) == cudaSuccess;
}

} // namespace culindblad