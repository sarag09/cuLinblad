#include "culindblad/cutensor_execute.hpp"

#include <cuda_runtime.h>
#include <cutensor.h>

#include <vector>

#include "culindblad/cutensor_operation_desc.hpp"
#include "culindblad/cutensor_plan.hpp"
#include "culindblad/cutensor_tensor_descs.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

bool cuda_malloc_bytes(void** ptr, size_t bytes)
{
    return cudaMalloc(ptr, bytes) == cudaSuccess;
}

bool cuda_copy_h2d(void* dst, const void* src, size_t bytes)
{
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool cuda_copy_d2h(void* dst, const void* src, size_t bytes)
{
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool execute_cutensor_contraction_impl(
    const CuTensorContractionDesc& desc,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor)
{
    CuTensorPlanBundle plan_bundle{};
    if (!create_cutensor_plan(desc, plan_bundle)) {
        return false;
    }

    const size_t op_bytes = local_op.size() * sizeof(Complex);
    const size_t input_bytes = input_tensor.size() * sizeof(Complex);
    const size_t output_bytes = output_tensor.size() * sizeof(Complex);

    void* d_op = nullptr;
    void* d_input = nullptr;
    void* d_output = nullptr;
    void* d_workspace = nullptr;

    if (!cuda_malloc_bytes(&d_op, op_bytes)) {
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    if (!cuda_malloc_bytes(&d_input, input_bytes)) {
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    if (!cuda_malloc_bytes(&d_output, output_bytes)) {
        cudaFree(d_input);
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    if (plan_bundle.workspace_size > 0) {
        if (!cuda_malloc_bytes(&d_workspace, static_cast<size_t>(plan_bundle.workspace_size))) {
            cudaFree(d_output);
            cudaFree(d_input);
            cudaFree(d_op);
            destroy_cutensor_plan(plan_bundle);
            return false;
        }
    }

    if (!cuda_copy_h2d(d_op, local_op.data(), op_bytes)) {
        cudaFree(d_workspace);
        cudaFree(d_output);
        cudaFree(d_input);
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    if (!cuda_copy_h2d(d_input, input_tensor.data(), input_bytes)) {
        cudaFree(d_workspace);
        cudaFree(d_output);
        cudaFree(d_input);
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    if (cudaMemset(d_output, 0, output_bytes) != cudaSuccess) {
        cudaFree(d_workspace);
        cudaFree(d_output);
        cudaFree(d_input);
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};

    const cutensorStatus_t exec_status =
        cutensorContract(
            plan_bundle.op_bundle.tensor_descs.handle,
            plan_bundle.plan,
            reinterpret_cast<const void*>(&alpha),
            d_op,
            d_input,
            reinterpret_cast<const void*>(&beta),
            d_output,
            d_output,
            d_workspace,
            plan_bundle.workspace_size,
            0);

    if (exec_status != CUTENSOR_STATUS_SUCCESS) {
        cudaFree(d_workspace);
        cudaFree(d_output);
        cudaFree(d_input);
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_workspace);
        cudaFree(d_output);
        cudaFree(d_input);
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    if (!cuda_copy_d2h(output_tensor.data(), d_output, output_bytes)) {
        cudaFree(d_workspace);
        cudaFree(d_output);
        cudaFree(d_input);
        cudaFree(d_op);
        destroy_cutensor_plan(plan_bundle);
        return false;
    }

    cudaFree(d_workspace);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(d_op);
    destroy_cutensor_plan(plan_bundle);

    return true;
}

} // namespace

bool execute_cutensor_left_contraction(
    const CuTensorContractionDesc& desc,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor)
{
    return execute_cutensor_contraction_impl(
        desc,
        local_op,
        input_tensor,
        output_tensor);
}

bool execute_cutensor_right_contraction(
    const CuTensorContractionDesc& desc,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor)
{
    return execute_cutensor_contraction_impl(
        desc,
        local_op,
        input_tensor,
        output_tensor);
}

} // namespace culindblad