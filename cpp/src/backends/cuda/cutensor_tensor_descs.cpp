#include "culindblad/cutensor_tensor_descs.hpp"

#include <cutensor.h>
#include <cuda_runtime.h>

#include <vector>

#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_ops.hpp"

namespace culindblad {

namespace {

std::vector<int64_t> make_row_major_strides(
    const std::vector<int64_t>& extents)
{
    std::vector<int64_t> strides(extents.size(), 1);

    if (extents.empty()) {
        return strides;
    }

    strides.back() = 1;
    for (Index i = extents.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * extents[i];
    }

    return strides;
}

} // namespace

bool create_cutensor_tensor_descs(
    const CuTensorContractionDesc& desc,
    CuTensorTensorDescs& tensor_descs)
{
    if (!validate_cutensor_contraction_desc(desc)) {
        return false;
    }

    const cutensorStatus_t handle_status = cutensorCreate(&tensor_descs.handle);
    if (handle_status != CUTENSOR_STATUS_SUCCESS) {
        return false;
    }

    constexpr uint32_t alignment = 256;
    constexpr cudaDataType_t data_type = CUDA_C_64F;

    const std::vector<int64_t> op_strides =
        make_row_major_strides(desc.operator_extents);
    const std::vector<int64_t> input_strides =
        make_row_major_strides(desc.input_extents);
    const std::vector<int64_t> output_strides =
        make_row_major_strides(desc.output_extents);

    const cutensorStatus_t op_status = cutensorCreateTensorDescriptor(
        tensor_descs.handle,
        &tensor_descs.op_desc,
        static_cast<uint32_t>(desc.operator_extents.size()),
        desc.operator_extents.data(),
        op_strides.data(),
        data_type,
        alignment);

    if (op_status != CUTENSOR_STATUS_SUCCESS) {
        cutensorDestroy(tensor_descs.handle);
        return false;
    }

    const cutensorStatus_t input_status = cutensorCreateTensorDescriptor(
        tensor_descs.handle,
        &tensor_descs.input_desc,
        static_cast<uint32_t>(desc.input_extents.size()),
        desc.input_extents.data(),
        input_strides.data(),
        data_type,
        alignment);

    if (input_status != CUTENSOR_STATUS_SUCCESS) {
        cutensorDestroyTensorDescriptor(tensor_descs.op_desc);
        cutensorDestroy(tensor_descs.handle);
        return false;
    }

    const cutensorStatus_t output_status = cutensorCreateTensorDescriptor(
        tensor_descs.handle,
        &tensor_descs.output_desc,
        static_cast<uint32_t>(desc.output_extents.size()),
        desc.output_extents.data(),
        output_strides.data(),
        data_type,
        alignment);

    if (output_status != CUTENSOR_STATUS_SUCCESS) {
        cutensorDestroyTensorDescriptor(tensor_descs.input_desc);
        cutensorDestroyTensorDescriptor(tensor_descs.op_desc);
        cutensorDestroy(tensor_descs.handle);
        return false;
    }

    return true;
}

bool destroy_cutensor_tensor_descs(
    CuTensorTensorDescs& tensor_descs)
{
    bool ok = true;

    if (cutensorDestroyTensorDescriptor(tensor_descs.output_desc) != CUTENSOR_STATUS_SUCCESS) {
        ok = false;
    }

    if (cutensorDestroyTensorDescriptor(tensor_descs.input_desc) != CUTENSOR_STATUS_SUCCESS) {
        ok = false;
    }

    if (cutensorDestroyTensorDescriptor(tensor_descs.op_desc) != CUTENSOR_STATUS_SUCCESS) {
        ok = false;
    }

    if (cutensorDestroy(tensor_descs.handle) != CUTENSOR_STATUS_SUCCESS) {
        ok = false;
    }

    return ok;
}

} // namespace culindblad