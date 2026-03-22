#include "culindblad/cutensor_tensor_descs.hpp"

#include <cutensor.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_ops.hpp"

namespace culindblad {

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

    const cutensorStatus_t op_status = cutensorCreateTensorDescriptor(
        tensor_descs.handle,
        &tensor_descs.op_desc,
        static_cast<uint32_t>(desc.operator_extents.size()),
        desc.operator_extents.data(),
        nullptr,
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
        nullptr,
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
        nullptr,
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
