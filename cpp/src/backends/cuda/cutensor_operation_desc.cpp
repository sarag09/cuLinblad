#include "culindblad/cutensor_operation_desc.hpp"

#include <cutensor.h>

#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/cutensor_tensor_descs.hpp"
#include "culindblad/k_site_contraction_api.hpp"

namespace culindblad {

bool create_cutensor_operation_desc(
    const CuTensorContractionDesc& desc,
    CuTensorOperationDesc& op_desc_bundle)
{
    if (!create_cutensor_tensor_descs(desc, op_desc_bundle.tensor_descs)) {
        return false;
    }

    const cutensorStatus_t op_status = cutensorCreateContraction(
        op_desc_bundle.tensor_descs.handle,
        &op_desc_bundle.op_desc,
        op_desc_bundle.tensor_descs.op_desc,
        desc.operator_modes.data(),
        CUTENSOR_OP_IDENTITY,
        op_desc_bundle.tensor_descs.input_desc,
        desc.input_modes.data(),
        CUTENSOR_OP_IDENTITY,
        op_desc_bundle.tensor_descs.output_desc,
        desc.output_modes.data(),
        CUTENSOR_OP_IDENTITY,
        op_desc_bundle.tensor_descs.output_desc,
        desc.output_modes.data(),
        CUTENSOR_COMPUTE_DESC_64F);

    if (op_status != CUTENSOR_STATUS_SUCCESS) {
        destroy_cutensor_tensor_descs(op_desc_bundle.tensor_descs);
        return false;
    }

    return true;
}

bool destroy_cutensor_operation_desc(
    CuTensorOperationDesc& op_desc_bundle)
{
    bool ok = true;

    if (cutensorDestroyOperationDescriptor(op_desc_bundle.op_desc) != CUTENSOR_STATUS_SUCCESS) {
        ok = false;
    }

    if (!destroy_cutensor_tensor_descs(op_desc_bundle.tensor_descs)) {
        ok = false;
    }

    return ok;
}

} // namespace culindblad