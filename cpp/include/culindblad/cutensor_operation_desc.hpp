#pragma once

#include <cutensor.h>

#include "culindblad/cutensor_tensor_descs.hpp"

namespace culindblad {

struct CuTensorOperationDesc {
    CuTensorTensorDescs tensor_descs;
    cutensorOperationDescriptor_t op_desc;
};

bool create_cutensor_left_operation_desc(
    const CuTensorContractionDesc& desc,
    CuTensorOperationDesc& op_desc_bundle);

bool destroy_cutensor_left_operation_desc(
    CuTensorOperationDesc& op_desc_bundle);

} // namespace culindblad
