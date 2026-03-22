#pragma once

#include <cutensor.h>

#include "culindblad/cutensor_contraction_desc.hpp"

namespace culindblad {

struct CuTensorTensorDescs {
    cutensorHandle_t handle;
    cutensorTensorDescriptor_t op_desc;
    cutensorTensorDescriptor_t input_desc;
    cutensorTensorDescriptor_t output_desc;
};

bool create_cutensor_tensor_descs(
    const CuTensorContractionDesc& desc,
    CuTensorTensorDescs& tensor_descs);

bool destroy_cutensor_tensor_descs(
    CuTensorTensorDescs& tensor_descs);

} // namespace culindblad
