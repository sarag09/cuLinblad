#include "culindblad/cutensor_ops.hpp"

#include <cutensor.h>

#include "culindblad/cutensor_contraction_desc.hpp"

namespace culindblad {

bool validate_cutensor_contraction_desc(
    const CuTensorContractionDesc& desc)
{
    if (desc.operator_modes.size() != desc.operator_extents.size()) {
        return false;
    }

    if (desc.input_modes.size() != desc.input_extents.size()) {
        return false;
    }

    if (desc.output_modes.size() != desc.output_extents.size()) {
        return false;
    }

    if (desc.operator_modes.empty()) {
        return false;
    }

    if (desc.input_modes.empty()) {
        return false;
    }

    if (desc.output_modes.empty()) {
        return false;
    }

    return true;
}

bool initialize_cutensor_handle_for_desc(
    const CuTensorContractionDesc& desc)
{
    if (!validate_cutensor_contraction_desc(desc)) {
        return false;
    }

    cutensorHandle_t handle;
    const cutensorStatus_t status = cutensorCreate(&handle);

    if (status != CUTENSOR_STATUS_SUCCESS) {
        return false;
    }

    return true;
}

} // namespace culindblad
