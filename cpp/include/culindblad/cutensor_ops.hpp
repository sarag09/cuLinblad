#pragma once

#include "culindblad/cutensor_contraction_desc.hpp"

namespace culindblad {

bool validate_cutensor_contraction_desc(
    const CuTensorContractionDesc& desc);

bool initialize_cutensor_handle_for_desc(
    const CuTensorContractionDesc& desc);

} // namespace culindblad
