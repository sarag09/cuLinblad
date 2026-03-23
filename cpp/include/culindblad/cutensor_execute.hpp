#pragma once

#include <vector>

#include "culindblad/cutensor_contraction_desc.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

bool execute_cutensor_left_contraction(
    const CuTensorContractionDesc& desc,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor);

bool execute_cutensor_right_contraction(
    const CuTensorContractionDesc& desc,
    const std::vector<Complex>& local_op,
    const std::vector<Complex>& input_tensor,
    std::vector<Complex>& output_tensor);

} // namespace culindblad