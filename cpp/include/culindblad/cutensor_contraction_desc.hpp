#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "culindblad/k_site_contraction_api.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct CuTensorContractionDesc {
    GroupedContractionSpec spec;

    std::vector<int32_t> operator_modes;
    std::vector<int32_t> input_modes;
    std::vector<int32_t> output_modes;

    std::vector<int64_t> operator_extents;
    std::vector<int64_t> input_extents;
    std::vector<int64_t> output_extents;

    std::string debug_name;
};

CuTensorContractionDesc make_cutensor_left_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

CuTensorContractionDesc make_cutensor_right_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad
