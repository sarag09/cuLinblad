#pragma once

#include <vector>

#include "culindblad/k_site_tensor_view.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct KSiteContractionDesc {
    KSiteTensorView view;

    Index local_dim;
    Index ket_preserved_dim;
    Index bra_preserved_dim;

    std::vector<Index> left_input_dims;
    std::vector<Index> left_output_dims;

    std::vector<Index> right_input_dims;
    std::vector<Index> right_output_dims;
};

KSiteContractionDesc make_k_site_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad
