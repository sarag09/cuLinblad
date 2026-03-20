#pragma once

#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

struct KSitePlan {
    std::vector<Index> target_sites;
    std::vector<Index> complement_sites;

    std::vector<Index> target_dims;
    std::vector<Index> complement_dims;

    std::vector<Index> grouped_sites;
    std::vector<Index> original_to_grouped_position;

    Index target_dim_product;
    Index complement_dim_product;
};

KSitePlan make_k_site_plan(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad