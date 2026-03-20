#pragma once

#include <vector>

#include "culindblad/k_site_plan.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct KSiteBlockMap {
    KSitePlan plan;

    std::vector<Index> grouped_to_flat_ket;
};

KSiteBlockMap make_k_site_block_map(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad
