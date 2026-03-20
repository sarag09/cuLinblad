#pragma once

#include <vector>

#include "culindblad/k_site_plan.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct KSiteTensorView {
    KSitePlan plan;

    std::vector<Index> ket_group_dims;
    std::vector<Index> bra_group_dims;

    std::vector<Index> ket_grouped_sites;
    std::vector<Index> bra_grouped_sites;

    std::vector<Index> ket_original_to_grouped_position;
    std::vector<Index> bra_original_to_grouped_position;

    Index ket_target_dim;
    Index ket_complement_dim;
    Index bra_target_dim;
    Index bra_complement_dim;

    Index density_target_block_size;
    Index density_complement_block_size;
};

KSiteTensorView make_k_site_tensor_view(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad