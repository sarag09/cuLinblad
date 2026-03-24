#pragma once

#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

struct GroupedStateLayout {
    std::vector<Index> target_sites;
    std::vector<Index> local_dims;

    Index hilbert_dim;
    Index grouped_size;

    std::vector<Index> flat_density_to_grouped;
    std::vector<Index> grouped_to_flat_density;
};

GroupedStateLayout make_grouped_state_layout(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

void regroup_flat_density_to_grouped(
    const GroupedStateLayout& layout,
    const std::vector<Complex>& flat_density,
    std::vector<Complex>& grouped_density);

void regroup_grouped_to_flat_density(
    const GroupedStateLayout& layout,
    const std::vector<Complex>& grouped_density,
    std::vector<Complex>& flat_density);

} // namespace culindblad
