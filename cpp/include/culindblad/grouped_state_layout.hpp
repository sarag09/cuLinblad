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

// build_host_tables: when false, skip allocating/filling the O(D^2) host permutation
// tables (flat_density_to_grouped / grouped_to_flat_density). The GPU path computes the
// relayout arithmetically from a compact descriptor and never reads those tables, so for
// GPU target runs they are pure waste (~2*D^2*sizeof(Index) per layout; ~25 GB host at
// 4Q4C over 36 layouts, plus the O(D^2) fill cost at startup). Defaults to true so the
// CPU reference (which uses regroup_*_density) is unaffected.
GroupedStateLayout make_grouped_state_layout(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    bool build_host_tables = true);

void regroup_flat_density_to_grouped(
    const GroupedStateLayout& layout,
    const std::vector<Complex>& flat_density,
    std::vector<Complex>& grouped_density);

void regroup_grouped_to_flat_density(
    const GroupedStateLayout& layout,
    const std::vector<Complex>& grouped_density,
    std::vector<Complex>& flat_density);

} // namespace culindblad
