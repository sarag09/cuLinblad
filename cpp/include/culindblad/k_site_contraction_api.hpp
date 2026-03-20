#pragma once

#include <string>
#include <vector>

#include "culindblad/k_site_index_roles.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

enum class GroupedContractionKind {
    LeftAction,
    RightAction
};

struct GroupedContractionSpec {
    GroupedContractionKind kind;
    KSiteIndexRoles roles;

    std::string contraction_name;

    Index contracted_dim;
    Index preserved_dim;
    Index passive_full_dim;

    std::vector<Index> input_dims;
    std::vector<Index> output_dims;
};

GroupedContractionSpec make_grouped_left_contraction_spec(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

GroupedContractionSpec make_grouped_right_contraction_spec(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad
