#pragma once

#include <vector>

#include "culindblad/k_site_contraction_desc.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct KSiteIndexRoles {
    KSiteContractionDesc desc;

    Index contracted_target_dim;

    Index left_target_output_dim;
    Index left_complement_preserved_dim;
    Index left_bra_total_dim;

    Index right_target_output_dim;
    Index right_complement_preserved_dim;
    Index right_ket_total_dim;
};

KSiteIndexRoles make_k_site_index_roles(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims);

} // namespace culindblad
