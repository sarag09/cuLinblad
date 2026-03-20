#include "culindblad/k_site_index_roles.hpp"

#include "culindblad/k_site_contraction_desc.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

KSiteIndexRoles make_k_site_index_roles(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    KSiteIndexRoles roles;
    roles.desc = make_k_site_contraction_desc(target_sites, local_dims);

    roles.contracted_target_dim = roles.desc.local_dim;

    roles.left_target_output_dim = roles.desc.view.ket_target_dim;
    roles.left_complement_preserved_dim = roles.desc.view.ket_complement_dim;
    roles.left_bra_total_dim =
        roles.desc.view.bra_target_dim * roles.desc.view.bra_complement_dim;

    roles.right_target_output_dim = roles.desc.view.bra_target_dim;
    roles.right_complement_preserved_dim = roles.desc.view.bra_complement_dim;
    roles.right_ket_total_dim =
        roles.desc.view.ket_target_dim * roles.desc.view.ket_complement_dim;

    return roles;
}

} // namespace culindblad
