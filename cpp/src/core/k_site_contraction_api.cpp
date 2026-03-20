#include "culindblad/k_site_contraction_api.hpp"

#include <string>
#include <vector>

#include "culindblad/k_site_index_roles.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

GroupedContractionSpec make_grouped_left_contraction_spec(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    GroupedContractionSpec spec;
    spec.kind = GroupedContractionKind::LeftAction;
    spec.roles = make_k_site_index_roles(target_sites, local_dims);

    spec.contraction_name = "grouped_left_action";

    spec.contracted_dim = spec.roles.contracted_target_dim;
    spec.preserved_dim = spec.roles.left_complement_preserved_dim;
    spec.passive_full_dim = spec.roles.left_bra_total_dim;

    spec.input_dims = spec.roles.desc.left_input_dims;
    spec.output_dims = spec.roles.desc.left_output_dims;

    return spec;
}

GroupedContractionSpec make_grouped_right_contraction_spec(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    GroupedContractionSpec spec;
    spec.kind = GroupedContractionKind::RightAction;
    spec.roles = make_k_site_index_roles(target_sites, local_dims);

    spec.contraction_name = "grouped_right_action";

    spec.contracted_dim = spec.roles.contracted_target_dim;
    spec.preserved_dim = spec.roles.right_complement_preserved_dim;
    spec.passive_full_dim = spec.roles.right_ket_total_dim;

    spec.input_dims = spec.roles.desc.right_input_dims;
    spec.output_dims = spec.roles.desc.right_output_dims;

    return spec;
}

} // namespace culindblad
