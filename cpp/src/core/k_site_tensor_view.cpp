#include "culindblad/k_site_tensor_view.hpp"

#include <vector>

#include "culindblad/k_site_plan.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

KSiteTensorView make_k_site_tensor_view(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    KSiteTensorView view;
    view.plan = make_k_site_plan(target_sites, local_dims);

    view.ket_group_dims = view.plan.target_dims;
    for (Index d : view.plan.complement_dims) {
        view.ket_group_dims.push_back(d);
    }

    view.bra_group_dims = view.plan.target_dims;
    for (Index d : view.plan.complement_dims) {
        view.bra_group_dims.push_back(d);
    }

    view.ket_grouped_sites = view.plan.grouped_sites;
    view.bra_grouped_sites = view.plan.grouped_sites;

    view.ket_original_to_grouped_position = view.plan.original_to_grouped_position;
    view.bra_original_to_grouped_position = view.plan.original_to_grouped_position;

    view.ket_target_dim = view.plan.target_dim_product;
    view.ket_complement_dim = view.plan.complement_dim_product;

    view.bra_target_dim = view.plan.target_dim_product;
    view.bra_complement_dim = view.plan.complement_dim_product;

    view.density_target_block_size =
        view.ket_target_dim * view.bra_target_dim;

    view.density_complement_block_size =
        view.ket_complement_dim * view.bra_complement_dim;

    return view;
}

} // namespace culindblad