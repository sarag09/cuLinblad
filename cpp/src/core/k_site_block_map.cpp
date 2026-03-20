#include <vector>

#include "culindblad/k_site_block_map.hpp"
#include "culindblad/k_site_plan.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

std::vector<Index> unflatten_with_dims(
    Index flat,
    const std::vector<Index>& dims)
{
    std::vector<Index> multi(dims.size(), 0);

    for (Index i = dims.size(); i-- > 0;) {
        multi[i] = flat % dims[i];
        flat /= dims[i];
    }

    return multi;
}

} // namespace

KSiteBlockMap make_k_site_block_map(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    KSiteBlockMap map;
    map.plan = make_k_site_plan(target_sites, local_dims);

    const StateLayout layout = make_state_layout(local_dims);

    const Index target_dim = map.plan.target_dim_product;
    const Index complement_dim = map.plan.complement_dim_product;

    map.grouped_to_flat_ket.resize(target_dim * complement_dim, 0);

    for (Index target_flat = 0; target_flat < target_dim; ++target_flat) {
        const std::vector<Index> target_multi =
            unflatten_with_dims(target_flat, map.plan.target_dims);

        for (Index comp_flat = 0; comp_flat < complement_dim; ++comp_flat) {
            const std::vector<Index> comp_multi =
                unflatten_with_dims(comp_flat, map.plan.complement_dims);

            std::vector<Index> full_multi(local_dims.size(), 0);

            for (Index i = 0; i < map.plan.target_sites.size(); ++i) {
                full_multi[map.plan.target_sites[i]] = target_multi[i];
            }

            for (Index i = 0; i < map.plan.complement_sites.size(); ++i) {
                full_multi[map.plan.complement_sites[i]] = comp_multi[i];
            }

            Index flat = 0;
            for (Index i = 0; i < full_multi.size(); ++i) {
                flat += full_multi[i] * layout.ket_strides[i];
            }

            map.grouped_to_flat_ket[target_flat * complement_dim + comp_flat] = flat;
        }
    }

    return map;
}

} // namespace culindblad
