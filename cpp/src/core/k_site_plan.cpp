#include <stdexcept>
#include <vector>

#include "culindblad/k_site_plan.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

KSitePlan make_k_site_plan(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    if (target_sites.empty()) {
        throw std::runtime_error("make_k_site_plan: target_sites is empty");
    }

    std::vector<bool> is_target(local_dims.size(), false);
    for (Index s : target_sites) {
        if (s >= local_dims.size()) {
            throw std::runtime_error("make_k_site_plan: target site out of range");
        }
        if (is_target[s]) {
            throw std::runtime_error("make_k_site_plan: duplicate target site");
        }
        is_target[s] = true;
    }

    KSitePlan plan;
    plan.target_sites = target_sites;
    plan.target_dim_product = 1;
    plan.complement_dim_product = 1;

    for (Index s : target_sites) {
        plan.target_dims.push_back(local_dims[s]);
        plan.target_dim_product *= local_dims[s];
        plan.grouped_sites.push_back(s);
    }

    for (Index s = 0; s < local_dims.size(); ++s) {
        if (!is_target[s]) {
            plan.complement_sites.push_back(s);
            plan.complement_dims.push_back(local_dims[s]);
            plan.complement_dim_product *= local_dims[s];
            plan.grouped_sites.push_back(s);
        }
    }

    plan.original_to_grouped_position.assign(local_dims.size(), 0);
    for (Index grouped_pos = 0; grouped_pos < plan.grouped_sites.size(); ++grouped_pos) {
        const Index original_site = plan.grouped_sites[grouped_pos];
        plan.original_to_grouped_position[original_site] = grouped_pos;
    }

    return plan;
}

} // namespace culindblad