#include "culindblad/grouped_state_layout.hpp"

#include <stdexcept>
#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

namespace {

std::vector<Index> make_complement_sites(
    const std::vector<Index>& target_sites,
    Index num_sites)
{
    std::vector<bool> is_target(num_sites, false);
    for (Index s : target_sites) {
        if (s >= num_sites) {
            throw std::runtime_error("make_complement_sites: target site out of range");
        }
        is_target[s] = true;
    }

    std::vector<Index> complement_sites;
    for (Index s = 0; s < num_sites; ++s) {
        if (!is_target[s]) {
            complement_sites.push_back(s);
        }
    }

    return complement_sites;
}

Index product_of_dims(
    const std::vector<Index>& dims,
    const std::vector<Index>& sites)
{
    Index prod = 1;
    for (Index s : sites) {
        prod *= dims[s];
    }
    return prod;
}

std::vector<Index> unflatten(
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

Index flatten_subset(
    const std::vector<Index>& full_multi,
    const std::vector<Index>& subset_sites,
    const std::vector<Index>& dims)
{
    Index flat = 0;
    for (Index s : subset_sites) {
        flat = flat * dims[s] + full_multi[s];
    }
    return flat;
}

} // namespace

GroupedStateLayout make_grouped_state_layout(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    GroupedStateLayout layout;
    layout.target_sites = target_sites;
    layout.local_dims = local_dims;

    const Index num_sites = local_dims.size();
    const std::vector<Index> complement_sites =
        make_complement_sites(target_sites, num_sites);

    layout.hilbert_dim = 1;
    for (Index d : local_dims) {
        layout.hilbert_dim *= d;
    }

    const Index target_dim = product_of_dims(local_dims, target_sites);
    const Index complement_dim = product_of_dims(local_dims, complement_sites);

    layout.grouped_size =
        target_dim * complement_dim * target_dim * complement_dim;

    layout.flat_density_to_grouped.resize(layout.hilbert_dim * layout.hilbert_dim);
    layout.grouped_to_flat_density.resize(layout.grouped_size);

    for (Index ket_flat = 0; ket_flat < layout.hilbert_dim; ++ket_flat) {
        const std::vector<Index> ket_multi = unflatten(ket_flat, local_dims);
        const Index ket_target = flatten_subset(ket_multi, target_sites, local_dims);
        const Index ket_comp = flatten_subset(ket_multi, complement_sites, local_dims);

        for (Index bra_flat = 0; bra_flat < layout.hilbert_dim; ++bra_flat) {
            const std::vector<Index> bra_multi = unflatten(bra_flat, local_dims);
            const Index bra_target = flatten_subset(bra_multi, target_sites, local_dims);
            const Index bra_comp = flatten_subset(bra_multi, complement_sites, local_dims);

            const Index flat_density_idx =
                ket_flat * layout.hilbert_dim + bra_flat;

            const Index grouped_idx =
                ((ket_target * complement_dim + ket_comp) * target_dim + bra_target)
                * complement_dim + bra_comp;

            layout.flat_density_to_grouped[flat_density_idx] = grouped_idx;
            layout.grouped_to_flat_density[grouped_idx] = flat_density_idx;
        }
    }

    return layout;
}

void regroup_flat_density_to_grouped(
    const GroupedStateLayout& layout,
    const std::vector<Complex>& flat_density,
    std::vector<Complex>& grouped_density)
{
    if (flat_density.size() != layout.flat_density_to_grouped.size()) {
        throw std::runtime_error("regroup_flat_density_to_grouped: flat density size mismatch");
    }

    if (grouped_density.size() != layout.grouped_size) {
        throw std::runtime_error("regroup_flat_density_to_grouped: grouped density size mismatch");
    }

    for (Index flat_idx = 0; flat_idx < flat_density.size(); ++flat_idx) {
        grouped_density[layout.flat_density_to_grouped[flat_idx]] = flat_density[flat_idx];
    }
}

void regroup_grouped_to_flat_density(
    const GroupedStateLayout& layout,
    const std::vector<Complex>& grouped_density,
    std::vector<Complex>& flat_density)
{
    if (grouped_density.size() != layout.grouped_size) {
        throw std::runtime_error("regroup_grouped_to_flat_density: grouped density size mismatch");
    }

    if (flat_density.size() != layout.flat_density_to_grouped.size()) {
        throw std::runtime_error("regroup_grouped_to_flat_density: flat density size mismatch");
    }

    for (Index grouped_idx = 0; grouped_idx < grouped_density.size(); ++grouped_idx) {
        flat_density[layout.grouped_to_flat_density[grouped_idx]] = grouped_density[grouped_idx];
    }
}

} // namespace culindblad
