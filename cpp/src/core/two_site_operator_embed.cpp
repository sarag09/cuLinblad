#include <stdexcept>
#include <vector>

#include "culindblad/state_layout.hpp"
#include "culindblad/two_site_operator_embed.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> embed_two_site_operator(
    const std::vector<Complex>& local_op,
    Index dim_site_a,
    Index dim_site_b,
    Index site_a,
    Index site_b,
    const std::vector<Index>& local_dims)
{
    if (site_a >= local_dims.size() || site_b >= local_dims.size()) {
        throw std::runtime_error("embed_two_site_operator: site index out of range");
    }

    if (site_a == site_b) {
        throw std::runtime_error("embed_two_site_operator: site_a and site_b must be different");
    }

    if (local_dims[site_a] != dim_site_a) {
        throw std::runtime_error("embed_two_site_operator: dim_site_a does not match local_dims");
    }

    if (local_dims[site_b] != dim_site_b) {
        throw std::runtime_error("embed_two_site_operator: dim_site_b does not match local_dims");
    }

    const Index local_dim_total = dim_site_a * dim_site_b;
    if (local_op.size() != local_dim_total * local_dim_total) {
        throw std::runtime_error("embed_two_site_operator: local_op has wrong size");
    }

    const StateLayout layout = make_state_layout(local_dims);
    const Index D = layout.hilbert_dim;

    std::vector<Complex> full_op(D * D, Complex{0.0, 0.0});

    for (Index ket_flat = 0; ket_flat < D; ++ket_flat) {
        const std::vector<Index> ket_multi =
            unflatten_ket_index(ket_flat, layout.ket_strides, layout.local_dims);

        for (Index bra_flat = 0; bra_flat < D; ++bra_flat) {
            const std::vector<Index> bra_multi =
                unflatten_ket_index(bra_flat, layout.bra_strides, layout.local_dims);

            bool other_sites_match = true;
            for (Index site = 0; site < local_dims.size(); ++site) {
                if (site == site_a || site == site_b) {
                    continue;
                }

                if (ket_multi[site] != bra_multi[site]) {
                    other_sites_match = false;
                    break;
                }
            }

            if (!other_sites_match) {
                continue;
            }

            const Index ket_a = ket_multi[site_a];
            const Index ket_b = ket_multi[site_b];
            const Index bra_a = bra_multi[site_a];
            const Index bra_b = bra_multi[site_b];

            const Index ket_local = ket_a * dim_site_b + ket_b;
            const Index bra_local = bra_a * dim_site_b + bra_b;

            full_op[ket_flat * D + bra_flat] =
                local_op[ket_local * local_dim_total + bra_local];
        }
    }

    return full_op;
}

} // namespace culindblad
