#include <stdexcept>
#include <vector>

#include "culindblad/local_dims.hpp"
#include "culindblad/local_operator_embed.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> embed_one_site_operator(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims)
{
    if (target_site >= local_dims.size()) {
        throw std::runtime_error("embed_one_site_operator: target_site out of range");
    }

    if (local_dims[target_site] != local_dim) {
        throw std::runtime_error("embed_one_site_operator: local_dim does not match target site dimension");
    }

    if (local_op.size() != local_dim * local_dim) {
        throw std::runtime_error("embed_one_site_operator: local_op has wrong size");
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
                if (site == target_site) {
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

            const Index ket_local = ket_multi[target_site];
            const Index bra_local = bra_multi[target_site];

            full_op[ket_flat * D + bra_flat] =
                local_op[ket_local * local_dim + bra_local];
        }
    }

    return full_op;
}

} // namespace culindblad
