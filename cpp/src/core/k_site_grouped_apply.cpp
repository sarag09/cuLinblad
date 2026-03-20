#include <stdexcept>
#include <vector>

#include "culindblad/k_site_block_map.hpp"
#include "culindblad/k_site_grouped_apply.hpp"
#include "culindblad/k_site_plan.hpp"
#include "culindblad/local_term_utils.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> apply_k_site_operator_left_grouped_reference(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    const KSitePlan plan = make_k_site_plan(target_sites, local_dims);
    const KSiteBlockMap block_map = make_k_site_block_map(target_sites, local_dims);
    const StateLayout layout = make_state_layout(local_dims);

    const Index D = layout.hilbert_dim;
    const Index target_dim = plan.target_dim_product;
    const Index complement_dim = plan.complement_dim_product;
    const Index local_dim_total =
        term_local_dimension(OperatorTerm{TermKind::Hamiltonian, "tmp", target_sites, {}, 0, 0},
                             local_dims);

    if (local_op.size() != local_dim_total * local_dim_total) {
        throw std::runtime_error("apply_k_site_operator_left_grouped_reference: local_op has wrong size");
    }

    if (rho.size != D * D) {
        throw std::runtime_error("apply_k_site_operator_left_grouped_reference: rho has wrong size");
    }

    std::vector<Complex> result(D * D, Complex{0.0, 0.0});

    for (Index target_out = 0; target_out < target_dim; ++target_out) {
        for (Index comp = 0; comp < complement_dim; ++comp) {
            const Index ket_out_flat =
                block_map.grouped_to_flat_ket[target_out * complement_dim + comp];

            for (Index bra_flat = 0; bra_flat < D; ++bra_flat) {
                Complex accum{0.0, 0.0};

                for (Index target_in = 0; target_in < target_dim; ++target_in) {
                    const Index ket_in_flat =
                        block_map.grouped_to_flat_ket[target_in * complement_dim + comp];

                    const Complex a_entry =
                        local_op[target_out * target_dim + target_in];

                    accum += a_entry * rho[ket_in_flat * D + bra_flat];
                }

                result[ket_out_flat * D + bra_flat] = accum;
            }
        }
    }

    return result;
}

} // namespace culindblad
