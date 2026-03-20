#include "culindblad/k_site_contraction_desc.hpp"

#include <vector>

#include "culindblad/k_site_tensor_view.hpp"
#include "culindblad/local_term_utils.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

KSiteContractionDesc make_k_site_contraction_desc(
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims)
{
    KSiteContractionDesc desc;
    desc.view = make_k_site_tensor_view(target_sites, local_dims);

    desc.local_dim =
        term_local_dimension(
            OperatorTerm{TermKind::Hamiltonian, "tmp", target_sites, {}, 0, 0},
            local_dims);

    desc.ket_preserved_dim = desc.view.ket_complement_dim;
    desc.bra_preserved_dim = desc.view.bra_complement_dim;

    desc.left_input_dims = {
        desc.local_dim,
        desc.view.ket_target_dim,
        desc.view.ket_complement_dim,
        desc.view.bra_target_dim,
        desc.view.bra_complement_dim
    };

    desc.left_output_dims = {
        desc.view.ket_target_dim,
        desc.view.ket_complement_dim,
        desc.view.bra_target_dim,
        desc.view.bra_complement_dim
    };

    desc.right_input_dims = {
        desc.view.ket_target_dim,
        desc.view.ket_complement_dim,
        desc.view.bra_target_dim,
        desc.view.bra_complement_dim,
        desc.local_dim
    };

    desc.right_output_dims = {
        desc.view.ket_target_dim,
        desc.view.ket_complement_dim,
        desc.view.bra_target_dim,
        desc.view.bra_complement_dim
    };

    return desc;
}

} // namespace culindblad
