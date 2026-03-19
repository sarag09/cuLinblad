#include <stdexcept>
#include <vector>

#include "culindblad/k_site_operator_embed.hpp"
#include "culindblad/local_term_utils.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

namespace {

bool site_is_target(Index site, const std::vector<Index>& sites)
{
    for (Index s : sites) {
        if (s == site) {
            return true;
        }
    }
    return false;
}

Index flatten_target_tuple(
    const std::vector<Index>& multi,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims)
{
    Index flat = 0;
    for (Index s : sites) {
        flat = flat * local_dims[s] + multi[s];
    }
    return flat;
}

void validate_k_site_embedding(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims)
{
    if (sites.empty()) {
        throw std::runtime_error("embed_k_site_operator: sites is empty");
    }

    std::vector<bool> seen(local_dims.size(), false);
    for (Index s : sites) {
        if (s >= local_dims.size()) {
            throw std::runtime_error("embed_k_site_operator: site out of range");
        }
        if (seen[s]) {
            throw std::runtime_error("embed_k_site_operator: duplicate site");
        }
        seen[s] = true;
    }

    const Index local_dim_total =
        term_local_dimension(OperatorTerm{TermKind::Hamiltonian, "tmp", sites, {}, 0, 0}, local_dims);

    if (local_op.size() != local_dim_total * local_dim_total) {
        throw std::runtime_error("embed_k_site_operator: local_op has wrong size");
    }
}

} // namespace

std::vector<Complex> embed_k_site_operator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims)
{
    validate_k_site_embedding(local_op, sites, local_dims);

    const StateLayout layout = make_state_layout(local_dims);
    const Index D = layout.hilbert_dim;
    const Index local_dim_total =
        term_local_dimension(OperatorTerm{TermKind::Hamiltonian, "tmp", sites, {}, 0, 0}, local_dims);

    std::vector<Complex> full_op(D * D, Complex{0.0, 0.0});

    for (Index ket_flat = 0; ket_flat < D; ++ket_flat) {
        const std::vector<Index> ket_multi =
            unflatten_ket_index(ket_flat, layout.ket_strides, layout.local_dims);

        for (Index bra_flat = 0; bra_flat < D; ++bra_flat) {
            const std::vector<Index> bra_multi =
                unflatten_ket_index(bra_flat, layout.bra_strides, layout.local_dims);

            bool other_sites_match = true;
            for (Index site = 0; site < local_dims.size(); ++site) {
                if (site_is_target(site, sites)) {
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

            const Index ket_local = flatten_target_tuple(ket_multi, sites, local_dims);
            const Index bra_local = flatten_target_tuple(bra_multi, sites, local_dims);

            full_op[ket_flat * D + bra_flat] =
                local_op[ket_local * local_dim_total + bra_local];
        }
    }

    return full_op;
}

} // namespace culindblad
