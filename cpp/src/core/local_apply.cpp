#include <stdexcept>
#include <vector>

#include "culindblad/local_apply.hpp"
#include "culindblad/local_operator_utils.hpp"
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

void validate_k_site_operator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims)
{
    if (sites.empty()) {
        throw std::runtime_error("validate_k_site_operator: sites is empty");
    }

    std::vector<bool> seen(local_dims.size(), false);
    for (Index s : sites) {
        if (s >= local_dims.size()) {
            throw std::runtime_error("validate_k_site_operator: site out of range");
        }
        if (seen[s]) {
            throw std::runtime_error("validate_k_site_operator: duplicate site");
        }
        seen[s] = true;
    }

    const Index dim = term_local_dimension(
        OperatorTerm{TermKind::Hamiltonian, "tmp", sites, {}, 0, 0},
        local_dims);

    if (local_op.size() != dim * dim) {
        throw std::runtime_error("validate_k_site_operator: local_op has wrong size");
    }
}

} // namespace

std::vector<Complex> apply_k_site_operator_left(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    validate_k_site_operator(local_op, sites, local_dims);

    const StateLayout layout = make_state_layout(local_dims);
    const Index D = layout.hilbert_dim;
    const Index local_dim_total =
        term_local_dimension(OperatorTerm{TermKind::Hamiltonian, "tmp", sites, {}, 0, 0}, local_dims);

    if (rho.size != D * D) {
        throw std::runtime_error("apply_k_site_operator_left: rho has wrong size");
    }

    std::vector<Complex> result(D * D, Complex{0.0, 0.0});

    for (Index ket_out_flat = 0; ket_out_flat < D; ++ket_out_flat) {
        const std::vector<Index> ket_out_multi =
            unflatten_ket_index(ket_out_flat, layout.ket_strides, layout.local_dims);

        for (Index bra_flat = 0; bra_flat < D; ++bra_flat) {
            Complex accum{0.0, 0.0};

            for (Index ket_in_flat = 0; ket_in_flat < D; ++ket_in_flat) {
                const std::vector<Index> ket_in_multi =
                    unflatten_ket_index(ket_in_flat, layout.ket_strides, layout.local_dims);

                bool other_sites_match = true;
                for (Index site = 0; site < local_dims.size(); ++site) {
                    if (site_is_target(site, sites)) {
                        continue;
                    }
                    if (ket_out_multi[site] != ket_in_multi[site]) {
                        other_sites_match = false;
                        break;
                    }
                }

                if (!other_sites_match) {
                    continue;
                }

                const Index ket_out_local = flatten_target_tuple(ket_out_multi, sites, local_dims);
                const Index ket_in_local  = flatten_target_tuple(ket_in_multi,  sites, local_dims);

                const Complex a_entry =
                    local_op[ket_out_local * local_dim_total + ket_in_local];

                accum += a_entry * rho[ket_in_flat * D + bra_flat];
            }

            result[ket_out_flat * D + bra_flat] = accum;
        }
    }

    return result;
}

std::vector<Complex> apply_k_site_operator_right(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    validate_k_site_operator(local_op, sites, local_dims);

    const StateLayout layout = make_state_layout(local_dims);
    const Index D = layout.hilbert_dim;
    const Index local_dim_total =
        term_local_dimension(OperatorTerm{TermKind::Hamiltonian, "tmp", sites, {}, 0, 0}, local_dims);

    if (rho.size != D * D) {
        throw std::runtime_error("apply_k_site_operator_right: rho has wrong size");
    }

    std::vector<Complex> result(D * D, Complex{0.0, 0.0});

    for (Index ket_flat = 0; ket_flat < D; ++ket_flat) {
        for (Index bra_out_flat = 0; bra_out_flat < D; ++bra_out_flat) {
            const std::vector<Index> bra_out_multi =
                unflatten_ket_index(bra_out_flat, layout.bra_strides, layout.local_dims);

            Complex accum{0.0, 0.0};

            for (Index bra_in_flat = 0; bra_in_flat < D; ++bra_in_flat) {
                const std::vector<Index> bra_in_multi =
                    unflatten_ket_index(bra_in_flat, layout.bra_strides, layout.local_dims);

                bool other_sites_match = true;
                for (Index site = 0; site < local_dims.size(); ++site) {
                    if (site_is_target(site, sites)) {
                        continue;
                    }
                    if (bra_in_multi[site] != bra_out_multi[site]) {
                        other_sites_match = false;
                        break;
                    }
                }

                if (!other_sites_match) {
                    continue;
                }

                const Index bra_in_local  = flatten_target_tuple(bra_in_multi,  sites, local_dims);
                const Index bra_out_local = flatten_target_tuple(bra_out_multi, sites, local_dims);

                const Complex a_entry =
                    local_op[bra_in_local * local_dim_total + bra_out_local];

                accum += rho[ket_flat * D + bra_in_flat] * a_entry;
            }

            result[ket_flat * D + bra_out_flat] = accum;
        }
    }

    return result;
}

std::vector<Complex> apply_k_site_commutator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    const std::vector<Complex> left =
        apply_k_site_operator_left(local_op, sites, local_dims, rho);

    const std::vector<Complex> right =
        apply_k_site_operator_right(local_op, sites, local_dims, rho);

    if (left.size() != right.size()) {
        throw std::runtime_error("apply_k_site_commutator: left/right sizes do not match");
    }

    std::vector<Complex> result(left.size(), Complex{0.0, 0.0});
    const Complex minus_i{0.0, -1.0};

    for (Index idx = 0; idx < left.size(); ++idx) {
        result[idx] = minus_i * (left[idx] - right[idx]);
    }

    return result;
}

std::vector<Complex> apply_k_site_dissipator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    const Index local_dim_total =
        term_local_dimension(OperatorTerm{TermKind::Hamiltonian, "tmp", sites, {}, 0, 0}, local_dims);

    const std::vector<Complex> local_op_dag =
        local_conjugate_transpose(local_op, local_dim_total);

    const std::vector<Complex> local_op_dag_op =
        local_multiply_square(local_op_dag, local_op, local_dim_total);

    const std::vector<Complex> left_once =
        apply_k_site_operator_left(local_op, sites, local_dims, rho);

    ConstStateBuffer left_once_buf{left_once.data(), left_once.size()};

    const std::vector<Complex> jump =
        apply_k_site_operator_right(local_op_dag, sites, local_dims, left_once_buf);

    const std::vector<Complex> left =
        apply_k_site_operator_left(local_op_dag_op, sites, local_dims, rho);

    const std::vector<Complex> right =
        apply_k_site_operator_right(local_op_dag_op, sites, local_dims, rho);

    if (jump.size() != left.size() || jump.size() != right.size()) {
        throw std::runtime_error("apply_k_site_dissipator: term sizes do not match");
    }

    std::vector<Complex> result(jump.size(), Complex{0.0, 0.0});
    const Complex half{0.5, 0.0};

    for (Index idx = 0; idx < jump.size(); ++idx) {
        result[idx] = jump[idx] - half * (left[idx] + right[idx]);
    }

    return result;
}

} // namespace culindblad