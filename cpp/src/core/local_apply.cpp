#include <stdexcept>
#include <vector>

#include "culindblad/local_apply.hpp"
#include "culindblad/local_dims.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/types.hpp"
#include "culindblad/local_operator_utils.hpp"

namespace culindblad {

std::vector<Complex> apply_one_site_operator_left(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    if (target_site >= local_dims.size()) {
        throw std::runtime_error("apply_one_site_operator_left: target_site out of range");
    }

    if (local_dims[target_site] != local_dim) {
        throw std::runtime_error("apply_one_site_operator_left: local_dim does not match target site dimension");
    }

    if (local_op.size() != local_dim * local_dim) {
        throw std::runtime_error("apply_one_site_operator_left: local_op has wrong size");
    }

    const StateLayout layout = make_state_layout(local_dims);
    const Index D = layout.hilbert_dim;

    if (rho.size != D * D) {
        throw std::runtime_error("apply_one_site_operator_left: rho has wrong size");
    }

    std::vector<Complex> result(D * D, Complex{0.0, 0.0});

    for (Index ket_out_flat = 0; ket_out_flat < D; ++ket_out_flat) {
        const std::vector<Index> ket_out_multi =
            unflatten_ket_index(ket_out_flat, layout.ket_strides, layout.local_dims);

        for (Index bra_flat = 0; bra_flat < D; ++bra_flat) {
            Complex accum{0.0, 0.0};

            const std::vector<Index> bra_multi =
                unflatten_ket_index(bra_flat, layout.bra_strides, layout.local_dims);

            for (Index ket_in_flat = 0; ket_in_flat < D; ++ket_in_flat) {
                const std::vector<Index> ket_in_multi =
                    unflatten_ket_index(ket_in_flat, layout.ket_strides, layout.local_dims);

                bool other_sites_match = true;
                for (Index site = 0; site < local_dims.size(); ++site) {
                    if (site == target_site) {
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

                const Index ket_out_local = ket_out_multi[target_site];
                const Index ket_in_local = ket_in_multi[target_site];

                const Complex a_entry =
                    local_op[ket_out_local * local_dim + ket_in_local];

                accum += a_entry * rho[ket_in_flat * D + bra_flat];
            }

            result[ket_out_flat * D + bra_flat] = accum;
        }
    }

    return result;
}

std::vector<Complex> apply_one_site_operator_right(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    if (target_site >= local_dims.size()) {
        throw std::runtime_error("apply_one_site_operator_right: target_site out of range");
    }

    if (local_dims[target_site] != local_dim) {
        throw std::runtime_error("apply_one_site_operator_right: local_dim does not match target site dimension");
    }

    if (local_op.size() != local_dim * local_dim) {
        throw std::runtime_error("apply_one_site_operator_right: local_op has wrong size");
    }

    const StateLayout layout = make_state_layout(local_dims);
    const Index D = layout.hilbert_dim;

    if (rho.size != D * D) {
        throw std::runtime_error("apply_one_site_operator_right: rho has wrong size");
    }

    std::vector<Complex> result(D * D, Complex{0.0, 0.0});

    for (Index ket_flat = 0; ket_flat < D; ++ket_flat) {
        const std::vector<Index> ket_multi =
            unflatten_ket_index(ket_flat, layout.ket_strides, layout.local_dims);

        for (Index bra_out_flat = 0; bra_out_flat < D; ++bra_out_flat) {
            const std::vector<Index> bra_out_multi =
                unflatten_ket_index(bra_out_flat, layout.bra_strides, layout.local_dims);

            Complex accum{0.0, 0.0};

            for (Index bra_in_flat = 0; bra_in_flat < D; ++bra_in_flat) {
                const std::vector<Index> bra_in_multi =
                    unflatten_ket_index(bra_in_flat, layout.bra_strides, layout.local_dims);

                bool other_sites_match = true;
                for (Index site = 0; site < local_dims.size(); ++site) {
                    if (site == target_site) {
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

                const Index bra_in_local = bra_in_multi[target_site];
                const Index bra_out_local = bra_out_multi[target_site];

                const Complex a_entry =
                    local_op[bra_in_local * local_dim + bra_out_local];

                accum += rho[ket_flat * D + bra_in_flat] * a_entry;
            }

            result[ket_flat * D + bra_out_flat] = accum;
        }
    }

    return result;
}

std::vector<Complex> apply_one_site_commutator(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    const std::vector<Complex> left =
        apply_one_site_operator_left(local_op, local_dim, target_site, local_dims, rho);

    const std::vector<Complex> right =
        apply_one_site_operator_right(local_op, local_dim, target_site, local_dims, rho);

    if (left.size() != right.size()) {
        throw std::runtime_error("apply_one_site_commutator: left/right sizes do not match");
    }

    std::vector<Complex> result(left.size(), Complex{0.0, 0.0});
    const Complex minus_i{0.0, -1.0};

    for (Index idx = 0; idx < left.size(); ++idx) {
        result[idx] = minus_i * (left[idx] - right[idx]);
    }

    return result;
}

std::vector<Complex> apply_one_site_dissipator(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    const std::vector<Complex> local_op_dag =
        local_conjugate_transpose(local_op, local_dim);

    const std::vector<Complex> local_op_dag_op =
        local_multiply_square(local_op_dag, local_op, local_dim);

    const std::vector<Complex> left_once =
        apply_one_site_operator_left(local_op, local_dim, target_site, local_dims, rho);

    ConstStateBuffer left_once_buf{left_once.data(), left_once.size()};

    const std::vector<Complex> jump =
        apply_one_site_operator_right(
            local_op_dag, local_dim, target_site, local_dims, left_once_buf);

    const std::vector<Complex> left =
        apply_one_site_operator_left(
            local_op_dag_op, local_dim, target_site, local_dims, rho);

    const std::vector<Complex> right =
        apply_one_site_operator_right(
            local_op_dag_op, local_dim, target_site, local_dims, rho);

    if (jump.size() != left.size() || jump.size() != right.size()) {
        throw std::runtime_error("apply_one_site_dissipator: term sizes do not match");
    }

    std::vector<Complex> result(jump.size(), Complex{0.0, 0.0});
    const Complex half{0.5, 0.0};

    for (Index idx = 0; idx < jump.size(); ++idx) {
        result[idx] = jump[idx] - half * (left[idx] + right[idx]);
    }

    return result;
}

} // namespace culindblad
