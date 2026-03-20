#include "culindblad/grouped_contraction_backend.hpp"

#include <vector>

#include "culindblad/k_site_grouped_apply.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> apply_grouped_left_contraction(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    return apply_k_site_operator_left_grouped_reference(
        local_op,
        target_sites,
        local_dims,
        rho);
}

std::vector<Complex> apply_grouped_right_contraction(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    return apply_k_site_operator_right_grouped_reference(
        local_op,
        target_sites,
        local_dims,
        rho);
}

std::vector<Complex> apply_grouped_commutator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    return apply_k_site_commutator_grouped_reference(
        local_op,
        target_sites,
        local_dims,
        rho);
}

std::vector<Complex> apply_grouped_dissipator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho)
{
    return apply_k_site_dissipator_grouped_reference(
        local_op,
        target_sites,
        local_dims,
        rho);
}

} // namespace culindblad