#pragma once

#include <vector>

#include "culindblad/state_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> apply_k_site_operator_left_grouped_reference(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_k_site_operator_right_grouped_reference(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_k_site_commutator_grouped_reference(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_k_site_dissipator_grouped_reference(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

} // namespace culindblad