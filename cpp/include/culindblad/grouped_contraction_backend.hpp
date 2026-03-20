#pragma once

#include <vector>

#include "culindblad/state_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> apply_grouped_left_contraction(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_grouped_right_contraction(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_grouped_commutator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_grouped_dissipator(
    const std::vector<Complex>& local_op,
    const std::vector<Index>& target_sites,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

} // namespace culindblad