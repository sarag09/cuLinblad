#pragma once

#include <vector>

#include "culindblad/state_buffer.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> apply_one_site_operator_left(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_one_site_operator_right(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

std::vector<Complex> apply_one_site_commutator(
    const std::vector<Complex>& local_op,
    Index local_dim,
    Index target_site,
    const std::vector<Index>& local_dims,
    ConstStateBuffer rho);

} // namespace culindblad