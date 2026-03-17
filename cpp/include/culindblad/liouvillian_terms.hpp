#pragma once

#include "culindblad/state_buffer.hpp"
#include "culindblad/types.hpp"

#include <vector>

namespace culindblad {

std::vector<Complex> apply_hamiltonian_commutator(
    const std::vector<Complex>& H,
    ConstStateBuffer rho,
    Index dim);

std::vector<Complex> apply_dissipator(
    const std::vector<Complex>& L,
    ConstStateBuffer rho,
    Index dim);

} // namespace culindblad