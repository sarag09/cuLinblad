#pragma once

#include <vector>

#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> apply_hamiltonian_commutator(
    const std::vector<Complex>& H,
    const std::vector<Complex>& rho,
    Index dim);

} // namespace culindblad
